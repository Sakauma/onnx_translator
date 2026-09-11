# /**
#   ******************************************************************************
#   * @file        test_post_push_layernorm_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       回归验证 LayerNormalization stash_type 精度与可选输出槽位语义。
#   * @details     2026.09.11  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from pathlib import Path

from onnx.reference import ReferenceEvaluator

from operator_test_context import *  # noqa: F401,F403
from tools.numerical import cuda as cuda_runner


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"


def _tensor(values, dtype="float32"):
    values = np.asarray(values)
    return Tensor(*values.shape, dtype=dtype, data=values)


def _float32_layer_norm(x, epsilon):
    work = np.asarray(x, dtype=np.float32).reshape(x.shape[0], -1)
    total = np.zeros((work.shape[0], 1), dtype=np.float32)
    for col in range(work.shape[1]):
        total = np.asarray(total + work[:, col:col + 1], dtype=np.float32)
    mean = np.asarray(total / np.float32(work.shape[1]), dtype=np.float32)
    diff = np.asarray(work - mean, dtype=np.float32)
    square_sum = np.zeros_like(mean)
    for col in range(work.shape[1]):
        square = np.asarray(diff[:, col:col + 1] * diff[:, col:col + 1], dtype=np.float32)
        square_sum = np.asarray(square_sum + square, dtype=np.float32)
    variance = np.asarray(square_sum / np.float32(work.shape[1]), dtype=np.float32)
    inv_std = np.asarray(
        np.float32(1.0) / np.asarray(np.sqrt(np.asarray(variance + np.float32(epsilon), dtype=np.float32)), dtype=np.float32),
        dtype=np.float32,
    )
    y = np.asarray(diff * inv_std, dtype=np.float32).reshape(x.shape)
    reduction_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return y, mean.reshape(reduction_shape), inv_std.reshape(reduction_shape)


def _layer_norm_model(x_shape, outputs=("y",), axis=-1, epsilon=1e-5, input_proto=TensorProto.FLOAT):
    output_infos = [helper.make_tensor_value_info("y", input_proto, list(x_shape))]
    reduction_shape = list(x_shape)
    normalized_axis = axis if axis >= 0 else axis + len(x_shape)
    for idx in range(normalized_axis, len(reduction_shape)):
        reduction_shape[idx] = 1
    if len(outputs) > 1 and outputs[1]:
        output_infos.append(helper.make_tensor_value_info(outputs[1], TensorProto.FLOAT, reduction_shape))
    if len(outputs) > 2 and outputs[2]:
        output_infos.append(helper.make_tensor_value_info(outputs[2], TensorProto.FLOAT, reduction_shape))
    graph = helper.make_graph(
        [helper.make_node("LayerNormalization", ["x", "scale", "bias"], list(outputs), axis=axis, epsilon=epsilon, stash_type=1)],
        "layer_norm_stash_float32",
        [
            helper.make_tensor_value_info("x", input_proto, list(x_shape)),
            helper.make_tensor_value_info("scale", input_proto, list(x_shape[normalized_axis:])),
            helper.make_tensor_value_info("bias", input_proto, list(x_shape[normalized_axis:])),
        ],
        output_infos,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _spy_c_symbol(monkeypatch, op, symbol_name):
    original = getattr(op.lib, symbol_name)
    calls = []

    def wrapper(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(op.lib, symbol_name, wrapper)
    return calls


def _run_layer_norm_cuda(monkeypatch, x, scale, bias, emit_stats, input_proto):
    executable = CACHE / "verify_layer_normalization"
    if not executable.exists():
        pytest.skip(f"CUDA verifier is not compiled: {executable}")
    monkeypatch.setattr(cuda_runner, "CUDA_VERIFY_DIR", str(CACHE))
    params = (
        np.array([x.shape[0], x.shape[-1], 1, 1, emit_stats, 1, input_proto], dtype=np.int32).tobytes()
        + np.array([1e-5], dtype=np.float32).tobytes()
    )
    sidecars = None
    if emit_stats:
        sidecars = [
            cuda_runner.CudaSidecarSpec("tmp_layer_norm_mean.bin", np.float64, (x.shape[0], 1)),
            cuda_runner.CudaSidecarSpec("tmp_layer_norm_inv_std.bin", np.float64, (x.shape[0], 1)),
        ]
    try:
        return cuda_runner.run_cuda_ground_truth(
            "layer_normalization",
            [x.astype(np.float64), scale.astype(np.float64), bias.astype(np.float64)],
            params_binary=params,
            output_dtype=np.float64,
            target_shape=x.shape,
            sidecars=sidecars,
        )
    except cuda_runner.CudaVerifierError as exc:
        unavailable = (
            "no CUDA-capable device",
            "CUDA driver version is insufficient",
            "initialization error",
        )
        if exc.stderr and any(message in exc.stderr for message in unavailable):
            pytest.skip(exc.stderr)
        raise


def _layer_norm_two_stage_formula(x, scale, bias):
    normalized, mean, inv_std = _float32_layer_norm(x, 1e-5)
    normalized_t = normalized.astype(x.dtype)
    y = (normalized_t * scale).astype(x.dtype)
    y = (y + bias).astype(x.dtype)
    return y, mean, inv_std


@pytest.mark.parametrize(
    "x,expected_bits",
    [
        (np.array([[1.0e8, 1.0e8, 1.0e8 + 8.0]], dtype=np.float32), np.array([0, 0, 1071494101], dtype=np.uint32)),
        (np.array([[1.0, 2.0, 3.0]], dtype=np.float32), np.array([3214722083, 0, 1067238435], dtype=np.uint32)),
    ],
)
def test_strict_imported_layer_norm_uses_float32_stash_and_new_c_symbol(monkeypatch, tmp_path, x, expected_bits):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")
    model = _layer_norm_model(x.shape)
    onnx.checker.check_model(model, full_check=True)
    model_path = tmp_path / "layer_norm_stash_float32.onnx"
    onnx.save(model, model_path)
    op = next(item for item in ONNXImport(str(model_path), strict=True) if isinstance(item, LayerNormalization))
    calls = _spy_c_symbol(monkeypatch, op, "layer_norm_float_stash_forward")
    scale = np.ones(x.shape[-1], dtype=np.float32)
    bias = np.zeros(x.shape[-1], dtype=np.float32)
    actual = op.forward(_tensor(x), _tensor(scale), _tensor(bias))["tensor"].data
    reference = ReferenceEvaluator(model).run(None, {"x": x, "scale": scale, "bias": bias})[0]
    independent = _float32_layer_norm(x, 1e-5)[0]
    assert len(calls) == 1
    np.testing.assert_array_equal(actual.view(np.uint32), expected_bits.reshape(x.shape))
    np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
    np.testing.assert_array_equal(actual.view(np.uint32), independent.view(np.uint32))


def test_layer_norm_stats_use_float32_stash_shapes_and_multi_output_symbol(monkeypatch):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")
    x = np.array([[1.0e8, 1.0e8, 1.0e8 + 8.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    op = LayerNormalization(["x", "scale", "bias"], ["y", "mean", "inv"], stash_type=1, dtype="float32")
    calls = _spy_c_symbol(monkeypatch, op, "layer_norm_float_stash_multi_output_forward")
    actual_y, actual_mean, actual_inv = op.forward(
        _tensor(x), _tensor(np.ones(3, dtype=np.float32)), _tensor(np.zeros(3, dtype=np.float32))
    )["tensor"]
    expected_y, expected_mean, expected_inv = _float32_layer_norm(x, 1e-5)
    assert len(calls) == 1
    assert actual_mean.dtype == actual_inv.dtype == "float32"
    assert actual_mean.size == actual_inv.size == (2, 1)
    np.testing.assert_array_equal(actual_y.data.view(np.uint32), expected_y.view(np.uint32))
    np.testing.assert_array_equal(actual_mean.data.view(np.uint32), expected_mean.view(np.uint32))
    np.testing.assert_array_equal(actual_inv.data.view(np.uint32), expected_inv.view(np.uint32))


def test_layer_norm_preserves_empty_optional_output_slot_in_operator_and_graph():
    x = _tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    scale = _tensor(np.ones(3, dtype=np.float32))
    bias = _tensor(np.zeros(3, dtype=np.float32))
    op = LayerNormalization(["x", "scale", "bias"], ["y", "", "inv"], stash_type=1, dtype="float32")
    direct = op.forward(x, scale, bias)["tensor"]
    inferred = op.forward_(Tensor_(1, 3, dtype="float32"), Tensor_(3, dtype="float32"), Tensor_(3, dtype="float32"))["tensor"]
    assert len(direct) == len(inferred) == 3
    assert direct[1] is inferred[1] is None
    assert direct[2].size == inferred[2].size == (1, 1)
    graph = Graph([op], ["x", "scale", "bias"], ["y", "inv"])
    graph_y, graph_inv = graph.forward(x, scale, bias)
    shape_y, shape_inv = graph.forward_(
        Tensor_(1, 3, dtype="float32"), Tensor_(3, dtype="float32"), Tensor_(3, dtype="float32")
    )
    assert graph_y.size == shape_y.size == (1, 3)
    assert graph_inv.size == shape_inv.size == (1, 1)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_layer_norm_low_precision_python_fallback_materializes_input_dtype(dtype):
    values = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    if dtype == "float16":
        stored = values.astype(np.float16)
        scale = np.array([1.25, 0.75, -0.5], dtype=np.float16)
        bias = np.array([0.125, -0.25, 0.5], dtype=np.float16)
    else:
        def bf16(v):
            bits = np.asarray(v, dtype=np.float32).view(np.uint32)
            bits = bits + np.uint32(0x7fff) + ((bits >> 16) & 1)
            return (bits >> 16).astype(np.uint16)
        stored, scale, bias = bf16(values), bf16([1.25, 0.75, -0.5]), bf16([0.125, -0.25, 0.5])
    op = LayerNormalization(["x", "scale", "bias"], ["y"], stash_type=1, dtype=dtype)
    op.lib = None
    actual = op.forward(_tensor(stored, dtype), _tensor(scale, dtype), _tensor(bias, dtype))["tensor"]
    assert actual.dtype == dtype
    assert actual.data.dtype == stored.dtype


def test_layer_norm_rejects_non_schema_stash_type():
    with pytest.raises(ValueError, match="stash_type"):
        LayerNormalization(["x"], ["y"], stash_type=11, dtype="float32")


def test_layer_norm_direct_empty_outputs_keeps_single_result_compatibility():
    x = _tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    scale = _tensor(np.ones(3, dtype=np.float32))
    bias = _tensor(np.zeros(3, dtype=np.float32))
    op = LayerNormalization(["x", "scale", "bias"], [], stash_type=1, dtype="float32")
    op.lib = None
    actual = op.forward(x, scale, bias)["tensor"]
    inferred = op.forward_(Tensor_(1, 3, dtype="float32"), Tensor_(3, dtype="float32"), Tensor_(3, dtype="float32"))["tensor"]
    assert isinstance(actual, Tensor)
    assert isinstance(inferred, Tensor_)
    assert actual.size == inferred.size == (1, 3)


@pytest.mark.parametrize(
    "x",
    [
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.array([[1.0e8, 1.0e8, 1.0e8 + 8.0]], dtype=np.float32),
    ],
    ids=["ordinary", "large_offset"],
)
def test_cuda_layer_norm_float32_single_output_matches_reference_and_independent_formula(monkeypatch, x):
    scale = np.ones(x.shape[-1], dtype=np.float32)
    bias = np.zeros(x.shape[-1], dtype=np.float32)
    actual = _run_layer_norm_cuda(monkeypatch, x, scale, bias, emit_stats=0, input_proto=TensorProto.FLOAT).astype(np.float32)
    model = _layer_norm_model(x.shape)
    reference = ReferenceEvaluator(model).run(None, {"x": x, "scale": scale, "bias": bias})[0]
    independent = _float32_layer_norm(x, 1e-5)[0]
    np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
    np.testing.assert_array_equal(actual.view(np.uint32), independent.view(np.uint32))


def test_cuda_layer_norm_float32_stats_match_reference_and_independent_formula(monkeypatch):
    x = np.array([[1.0, 2.0, 3.0], [1.0e8, 1.0e8, 1.0e8 + 8.0]], dtype=np.float32)
    scale = np.ones(x.shape[-1], dtype=np.float32)
    bias = np.zeros(x.shape[-1], dtype=np.float32)
    actual = _run_layer_norm_cuda(monkeypatch, x, scale, bias, emit_stats=1, input_proto=TensorProto.FLOAT)
    model = _layer_norm_model(x.shape, ("y", "mean", "inv"))
    reference_y, reference_mean, reference_inv = ReferenceEvaluator(model).run(
        None, {"x": x, "scale": scale, "bias": bias}
    )
    independent_y, independent_mean, independent_inv = _float32_layer_norm(x, 1e-5)
    comparisons = (
        (actual.output, reference_y, independent_y),
        (actual.sidecars["tmp_layer_norm_mean.bin"], reference_mean, independent_mean),
        (actual.sidecars["tmp_layer_norm_inv_std.bin"], reference_inv, independent_inv),
    )
    for cuda_value, reference_value, independent_value in comparisons:
        cuda_float32 = cuda_value.astype(np.float32)
        np.testing.assert_array_equal(cuda_float32.view(np.uint32), reference_value.view(np.uint32))
        np.testing.assert_array_equal(cuda_float32.view(np.uint32), independent_value.view(np.uint32))


@pytest.mark.parametrize(
    ("np_dtype", "input_proto"),
    [(np.float16, TensorProto.FLOAT16), (np.float64, TensorProto.DOUBLE)],
    ids=["float16", "float64"],
)
def test_cuda_layer_norm_stage_two_uses_input_dtype_with_nontrivial_affine(monkeypatch, np_dtype, input_proto):
    x = np.array([[1.0, 2.25, 4.5], [-3.0, 0.75, 8.0]], dtype=np_dtype)
    scale = np.array([1.25, -0.75, 0.375], dtype=np_dtype)
    bias = np.array([0.125, -0.5, 1.75], dtype=np_dtype)
    actual = _run_layer_norm_cuda(monkeypatch, x, scale, bias, emit_stats=1, input_proto=input_proto)
    model = _layer_norm_model(x.shape, ("y", "mean", "inv"), input_proto=input_proto)
    reference_y, reference_mean, reference_inv = ReferenceEvaluator(model).run(
        None, {"x": x, "scale": scale, "bias": bias}
    )
    independent_y, independent_mean, independent_inv = _layer_norm_two_stage_formula(x, scale, bias)

    assert reference_y.dtype == np.dtype(np_dtype)
    cuda_y = actual.output.astype(np_dtype)
    cuda_mean = actual.sidecars["tmp_layer_norm_mean.bin"].astype(np.float32)
    cuda_inv = actual.sidecars["tmp_layer_norm_inv_std.bin"].astype(np.float32)
    assert independent_mean.dtype == independent_inv.dtype == np.dtype(np.float32)
    assert cuda_mean.dtype == cuda_inv.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(cuda_y, independent_y)
    np.testing.assert_array_equal(cuda_mean.view(np.uint32), independent_mean.view(np.uint32))
    np.testing.assert_array_equal(cuda_inv.view(np.uint32), independent_inv.view(np.uint32))
    # ONNX 1.21 ReferenceEvaluator vectorizes stage one and exposes aux outputs in T rather than U.
    reference_rtol = 1e-3 if np_dtype == np.float16 else 2e-7
    np.testing.assert_allclose(cuda_y, reference_y, rtol=reference_rtol, atol=reference_rtol)
    np.testing.assert_allclose(cuda_mean, reference_mean.astype(np.float32), rtol=reference_rtol, atol=reference_rtol)
    np.testing.assert_allclose(cuda_inv, reference_inv.astype(np.float32), rtol=reference_rtol, atol=reference_rtol)
