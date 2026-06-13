# /**
#   ******************************************************************************
#   * @file        test_operator_normalization_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用 ONNX reference 和独立公式验证归一化算子的混合精度语义。
#   * @details     2026.06.04  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from onnx.reference import ReferenceEvaluator

from conftest import _disable_c_backend
from operator_test_context import *  # noqa: F401,F403
from nn.Operators import BatchNormalization, InstanceNormalization, LayerNormalization, LpNormalization, LRN, MeanVarianceNormalization, RMSNormalization


# 将 float32 数值转换为 bfloat16 的 uint16 位模式，匹配 Tensor 内部存储。
def _bf16_bits(values):
    data = np.asarray(values, dtype=np.float32)
    bits = data.view(np.uint32)
    lsb = (bits >> 16) & 1
    guard = (bits >> 15) & 1
    sticky = (bits & 0x7FFF) != 0
    rounded = bits + ((guard & (sticky | lsb)).astype(np.uint32) << 16)
    rounded = np.where(np.isnan(data), bits, rounded)
    return (rounded >> 16).astype(np.uint16)


# 将 bfloat16 的 uint16 位模式解码成 float32，便于按数值容差比较输出。
def _bf16_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


# 构造 Tensor，避免每个断言重复 dtype、shape 和 data 样板。
def _tensor(data, dtype):
    return Tensor(*data.shape, dtype=dtype, data=data)


# 调用 ONNX reference evaluator，获得指定归一化 op 的官方参考输出。
def _onnx_reference(op_name, inputs, protos, attrs, output_shapes, opset=17, output_protos=None):
    names = [f"i{i}" for i in range(len(inputs))]
    outputs = [f"o{i}" for i in range(len(output_shapes))]
    output_protos = output_protos or [protos[0]] * len(output_shapes)
    graph = helper.make_graph(
        [helper.make_node(op_name, names, outputs, **attrs)],
        f"{op_name}_reference",
        [helper.make_tensor_value_info(name, proto, list(value.shape)) for name, proto, value in zip(names, protos, inputs)],
        [helper.make_tensor_value_info(name, proto, list(shape)) for name, proto, shape in zip(outputs, output_protos, output_shapes)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return ReferenceEvaluator(model).run(None, dict(zip(names, inputs)))


# 按 ONNX BatchNormalization 推理公式独立计算输出。
def _batch_norm_formula(x, scale, bias, mean, var, epsilon):
    shape = (-1,) + (1,) * (x.ndim - 2)
    return scale.reshape(shape) * (x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + epsilon) + bias.reshape(shape)


# 按 ONNX BatchNormalization 训练公式独立计算 Y、running_mean 和 running_var。
def _batch_norm_training_formula(x, scale, bias, mean, var, epsilon, momentum):
    axes = tuple(axis for axis in range(x.ndim) if axis != 1)
    saved_mean = np.mean(x, axis=axes)
    saved_var = np.var(x, axis=axes)
    y = _batch_norm_formula(x, scale, bias, saved_mean, saved_var, epsilon)
    running_mean = mean * momentum + saved_mean * (1.0 - momentum)
    running_var = var * momentum + saved_var * (1.0 - momentum)
    return y, running_mean, running_var


# 按 ONNX MeanVarianceNormalization 公式独立计算输出。
def _mvn_formula(x, axes):
    mean = np.mean(x, axis=tuple(axes), keepdims=True)
    variance = np.mean((x - mean) ** 2, axis=tuple(axes), keepdims=True)
    return (x - mean) / np.sqrt(variance)


# 按 ONNX RMSNormalization 公式独立计算输出。
def _rms_norm_formula(x, scale, axis, epsilon):
    axes = tuple(range(axis if axis >= 0 else axis + x.ndim, x.ndim))
    mean_square = np.mean(x * x, axis=axes, keepdims=True)
    return x / np.sqrt(mean_square + epsilon) * scale


# 按 ONNX LRN schema 公式独立计算输出，避免依赖本地 reference 中的通道循环缺陷。
def _lrn_formula(x, size, alpha, beta, bias):
    data = np.asarray(x, dtype=np.float64)
    out = np.empty_like(data, dtype=np.float64)
    channels = data.shape[1]
    lower = (size - 1) // 2
    upper = size - 1 - lower
    for c in range(channels):
        begin = max(0, c - lower)
        end = min(channels, c + upper + 1)
        square_sum = np.sum(data[:, begin:end, ...] ** 2, axis=1)
        out[:, c, ...] = data[:, c, ...] / np.power(bias + alpha / size * square_sum, beta)
    return out


# 验证 BatchNormalization 推理路径在 float16/float64 下与 ONNX reference 一致。
@pytest.mark.parametrize("dtype,proto,rtol,atol", [("float64", TensorProto.DOUBLE, 1e-10, 1e-10), ("float16", TensorProto.FLOAT16, 1e-2, 1e-2)])
def test_c_backend_batch_normalization_inference_matches_onnx_reference(dtype, proto, rtol, atol):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    np_dtype = np.float16 if dtype == "float16" else np.float64
    x = ((np.arange(2 * 3 * 2 * 2, dtype=np.float64).reshape(2, 3, 2, 2) / 10.0) - 1.0).astype(np_dtype)
    scale = np.array([1.0, 0.5, 1.5], dtype=np_dtype)
    bias = np.array([0.1, -0.2, 0.3], dtype=np_dtype)
    mean = np.array([0.0, 0.2, -0.1], dtype=np_dtype)
    var = np.array([1.0, 0.5, 2.0], dtype=np_dtype)
    expected = _onnx_reference(
        "BatchNormalization",
        [x, scale, bias, mean, var],
        [proto] * 5,
        {"epsilon": 1e-4, "training_mode": 0},
        [x.shape],
    )[0]
    actual = BatchNormalization(["x", "scale", "b", "mean", "var"], ["y"], epsilon=1e-4, dtype=dtype).forward(
        _tensor(x, dtype), _tensor(scale, dtype), _tensor(bias, dtype), _tensor(mean, dtype), _tensor(var, dtype)
    )["tensor"]
    np.testing.assert_allclose(actual.data, expected, rtol=rtol, atol=atol)


# 验证 BatchNormalization 训练模式三输出由 C 后端承载，并覆盖 float32/float16/bfloat16 写回。
@pytest.mark.parametrize("dtype,rtol,atol", [("float32", 1e-6, 1e-6), ("float16", 2e-2, 2e-2), ("bfloat16", 2e-2, 2e-2)])
def test_c_backend_batch_normalization_training_mode_outputs(dtype, rtol, atol):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x_f32 = np.linspace(-1.2, 1.3, 2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2)
    scale_f32 = np.array([0.75, 1.125, 1.5], dtype=np.float32)
    bias_f32 = np.array([-0.2, 0.05, 0.3], dtype=np.float32)
    mean_f32 = np.array([-0.1, 0.05, 0.2], dtype=np.float32)
    var_f32 = np.array([0.5, 1.25, 2.0], dtype=np.float32)

    if dtype == "bfloat16":
        x_data = _bf16_bits(x_f32)
        scale_data = _bf16_bits(scale_f32)
        bias_data = _bf16_bits(bias_f32)
        mean_data = _bf16_bits(mean_f32)
        var_data = _bf16_bits(var_f32)
        x_ref = _bf16_to_float32(x_data)
        scale_ref = _bf16_to_float32(scale_data)
        bias_ref = _bf16_to_float32(bias_data)
        mean_ref = _bf16_to_float32(mean_data)
        var_ref = _bf16_to_float32(var_data)
    else:
        np_dtype = np.float16 if dtype == "float16" else np.float32
        x_data = x_f32.astype(np_dtype)
        scale_data = scale_f32.astype(np_dtype)
        bias_data = bias_f32.astype(np_dtype)
        mean_data = mean_f32.astype(np_dtype)
        var_data = var_f32.astype(np_dtype)
        x_ref = x_data.astype(np.float32)
        scale_ref = scale_data.astype(np.float32)
        bias_ref = bias_data.astype(np.float32)
        mean_ref = mean_data.astype(np.float32)
        var_ref = var_data.astype(np.float32)

    op = BatchNormalization(
        ["x", "scale", "b", "mean", "var"],
        ["y", "running_mean", "running_var"],
        epsilon=1e-4,
        momentum=0.75,
        training_mode=1,
        dtype=dtype,
    )
    assert getattr(op, "_has_batch_norm_training_c_backend", False)

    actual_y, actual_mean, actual_var = op.forward(
        _tensor(x_data, dtype),
        _tensor(scale_data, dtype),
        _tensor(bias_data, dtype),
        _tensor(mean_data, dtype),
        _tensor(var_data, dtype),
    )["tensor"]
    expected_y, expected_mean, expected_var = _batch_norm_training_formula(
        x_ref, scale_ref, bias_ref, mean_ref, var_ref, 1e-4, 0.75
    )

    if dtype == "bfloat16":
        actual_y_data = _bf16_to_float32(actual_y.data)
        actual_mean_data = _bf16_to_float32(actual_mean.data)
        actual_var_data = _bf16_to_float32(actual_var.data)
    else:
        actual_y_data = actual_y.data.astype(np.float32)
        actual_mean_data = actual_mean.data.astype(np.float32)
        actual_var_data = actual_var.data.astype(np.float32)

    np.testing.assert_allclose(actual_y_data, expected_y, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual_mean_data, expected_mean, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual_var_data, expected_var, rtol=rtol, atol=atol)


# 验证 InstanceNormalization、LayerNormalization 和 LpNormalization 的低精度 reference 对齐。
def test_c_backend_normalization_ops_match_onnx_reference_mixed_precision():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = ((np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2) / 10.0) - 1.0).astype(np.float16)
    scale = np.array([1.0, 0.5, 1.5], dtype=np.float16)
    bias = np.array([0.1, -0.2, 0.3], dtype=np.float16)
    expected_instance = _onnx_reference(
        "InstanceNormalization", [x, scale, bias], [TensorProto.FLOAT16] * 3, {"epsilon": 1e-4}, [x.shape]
    )[0]
    actual_instance = InstanceNormalization(["x", "scale", "b"], ["y"], epsilon=1e-4, dtype="float16").forward(
        _tensor(x, "float16"), _tensor(scale, "float16"), _tensor(bias, "float16")
    )["tensor"]
    np.testing.assert_allclose(actual_instance.data, expected_instance, rtol=1e-2, atol=1e-2)

    lp_expected = _onnx_reference(
        "LpNormalization", [x], [TensorProto.FLOAT16], {"axis": 1, "p": 2}, [x.shape]
    )[0]
    lp_actual = LpNormalization(["x"], ["y"], axis=1, p=2, dtype="float16").forward(_tensor(x, "float16"))["tensor"]
    np.testing.assert_allclose(lp_actual.data, lp_expected, rtol=2e-3, atol=2e-3)

    layer_x = ((np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4) / 7.0) - 1.0).astype(np.float64)
    layer_scale = np.array([1.0, 0.5, 1.5, -0.5], dtype=np.float64)
    layer_bias = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float64)
    layer_expected = _onnx_reference(
        "LayerNormalization",
        [layer_x, layer_scale, layer_bias],
        [TensorProto.DOUBLE] * 3,
        {"axis": -1, "epsilon": 1e-4, "stash_type": 1},
        [layer_x.shape],
    )[0]
    layer_actual = LayerNormalization(["x", "scale", "b"], ["y"], axis=-1, epsilon=1e-4, stash_type=1, dtype="float64").forward(
        _tensor(layer_x, "float64"), _tensor(layer_scale, "float64"), _tensor(layer_bias, "float64")
    )["tensor"]
    np.testing.assert_allclose(layer_actual.data, layer_expected, rtol=1e-10, atol=1e-10)


# 验证 LRN 和 MVN 按 ONNX schema 公式计算，覆盖 reference evaluator 当前不可靠的路径。
def test_c_backend_lrn_and_mvn_match_independent_onnx_formulas():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = ((np.arange(2 * 3 * 2 * 2, dtype=np.float64).reshape(2, 3, 2, 2) / 10.0) - 1.0).astype(np.float64)
    lrn_actual = LRN(["x"], ["y"], size=3, alpha=0.3, beta=0.5, bias=1.0, dtype="float64").forward(_tensor(x, "float64"))["tensor"]
    np.testing.assert_allclose(lrn_actual.data, _lrn_formula(x, 3, 0.3, 0.5, 1.0), rtol=1e-8, atol=1e-8)

    x16 = x.astype(np.float16)
    mvn_expected = _mvn_formula(x16.astype(np.float64), axes=[0, 2, 3]).astype(np.float16)
    mvn_actual = MeanVarianceNormalization(["x"], ["y"], axes=[0, 2, 3], dtype="float16").forward(_tensor(x16, "float16"))["tensor"]
    np.testing.assert_allclose(mvn_actual.data, mvn_expected, rtol=1e-2, atol=1e-2)


# 验证 RMSNormalization 的官方 axis、epsilon、stash_type 和 scale 广播语义。
@pytest.mark.parametrize(
    "axis,scale_shape",
    [(-1, (4,)), (1, (3, 4))],
)
def test_c_backend_rms_normalization_matches_onnx_reference(axis, scale_shape):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = (np.linspace(-1.5, 1.5, 24, dtype=np.float32).reshape(2, 3, 4))
    scale = np.linspace(0.5, 1.5, int(np.prod(scale_shape)), dtype=np.float32).reshape(scale_shape)
    expected = _onnx_reference(
        "RMSNormalization",
        [x, scale],
        [TensorProto.FLOAT, TensorProto.FLOAT],
        {"axis": axis, "epsilon": 1e-4, "stash_type": 1},
        [x.shape],
        opset=23,
    )[0]
    actual = RMSNormalization(["x", "scale"], ["y"], axis=axis, epsilon=1e-4, stash_type=1, dtype="float32").forward(
        _tensor(x, "float32"),
        _tensor(scale, "float32"),
    )["tensor"]
    np.testing.assert_allclose(actual.data, expected, rtol=1e-6, atol=1e-6)


# 验证 RMSNormalization 导入时保留 axis、epsilon 和 stash_type 属性。
def test_onnx_import_rms_normalization_preserves_attributes(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("RMSNormalization", ["x", "scale"], ["y"], axis=1, epsilon=1e-4, stash_type=1)],
        "rms_norm_import",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [3, 4]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])],
    )
    model_path = tmp_path / "rms_norm.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)]), model_path)

    imported = [op for op in ONNXImport(str(model_path), strict=True) if isinstance(op, RMSNormalization)]
    assert len(imported) == 1
    assert imported[0].axis == 1
    assert imported[0].epsilon == pytest.approx(1e-4)
    assert imported[0].stash_type == 1
    assert imported[0].version == "23"


# 验证官方支持 bfloat16 的归一化算子正确读写低精度位模式。
def test_c_backend_bfloat16_normalization_ops_decode_and_write_bit_storage():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x_values = ((np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2) / 10.0) - 1.0).astype(np.float32)
    scale = np.array([1.0, 0.5, 1.5], dtype=np.float32)
    bias = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    mean = np.array([0.0, 0.2, -0.1], dtype=np.float32)
    var = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    bf16_x = _tensor(_bf16_bits(x_values), "bfloat16")
    bf16_scale = _tensor(_bf16_bits(scale), "bfloat16")
    bf16_bias = _tensor(_bf16_bits(bias), "bfloat16")
    bf16_mean = _tensor(_bf16_bits(mean), "bfloat16")
    bf16_var = _tensor(_bf16_bits(var), "bfloat16")

    batch = BatchNormalization(["x", "scale", "b", "mean", "var"], ["y"], epsilon=1e-4, dtype="bfloat16").forward(
        bf16_x, bf16_scale, bf16_bias, bf16_mean, bf16_var
    )["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(batch.data), _batch_norm_formula(x_values, scale, bias, mean, var, 1e-4), rtol=2e-2, atol=2e-2)

    mvn = MeanVarianceNormalization(["x"], ["y"], axes=[0, 2, 3], dtype="bfloat16").forward(bf16_x)["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(mvn.data), _mvn_formula(x_values, [0, 2, 3]), rtol=2e-2, atol=2e-2)

    lrn = LRN(["x"], ["y"], size=3, alpha=0.3, beta=0.5, bias=1.0, dtype="bfloat16").forward(bf16_x)["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(lrn.data), _lrn_formula(x_values, 3, 0.3, 0.5, 1.0), rtol=2e-2, atol=2e-2)

    rms_scale_values = np.linspace(0.5, 1.5, 2, dtype=np.float32)
    rms = RMSNormalization(["x", "scale"], ["y"], axis=-1, epsilon=1e-4, stash_type=1, dtype="bfloat16").forward(
        bf16_x,
        _tensor(_bf16_bits(rms_scale_values), "bfloat16"),
    )["tensor"]
    decoded_x = _bf16_to_float32(_bf16_bits(x_values))
    decoded_scale = _bf16_to_float32(_bf16_bits(rms_scale_values))
    np.testing.assert_allclose(
        _bf16_to_float32(rms.data),
        _rms_norm_formula(decoded_x, decoded_scale, axis=-1, epsilon=1e-4),
        rtol=2e-2,
        atol=2e-2,
    )


# 验证 Python fallback 归一化路径同样按 bfloat16 位模式解码输入并写回位模式。
def test_python_normalization_fallback_bfloat16_decodes_bit_storage(monkeypatch):
    _disable_c_backend(monkeypatch)

    x_values = ((np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2) / 10.0) - 1.0).astype(np.float32)
    scale = np.array([1.0, 0.5, 1.5], dtype=np.float32)
    bias = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    mean = np.array([0.0, 0.2, -0.1], dtype=np.float32)
    var = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    bf16_x = _tensor(_bf16_bits(x_values), "bfloat16")
    bf16_scale = _tensor(_bf16_bits(scale), "bfloat16")
    bf16_bias = _tensor(_bf16_bits(bias), "bfloat16")

    batch = BatchNormalization(["x", "scale", "b", "mean", "var"], ["y"], epsilon=1e-4, dtype="bfloat16").forward(
        bf16_x,
        bf16_scale,
        bf16_bias,
        _tensor(_bf16_bits(mean), "bfloat16"),
        _tensor(_bf16_bits(var), "bfloat16"),
    )["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(batch.data), _batch_norm_formula(x_values, scale, bias, mean, var, 1e-4), rtol=2e-2, atol=2e-2)

    mvn = MeanVarianceNormalization(["x"], ["y"], axes=[0, 2, 3], dtype="bfloat16").forward(bf16_x)["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(mvn.data), _mvn_formula(x_values, [0, 2, 3]), rtol=2e-2, atol=2e-2)

    lrn = LRN(["x"], ["y"], size=3, alpha=0.3, beta=0.5, bias=1.0, dtype="bfloat16").forward(bf16_x)["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(lrn.data), _lrn_formula(x_values, 3, 0.3, 0.5, 1.0), rtol=2e-2, atol=2e-2)

    lp = LpNormalization(["x"], ["y"], axis=1, p=2, dtype="bfloat16").forward(bf16_x)["tensor"]
    lp_norm = np.linalg.norm(x_values, ord=2, axis=1, keepdims=True)
    lp_expected = np.where(lp_norm == 0, 0, x_values / lp_norm)
    np.testing.assert_allclose(_bf16_to_float32(lp.data), lp_expected, rtol=2e-2, atol=2e-2)

    rms_scale_values = np.linspace(0.5, 1.5, 2, dtype=np.float32)
    rms = RMSNormalization(["x", "scale"], ["y"], axis=-1, epsilon=1e-4, stash_type=1, dtype="bfloat16").forward(
        bf16_x,
        _tensor(_bf16_bits(rms_scale_values), "bfloat16"),
    )["tensor"]
    decoded_x = _bf16_to_float32(_bf16_bits(x_values))
    decoded_scale = _bf16_to_float32(_bf16_bits(rms_scale_values))
    np.testing.assert_allclose(
        _bf16_to_float32(rms.data),
        _rms_norm_formula(decoded_x, decoded_scale, axis=-1, epsilon=1e-4),
        rtol=2e-2,
        atol=2e-2,
    )

    mean_out = Mean(["a", "b"], ["y"], dtype="bfloat16").forward(
        bf16_x,
        _tensor(_bf16_bits(x_values + 1.0), "bfloat16"),
    )["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(mean_out.data), x_values + 0.5, rtol=2e-2, atol=2e-2)


# 验证 LayerNormalization 的 aux-output fallback 不把 bfloat16 存储位模式当作整数幅值。
def test_python_layer_normalization_bfloat16_fallback_decodes_bit_storage(monkeypatch):
    _disable_c_backend(monkeypatch)

    x_values = ((np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) / 7.0) - 1.0).astype(np.float32)
    scale = np.array([1.0, 0.5, 1.5, -0.5], dtype=np.float32)
    bias = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
    actual_y, actual_mean, actual_inv_std = LayerNormalization(
        ["x", "scale", "bias"],
        ["y", "mean", "inv_std"],
        axis=-1,
        epsilon=1e-4,
        stash_type=1,
        dtype="bfloat16",
    ).forward(
        _tensor(_bf16_bits(x_values), "bfloat16"),
        _tensor(_bf16_bits(scale), "bfloat16"),
        _tensor(_bf16_bits(bias), "bfloat16"),
    )["tensor"]

    decoded_x = _bf16_to_float32(_bf16_bits(x_values))
    decoded_scale = _bf16_to_float32(_bf16_bits(scale))
    decoded_bias = _bf16_to_float32(_bf16_bits(bias))
    mean = np.mean(decoded_x, axis=-1, keepdims=True)
    inv_std = np.reciprocal(np.sqrt(np.mean((decoded_x - mean) ** 2, axis=-1, keepdims=True) + 1e-4))
    expected_y = (decoded_x - mean) * inv_std * decoded_scale.reshape((1, 1, 4)) + decoded_bias.reshape((1, 1, 4))
    np.testing.assert_allclose(_bf16_to_float32(actual_y.data), expected_y, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(actual_mean.data, mean.reshape(2, 3, 1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_inv_std.data, inv_std.reshape(2, 3, 1), rtol=1e-6, atol=1e-6)
