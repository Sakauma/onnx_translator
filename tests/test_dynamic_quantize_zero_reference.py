# /**
#   ******************************************************************************
#   * @file        test_dynamic_quantize_zero_reference.py
#   * @author      Egor Izmaylov
#   * @brief       固化 DynamicQuantizeLinear 全零输入与 ONNX 1.21 reference 的兼容行为。
#   * @details     2026.09.11  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from onnx.reference import ReferenceEvaluator

from operator_test_context import *  # noqa: F401,F403
from nn.Operators import DynamicQuantizeLinear
from tools.numerical.cuda import run_cuda_ground_truth


def _reference(x):
    graph = helper.make_graph(
        [helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "scale", "zero_point"])],
        "dynamic_quantize_zero_reference",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [
            helper.make_tensor_value_info("y", TensorProto.UINT8, list(x.shape)),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("zero_point", TensorProto.UINT8, []),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return ReferenceEvaluator(model).run(None, {"x": x})


def _require_c_backend():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")


def _run_c(x):
    tensor = Tensor(*x.shape, dtype="float32", data=x)
    return DynamicQuantizeLinear(["x"], ["y", "scale", "zero_point"]).forward(tensor)["tensor"]


def _run_cuda(x):
    packed = run_cuda_ground_truth(
        "dynamic_quantize_linear",
        [x],
        output_dtype=np.float32,
        target_shape=(x.size + 2,),
    )
    return (
        packed[: x.size].reshape(x.shape).astype(np.uint8),
        np.asarray(packed[x.size], dtype=np.float32),
        np.asarray(packed[x.size + 1], dtype=np.uint8),
    )


def _assert_outputs(actual, expected):
    for actual_value, expected_value in zip(actual, expected):
        actual_array = actual_value.data if isinstance(actual_value, Tensor) else np.asarray(actual_value)
        expected_array = np.asarray(expected_value)
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        np.testing.assert_array_equal(actual_array, expected_array)


ZERO_INPUTS = [
    pytest.param(np.array([0.0], dtype=np.float32), id="scalar-sized-positive-zero"),
    pytest.param(np.array([0.0, -0.0], dtype=np.float32), id="signed-zero-vector"),
    pytest.param(np.array([[0.0, -0.0], [-0.0, 0.0]], dtype=np.float32), id="signed-zero-matrix"),
]


@pytest.mark.parametrize("x", ZERO_INPUTS)
def test_c_backend_all_zero_matches_onnx_121_reference_bit_exact(x, monkeypatch):
    _require_c_backend()
    assert onnx.__version__.startswith("1.21.")
    expected = _reference(x)
    expected_scale_bits = np.asarray(expected[1], dtype=np.float32).view(np.uint32).item()
    assert expected_scale_bits == (np.float32(1.0) / np.float32(255.0)).view(np.uint32).item()

    op = DynamicQuantizeLinear(["x"], ["y", "scale", "zero_point"])
    c_forward = op.lib.dynamic_quantize_linear_forward
    calls = []

    def counted_c_forward(*args):
        calls.append(True)
        return c_forward(*args)

    monkeypatch.setattr(op.lib, "dynamic_quantize_linear_forward", counted_c_forward)
    actual = op.forward(Tensor(*x.shape, dtype="float32", data=x))["tensor"]
    assert calls == [True]
    _assert_outputs(actual, expected)
    assert actual[1].data.view(np.uint32).item() == expected_scale_bits


@pytest.mark.parametrize("x", ZERO_INPUTS)
def test_cuda_verifier_all_zero_matches_onnx_121_reference_bit_exact(x):
    assert onnx.__version__.startswith("1.21.")
    expected = _reference(x)
    actual = _run_cuda(x)
    _assert_outputs(actual, expected)
    assert actual[1].view(np.uint32).item() == np.asarray(expected[1]).view(np.uint32).item()


@pytest.mark.parametrize(
    "x",
    [
        pytest.param(np.full((3,), 2.0, dtype=np.float32), id="positive-nonzero-constant"),
        pytest.param(np.full((2, 2), -2.0, dtype=np.float32), id="negative-nonzero-constant"),
        pytest.param(np.array([-3.25, -0.5, 0.0, 1.75, 8.0], dtype=np.float32), id="nondegenerate"),
        pytest.param(
            np.array([-6.7799618e17, 4.5765801e17, 0.0, 7.7123871e19], dtype=np.float32),
            id="existing-float32-precision-fixture",
        ),
    ],
)
def test_nonzero_paths_still_match_reference_in_c_and_cuda(x):
    _require_c_backend()
    expected = _reference(x)
    _assert_outputs(_run_c(x), expected)
    _assert_outputs(_run_cuda(x), expected)

def test_nonzero_range_scale_underflow_retains_existing_fallback_in_c_and_cuda():
    _require_c_backend()
    smallest_positive = np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)
    x = np.array([0.0, smallest_positive], dtype=np.float32)
    expected = (
        np.zeros(x.shape, dtype=np.uint8),
        np.array(1.0, dtype=np.float32),
        np.array(0, dtype=np.uint8),
    )

    _assert_outputs(_run_c(x), expected)
    _assert_outputs(_run_cuda(x), expected)
