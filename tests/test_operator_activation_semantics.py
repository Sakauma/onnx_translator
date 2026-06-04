# /**
#   ******************************************************************************
#   * @file        test_operator_activation_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用 ONNX reference 验证激活和一元数学算子的混合精度语义。
#   * @details     2026.06.04  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import math

from onnx.reference import ReferenceEvaluator

from operator_test_context import *  # noqa: F401,F403
from nn.ONNXImport import ONNXImport
from nn.Operators import (
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atanh,
    Celu,
    Cosh,
    Elu,
    Erf,
    HardSigmoid,
    HardSwish,
    LeakyRelu,
    Round,
    Selu,
    Shrink,
    Sinh,
    Softplus,
    Softsign,
    Swish,
    ThresholdedRelu,
)


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


# 将 bfloat16 的 uint16 位模式解码成 float32，用于按数值容差比较输出。
def _bf16_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


# 调用 ONNX reference evaluator，获得指定 op 在对应 dtype 下的官方参考输出。
def _onnx_reference(op_name, values, proto_dtype, attrs, opset=17):
    graph = helper.make_graph(
        [helper.make_node(op_name, ["x"], ["y"], **attrs)],
        f"{op_name}_reference",
        [helper.make_tensor_value_info("x", proto_dtype, list(values.shape))],
        [helper.make_tensor_value_info("y", proto_dtype, list(values.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return ReferenceEvaluator(model).run(None, {"x": values})[0]


REFERENCE_CASES = [
    (Elu, "Elu", {"alpha": 0.7}, np.array([-2.0, -0.5, 0.0, 1.5], dtype=np.float64), True),
    (Selu, "Selu", {"alpha": 1.2, "gamma": 0.8}, np.array([-2.0, -0.5, 0.0, 1.5], dtype=np.float64), True),
    (Celu, "Celu", {"alpha": 0.7}, np.array([-2.0, -0.5, 0.0, 1.5], dtype=np.float64), False),
    (LeakyRelu, "LeakyRelu", {"alpha": 0.25}, np.array([-2.0, -0.5, 0.0, 1.5], dtype=np.float64), True),
    (HardSigmoid, "HardSigmoid", {"alpha": 0.3, "beta": 0.4}, np.array([-2.0, -0.5, 0.0, 1.5, 4.0], dtype=np.float64), True),
    (HardSwish, "HardSwish", {}, np.array([-4.0, -2.0, 0.0, 2.0, 4.0], dtype=np.float64), True),
    (ThresholdedRelu, "ThresholdedRelu", {"alpha": 0.3}, np.array([-1.0, 0.2, 0.3, 0.4], dtype=np.float64), True),
    (Softplus, "Softplus", {}, np.array([-4.0, -1.0, 0.0, 2.0], dtype=np.float64), True),
    (Softsign, "Softsign", {}, np.array([-4.0, -1.0, 0.0, 2.0], dtype=np.float64), True),
    (Shrink, "Shrink", {"bias": 0.2, "lambd": 0.5}, np.array([-1.0, -0.4, 0.0, 0.6, 2.0], dtype=np.float64), True),
    (Round, "Round", {}, np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float64), True),
    (Erf, "Erf", {}, np.array([-2.0, -0.5, 0.0, 1.5], dtype=np.float64), True),
    (Acos, "Acos", {}, np.array([-0.9, -0.2, 0.3, 0.8], dtype=np.float64), True),
    (Asin, "Asin", {}, np.array([-0.9, -0.2, 0.3, 0.8], dtype=np.float64), True),
    (Cosh, "Cosh", {}, np.array([-2.0, -0.5, 0.0, 1.5], dtype=np.float64), True),
    (Sinh, "Sinh", {}, np.array([-2.0, -0.5, 0.0, 1.5], dtype=np.float64), True),
    (Asinh, "Asinh", {}, np.array([-2.0, -0.5, 0.0, 1.5], dtype=np.float64), True),
    (Acosh, "Acosh", {}, np.array([1.0, 1.2, 2.0, 4.0], dtype=np.float64), True),
    (Atanh, "Atanh", {}, np.array([-0.8, -0.2, 0.3, 0.8], dtype=np.float64), True),
]


# 验证激活和一元数学算子在 float64 下与 ONNX reference 保持一致。
@pytest.mark.parametrize("op_cls,op_name,attrs,values,_supports_float16", REFERENCE_CASES)
def test_c_backend_unary_ops_float64_match_onnx_reference(op_cls, op_name, attrs, values, _supports_float16):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    inputs = values.astype(np.float64)
    expected = _onnx_reference(op_name, inputs, TensorProto.DOUBLE, attrs)
    actual = op_cls(["x"], ["y"], dtype="float64", **attrs).forward(
        Tensor(*inputs.shape, dtype="float64", data=inputs)
    )["tensor"]
    np.testing.assert_allclose(actual.data, expected, rtol=1e-7, atol=1e-7)


# 验证官方支持 float16 的激活和一元数学算子与 ONNX reference 保持一致。
@pytest.mark.parametrize("op_cls,op_name,attrs,values,supports_float16", REFERENCE_CASES)
def test_c_backend_unary_ops_float16_match_onnx_reference(op_cls, op_name, attrs, values, supports_float16):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")
    if not supports_float16:
        pytest.skip(f"{op_name} does not support float16 in ONNX17")

    inputs = values.astype(np.float16)
    expected = _onnx_reference(op_name, inputs, TensorProto.FLOAT16, attrs)
    actual = op_cls(["x"], ["y"], dtype="float16", **attrs).forward(
        Tensor(*inputs.shape, dtype="float16", data=inputs)
    )["tensor"]
    np.testing.assert_allclose(actual.data, expected, rtol=2e-3, atol=2e-3)


# 验证官方支持 bfloat16 的 LeakyRelu/Erf 正确按位读取输入并写回 bfloat16 输出。
def test_c_backend_bfloat16_unary_ops_decode_and_write_bit_storage():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    values = np.array([-2.0, -0.5, 0.0, 1.5], dtype=np.float32)
    bf16_input = Tensor(*values.shape, dtype="bfloat16", data=_bf16_bits(values))
    leaky = LeakyRelu(["x"], ["y"], alpha=0.25, dtype="bfloat16").forward(bf16_input)["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(leaky.data), np.where(values >= 0.0, values, values * 0.25), rtol=1e-2, atol=1e-2)

    erf_expected = np.vectorize(math.erf, otypes=[np.float32])(values).astype(np.float32)
    erf = Erf(["x"], ["y"], dtype="bfloat16").forward(bf16_input)["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(erf.data), erf_expected, rtol=1e-2, atol=1e-2)


# 验证 Swish 的 opset 24 官方公式、alpha 属性和混合精度写回。
def test_c_backend_swish_opset24_alpha_and_bfloat16_match_reference(tmp_path):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    values = np.array([-4.0, -1.0, 0.0, 1.5, 4.0], dtype=np.float32)
    expected_default = _onnx_reference("Swish", values, TensorProto.FLOAT, {}, opset=24)
    actual_default = Swish(["x"], ["y"], dtype="float32").forward(Tensor(*values.shape, dtype="float32", data=values))["tensor"]
    np.testing.assert_allclose(actual_default.data, expected_default, rtol=1e-6, atol=1e-6)

    alpha = 1.5
    expected_alpha = _onnx_reference("Swish", values, TensorProto.FLOAT, {"alpha": alpha}, opset=24)
    actual_alpha = Swish(["x"], ["y"], alpha=alpha, dtype="float32").forward(
        Tensor(*values.shape, dtype="float32", data=values)
    )["tensor"]
    np.testing.assert_allclose(actual_alpha.data, expected_alpha, rtol=1e-6, atol=1e-6)

    bf16_input = Tensor(*values.shape, dtype="bfloat16", data=_bf16_bits(values))
    bf16_actual = Swish(["x"], ["y"], alpha=alpha, dtype="bfloat16").forward(bf16_input)["tensor"]
    bf16_expected = values * (1.0 / (1.0 + np.exp(-alpha * values)))
    np.testing.assert_allclose(_bf16_to_float32(bf16_actual.data), bf16_expected, rtol=1e-2, atol=1e-2)

    graph = helper.make_graph(
        [helper.make_node("Swish", ["x"], ["y"], alpha=alpha)],
        "swish_import",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(values.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(values.shape))],
    )
    model_path = tmp_path / "swish.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)]), model_path)
    imported = [op for op in ONNXImport(str(model_path), strict=True) if isinstance(op, Swish)]
    assert len(imported) == 1
    assert imported[0].alpha == pytest.approx(alpha)
