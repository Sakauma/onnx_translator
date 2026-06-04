# /**
#   ******************************************************************************
#   * @file        test_operator_elementwise_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用 ONNX reference 验证基础元素级算子的官方语义和混合精度路径。
#   * @details     2026.06.05  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from onnx.reference import ReferenceEvaluator

from operator_test_context import *  # noqa: F401,F403
from nn.Operators import (
    ABS,
    ADD,
    COS,
    DIV,
    EXP,
    LOG,
    MUL,
    RELU,
    SIGMOID,
    SQRT,
    SUB,
    TANH,
    And,
    Atan,
    Equal,
    Floor,
    Greater,
    GreaterOrEqual,
    IsNaN,
    Less,
    LessOrEqual,
    Max,
    Min,
    Neg,
    Not,
    Or,
    Pow,
    Sign,
    Sin,
    Softmax,
    Tan,
    Xor,
)


# 构造 Tensor，避免每个用例重复写 shape、dtype 和 data 样板。
def _tensor(data, dtype):
    data = np.asarray(data)
    return Tensor(*data.shape, dtype=dtype, data=data)


# 调用 ONNX reference evaluator，得到指定单节点模型的官方输出。
def _onnx_reference(op_name, inputs, input_protos, attrs, output_shapes, output_protos=None):
    output_protos = output_protos or [input_protos[0]] * len(output_shapes)
    input_names = [f"i{i}" for i in range(len(inputs))]
    output_names = [f"o{i}" for i in range(len(output_shapes))]
    graph = helper.make_graph(
        [helper.make_node(op_name, input_names, output_names, **attrs)],
        f"{op_name}_reference",
        [
            helper.make_tensor_value_info(name, proto, list(np.asarray(value).shape))
            for name, proto, value in zip(input_names, input_protos, inputs)
        ],
        [
            helper.make_tensor_value_info(name, proto, list(shape))
            for name, proto, shape in zip(output_names, output_protos, output_shapes)
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return ReferenceEvaluator(model).run(None, dict(zip(input_names, inputs)))


# 根据输出 dtype 选择精确比较或容差比较。
def _assert_matches(actual, expected, rtol=2e-3, atol=2e-3):
    actual_data = np.asarray(actual.data)
    expected_data = np.asarray(expected)
    assert actual_data.shape == expected_data.shape
    if np.issubdtype(expected_data.dtype, np.floating):
        np.testing.assert_allclose(actual_data, expected_data, rtol=rtol, atol=atol)
    else:
        np.testing.assert_array_equal(actual_data, expected_data)


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


BINARY_LEFT_F16 = np.array([[1.0, -2.0, 3.5], [4.0, 0.5, -1.0]], dtype=np.float16)
BINARY_RIGHT_F16 = np.array([[0.5], [2.0]], dtype=np.float16)
POW_LEFT_F32 = np.array([[1.0, 2.0, 3.0], [4.0, 1.5, 2.5]], dtype=np.float32)
POW_RIGHT_F32 = np.array([[2.0], [0.5]], dtype=np.float32)

BINARY_REFERENCE_CASES = [
    (ADD, "Add", {}, BINARY_LEFT_F16, BINARY_RIGHT_F16, TensorProto.FLOAT16),
    (SUB, "Sub", {}, BINARY_LEFT_F16, BINARY_RIGHT_F16, TensorProto.FLOAT16),
    (MUL, "Mul", {}, BINARY_LEFT_F16, BINARY_RIGHT_F16, TensorProto.FLOAT16),
    (DIV, "Div", {}, BINARY_LEFT_F16, BINARY_RIGHT_F16, TensorProto.FLOAT16),
    (Pow, "Pow", {}, POW_LEFT_F32, POW_RIGHT_F32, TensorProto.FLOAT),
]


# 验证基础二元算子的 ONNX 广播和 float16/float32 输出语义。
@pytest.mark.parametrize("op_cls,op_name,attrs,left,right,proto", BINARY_REFERENCE_CASES)
def test_c_backend_binary_elementwise_ops_match_onnx_reference(op_cls, op_name, attrs, left, right, proto):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    dtype = "float16" if proto == TensorProto.FLOAT16 else "float32"
    expected = _onnx_reference(
        op_name,
        [left, right],
        [proto, proto],
        attrs,
        [np.broadcast_shapes(left.shape, right.shape)],
    )[0]
    actual = op_cls(["a", "b"], ["y"], dtype=dtype, **attrs).forward(
        _tensor(left, dtype),
        _tensor(right, dtype),
    )["tensor"]
    _assert_matches(actual, expected)


# 验证 Max/Min 的多输入广播路径与 ONNX variadic elementwise 语义一致。
@pytest.mark.parametrize("op_cls,op_name", [(Max, "Max"), (Min, "Min")])
def test_c_backend_variadic_min_max_match_onnx_reference(op_cls, op_name):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    a = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]], dtype=np.float16)
    b = np.array([[0.5], [2.0]], dtype=np.float16)
    c = np.array([1.5, -3.0, 2.0], dtype=np.float16)
    expected = _onnx_reference(
        "Max" if op_name == "Max" else "Min",
        [a, b, c],
        [TensorProto.FLOAT16] * 3,
        {},
        [np.broadcast_shapes(a.shape, b.shape, c.shape)],
    )[0]
    actual = op_cls(["a", "b", "c"], ["y"], dtype="float16").forward(
        _tensor(a, "float16"),
        _tensor(b, "float16"),
        _tensor(c, "float16"),
    )["tensor"]
    _assert_matches(actual, expected)


COMPARISON_REFERENCE_CASES = [
    (Equal, "Equal"),
    (Greater, "Greater"),
    (Less, "Less"),
    (GreaterOrEqual, "GreaterOrEqual"),
    (LessOrEqual, "LessOrEqual"),
]


# 验证比较算子的广播和 bool 输出与 ONNX reference 一致。
@pytest.mark.parametrize("op_cls,op_name", COMPARISON_REFERENCE_CASES)
def test_c_backend_comparison_ops_match_onnx_reference(op_cls, op_name):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    left = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 0.0]], dtype=np.float16)
    right = np.array([[2.0], [2.0]], dtype=np.float16)
    expected = _onnx_reference(
        op_name,
        [left, right],
        [TensorProto.FLOAT16, TensorProto.FLOAT16],
        {},
        [np.broadcast_shapes(left.shape, right.shape)],
        [TensorProto.BOOL],
    )[0]
    actual = op_cls(["a", "b"], ["y"]).forward(_tensor(left, "float16"), _tensor(right, "float16"))["tensor"]
    _assert_matches(actual, expected)


LOGICAL_REFERENCE_CASES = [
    (And, "And"),
    (Or, "Or"),
    (Xor, "Xor"),
]


# 验证布尔逻辑算子的广播行为与 ONNX reference 一致。
@pytest.mark.parametrize("op_cls,op_name", LOGICAL_REFERENCE_CASES)
def test_c_backend_logical_binary_ops_match_onnx_reference(op_cls, op_name):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    left = np.array([[True, False, True], [False, True, False]], dtype=bool)
    right = np.array([[True], [False]], dtype=bool)
    expected = _onnx_reference(
        op_name,
        [left, right],
        [TensorProto.BOOL, TensorProto.BOOL],
        {},
        [np.broadcast_shapes(left.shape, right.shape)],
        [TensorProto.BOOL],
    )[0]
    actual = op_cls(["a", "b"], ["y"]).forward(_tensor(left, "bool"), _tensor(right, "bool"))["tensor"]
    _assert_matches(actual, expected)


# 验证 Not 的布尔逐元素取反语义。
def test_c_backend_not_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    values = np.array([[True, False, True], [False, True, False]], dtype=bool)
    expected = _onnx_reference("Not", [values], [TensorProto.BOOL], {}, [values.shape], [TensorProto.BOOL])[0]
    actual = Not(["x"], ["y"]).forward(_tensor(values, "bool"))["tensor"]
    _assert_matches(actual, expected)


UNARY_REFERENCE_CASES = [
    (RELU, "Relu", np.array([-2.0, -0.0, 0.5, 3.0], dtype=np.float16), TensorProto.FLOAT16, "float16"),
    (ABS, "Abs", np.array([-2.0, -0.0, 0.5, 3.0], dtype=np.float16), TensorProto.FLOAT16, "float16"),
    (Neg, "Neg", np.array([-2.0, -0.0, 0.5, 3.0], dtype=np.float16), TensorProto.FLOAT16, "float16"),
    (Floor, "Floor", np.array([-2.2, -0.1, 0.5, 3.7], dtype=np.float32), TensorProto.FLOAT, "float32"),
    (Sign, "Sign", np.array([-2.0, -0.0, 0.0, 3.0], dtype=np.float32), TensorProto.FLOAT, "float32"),
    (Sin, "Sin", np.array([-1.0, -0.25, 0.5, 1.0], dtype=np.float32), TensorProto.FLOAT, "float32"),
    (COS, "Cos", np.array([-1.0, -0.25, 0.5, 1.0], dtype=np.float32), TensorProto.FLOAT, "float32"),
    (Tan, "Tan", np.array([-1.0, -0.25, 0.5, 1.0], dtype=np.float32), TensorProto.FLOAT, "float32"),
    (Atan, "Atan", np.array([-2.0, -0.25, 0.5, 2.0], dtype=np.float32), TensorProto.FLOAT, "float32"),
    (EXP, "Exp", np.array([-2.0, -0.25, 0.5, 2.0], dtype=np.float32), TensorProto.FLOAT, "float32"),
    (LOG, "Log", np.array([0.25, 0.5, 1.0, 3.0], dtype=np.float32), TensorProto.FLOAT, "float32"),
    (SQRT, "Sqrt", np.array([0.0, 0.5, 1.0, 4.0], dtype=np.float32), TensorProto.FLOAT, "float32"),
    (SIGMOID, "Sigmoid", np.array([-4.0, -1.0, 0.5, 3.0], dtype=np.float32), TensorProto.FLOAT, "float32"),
    (TANH, "Tanh", np.array([-4.0, -1.0, 0.5, 3.0], dtype=np.float32), TensorProto.FLOAT, "float32"),
]


# 验证基础一元算子的官方输出语义。
@pytest.mark.parametrize("op_cls,op_name,values,proto,dtype", UNARY_REFERENCE_CASES)
def test_c_backend_unary_elementwise_ops_match_onnx_reference(op_cls, op_name, values, proto, dtype):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    expected = _onnx_reference(op_name, [values], [proto], {}, [values.shape])[0]
    actual = op_cls(["x"], ["y"], dtype=dtype).forward(_tensor(values, dtype))["tensor"]
    _assert_matches(actual, expected, rtol=2e-3, atol=2e-3)


# 验证 IsNaN 对 NaN、Inf 和普通值的 bool 输出与 ONNX reference 一致。
def test_c_backend_isnan_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    values = np.array([np.nan, -np.inf, 0.0, np.inf], dtype=np.float32)
    expected = _onnx_reference("IsNaN", [values], [TensorProto.FLOAT], {}, [values.shape], [TensorProto.BOOL])[0]
    actual = IsNaN(["x"], ["y"]).forward(_tensor(values, "float32"))["tensor"]
    _assert_matches(actual, expected)


# 验证 Softmax 的负轴归一化语义与 ONNX reference 一致。
def test_c_backend_softmax_negative_axis_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    values = np.array([[1.0, 2.0, 3.0], [1.0, -1.0, 0.5]], dtype=np.float32)
    expected = _onnx_reference("Softmax", [values], [TensorProto.FLOAT], {"axis": -1}, [values.shape])[0]
    actual = Softmax(["x"], ["y"], axis=-1, dtype="float32").forward(_tensor(values, "float32"))["tensor"]
    _assert_matches(actual, expected, rtol=1e-6, atol=1e-6)


# 验证 bfloat16 的读写路径确实按位解码输入并按低精度写回输出。
def test_c_backend_bfloat16_elementwise_decode_and_write_bit_storage():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    left_values = np.array([-2.0, -0.5, 0.0, 1.5], dtype=np.float32)
    right_values = np.array([0.5, -1.5, 2.0, 1.0], dtype=np.float32)
    left = _tensor(_bf16_bits(left_values), "bfloat16")
    right = _tensor(_bf16_bits(right_values), "bfloat16")

    add = ADD(["a", "b"], ["y"], dtype="bfloat16").forward(left, right)["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(add.data), left_values + right_values, rtol=1e-2, atol=1e-2)

    relu = RELU(["x"], ["y"], dtype="bfloat16").forward(left)["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(relu.data), np.maximum(left_values, 0.0), rtol=1e-2, atol=1e-2)

    max_out = Max(["a", "b"], ["y"], dtype="bfloat16").forward(left, right)["tensor"]
    np.testing.assert_allclose(
        _bf16_to_float32(max_out.data),
        np.maximum(left_values, right_values),
        rtol=1e-2,
        atol=1e-2,
    )

    equal = Equal(["a", "b"], ["y"]).forward(left, right)["tensor"]
    np.testing.assert_array_equal(equal.data, left_values == right_values)
