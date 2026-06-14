# /**
#   ******************************************************************************
#   * @file        test_operator_core_numeric_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用 ONNX reference 验证核心卷积、矩阵、池化和量化算子的官方语义。
#   * @details     2026.06.05  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from onnx.reference import ReferenceEvaluator

from operator_test_context import *  # noqa: F401,F403
from nn.Operators import (
    AveragePool,
    Clip,
    Conv,
    ConvInteger,
    ConvTranspose,
    DequantizeLinear,
    Gemm,
    GlobalAveragePool,
    GlobalLpPool,
    GlobalMaxPool,
    LpPool,
    MatMul,
    MatMulInteger,
    MaxPool,
    Mod,
    QLinearConv,
    QLinearMatMul,
    QuantizeLinear,
)


# 构造 Tensor，避免测试主体重复 shape、dtype 和 data 样板。
def _tensor(data, dtype):
    data = np.asarray(data)
    return Tensor(*data.shape, dtype=dtype, data=data)


# 调用 ONNX reference evaluator，返回单节点模型的官方输出。
def _onnx_reference(op_name, inputs, protos, attrs, output_shapes, output_protos=None, opset=17):
    output_protos = output_protos or [protos[0]] * len(output_shapes)
    input_names = [f"i{i}" for i in range(len(inputs))]
    output_names = [f"o{i}" for i in range(len(output_shapes))]
    graph = helper.make_graph(
        [helper.make_node(op_name, input_names, output_names, **attrs)],
        f"{op_name}_reference",
        [
            helper.make_tensor_value_info(name, proto, list(np.asarray(value).shape))
            for name, proto, value in zip(input_names, protos, inputs)
        ],
        [
            helper.make_tensor_value_info(name, proto, list(shape))
            for name, proto, shape in zip(output_names, output_protos, output_shapes)
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return ReferenceEvaluator(model).run(None, dict(zip(input_names, inputs)))


# 对浮点输出使用容差比较，对整数和布尔输出使用精确比较。
def _assert_tensor_matches(actual, expected, rtol=1e-4, atol=1e-4):
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


# 将 bfloat16 的 uint16 位模式解码成 float32，用于混合精度断言。
def _bf16_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


# 验证 Conv 的 group、pads、strides 和 dilations 属性与官方 reference 一致。
def test_c_backend_conv_group_dilation_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = (np.arange(1 * 4 * 5 * 5, dtype=np.float32).reshape(1, 4, 5, 5) - 20.0) / 10.0
    w = (np.arange(6 * 2 * 2 * 2, dtype=np.float32).reshape(6, 2, 2, 2) - 12.0) / 9.0
    b = np.linspace(-0.3, 0.3, 6, dtype=np.float32)
    attrs = {
        "group": 2,
        "pads": [1, 0, 1, 0],
        "strides": [1, 2],
        "dilations": [2, 1],
    }
    expected = _onnx_reference(
        "Conv",
        [x, w, b],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
        attrs,
        [(1, 6, 5, 2)],
    )[0]
    actual = Conv(["x", "w", "b"], ["y"], dtype="float32", kernel_shape=[2, 2], **attrs).forward(
        _tensor(x, "float32"),
        _tensor(w, "float32"),
        _tensor(b, "float32"),
    )["tensor"]
    _assert_tensor_matches(actual, expected, rtol=1e-5, atol=1e-5)


# 验证 ConvTranspose 的 pads、strides、dilations 和 output_padding 属性。
def test_c_backend_conv_transpose_output_padding_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = (np.arange(1 * 2 * 3 * 3, dtype=np.float32).reshape(1, 2, 3, 3) - 4.0) / 5.0
    w = (np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2) - 6.0) / 7.0
    b = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    attrs = {
        "pads": [1, 0, 1, 0],
        "strides": [2, 2],
        "dilations": [1, 1],
        "output_padding": [1, 0],
    }
    expected = _onnx_reference(
        "ConvTranspose",
        [x, w, b],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
        attrs,
        [(1, 3, 6, 6)],
    )[0]
    actual = ConvTranspose(["x", "w", "b"], ["y"], dtype="float32", kernel_shape=[2, 2], group=1, **attrs).forward(
        _tensor(x, "float32"),
        _tensor(w, "float32"),
        _tensor(b, "float32"),
    )["tensor"]
    _assert_tensor_matches(actual, expected, rtol=1e-5, atol=1e-5)


# 验证 ConvInteger 和 QLinearConv 的零点、per-output-channel scale 语义。
def test_c_backend_integer_conv_ops_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.array([[[[127, 128, 129], [130, 131, 132], [133, 134, 135]]]], dtype=np.uint8)
    w = np.array(
        [
            [[[127, 128], [129, 130]]],
            [[[130, 129], [128, 127]]],
        ],
        dtype=np.uint8,
    )
    x_zero_point = np.array([128], dtype=np.uint8)
    w_zero_point = np.array([128, 129], dtype=np.uint8)
    attrs = {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "group": 1}
    conv_integer_expected = _onnx_reference(
        "ConvInteger",
        [x, w, x_zero_point, w_zero_point],
        [TensorProto.UINT8, TensorProto.UINT8, TensorProto.UINT8, TensorProto.UINT8],
        attrs,
        [(1, 2, 2, 2)],
        [TensorProto.INT32],
    )[0]
    conv_integer_actual = ConvInteger(["x", "w", "xzp", "wzp"], ["y"], kernel_shape=[2, 2], **attrs).forward(
        _tensor(x, "uint8"),
        _tensor(w, "uint8"),
        _tensor(x_zero_point, "uint8"),
        _tensor(w_zero_point, "uint8"),
    )["tensor"]
    _assert_tensor_matches(conv_integer_actual, conv_integer_expected)

    x_scale = np.array([0.1], dtype=np.float32)
    w_scale = np.array([0.2, 0.15], dtype=np.float32)
    y_scale = np.array([0.05], dtype=np.float32)
    y_zero_point = np.array([10], dtype=np.uint8)
    qlinear_expected = _onnx_reference(
        "QLinearConv",
        [x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point],
        [
            TensorProto.UINT8,
            TensorProto.FLOAT,
            TensorProto.UINT8,
            TensorProto.UINT8,
            TensorProto.FLOAT,
            TensorProto.UINT8,
            TensorProto.FLOAT,
            TensorProto.UINT8,
        ],
        attrs,
        [(1, 2, 2, 2)],
        [TensorProto.UINT8],
    )[0]
    qlinear_actual = QLinearConv(
        ["x", "xs", "xzp", "w", "ws", "wzp", "ys", "yzp"],
        ["y"],
        kernel_shape=[2, 2],
        dtype="uint8",
        **attrs,
    ).forward(
        _tensor(x, "uint8"),
        _tensor(x_scale, "float32"),
        _tensor(x_zero_point, "uint8"),
        _tensor(w, "uint8"),
        _tensor(w_scale, "float32"),
        _tensor(w_zero_point, "uint8"),
        _tensor(y_scale, "float32"),
        _tensor(y_zero_point, "uint8"),
    )["tensor"]
    _assert_tensor_matches(qlinear_actual, qlinear_expected)


# 验证 Gemm 的转置、alpha/beta 和 C 广播语义。
def test_c_backend_gemm_transpose_alpha_beta_and_bias_broadcast_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    a = np.array([[1.0, -2.0], [3.0, 0.5], [-1.0, 2.0]], dtype=np.float32)
    b = np.array([[0.25, -0.5], [1.5, 2.0], [-1.0, 0.75]], dtype=np.float32)
    c = np.array([0.5, -1.5], dtype=np.float32)
    attrs = {"alpha": 0.75, "beta": 1.25, "transA": 1, "transB": 0}
    expected = _onnx_reference(
        "Gemm",
        [a, b, c],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
        attrs,
        [(2, 2)],
    )[0]
    actual = Gemm(["a", "b", "c"], ["y"], dtype="float32", **attrs).forward(
        _tensor(a, "float32"),
        _tensor(b, "float32"),
        _tensor(c, "float32"),
    )["tensor"]
    _assert_tensor_matches(actual, expected, rtol=1e-5, atol=1e-5)


# 验证 MatMul 的批量广播和一维输入压缩输出语义。
def test_c_backend_matmul_batch_broadcast_and_vector_shapes_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    a = (np.arange(2 * 1 * 3 * 4, dtype=np.float32).reshape(2, 1, 3, 4) - 7.0) / 8.0
    b = (np.arange(1 * 5 * 4 * 2, dtype=np.float32).reshape(1, 5, 4, 2) + 3.0) / 6.0
    expected = _onnx_reference(
        "MatMul",
        [a, b],
        [TensorProto.FLOAT, TensorProto.FLOAT],
        {},
        [(2, 5, 3, 2)],
    )[0]
    actual = MatMul(["a", "b"], ["y"], dtype="float32").forward(_tensor(a, "float32"), _tensor(b, "float32"))["tensor"]
    _assert_tensor_matches(actual, expected, rtol=1e-5, atol=1e-5)

    vec = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    matrix = np.arange(4 * 3, dtype=np.float32).reshape(4, 3) / 4.0
    expected_vec = _onnx_reference(
        "MatMul",
        [vec, matrix],
        [TensorProto.FLOAT, TensorProto.FLOAT],
        {},
        [(3,)],
    )[0]
    actual_vec = MatMul(["a", "b"], ["y"], dtype="float32").forward(_tensor(vec, "float32"), _tensor(matrix, "float32"))["tensor"]
    _assert_tensor_matches(actual_vec, expected_vec, rtol=1e-5, atol=1e-5)


# 验证 MaxPool 的 ceil_mode/dilations 以及索引输出语义。
def test_max_pool_dilations_ceil_and_indices_match_onnx_reference():
    x = np.array(
        [[[[1.0, 3.0, 2.0, 4.0], [5.0, 6.0, 1.0, 0.0], [2.0, 9.0, 8.0, 7.0]]]],
        dtype=np.float32,
    )
    attrs = {
        "kernel_shape": [2, 2],
        "pads": [0, 1, 1, 0],
        "strides": [2, 1],
        "dilations": [1, 2],
        "ceil_mode": 1,
        "storage_order": 0,
    }
    expected_y, expected_indices = _onnx_reference(
        "MaxPool",
        [x],
        [TensorProto.FLOAT],
        attrs,
        [(1, 1, 2, 3), (1, 1, 2, 3)],
        [TensorProto.FLOAT, TensorProto.INT64],
    )
    actual_y, actual_indices = MaxPool(["x"], ["y", "indices"], dtype="float32", **attrs).forward(_tensor(x, "float32"))["tensor"]
    _assert_tensor_matches(actual_y, expected_y)
    _assert_tensor_matches(actual_indices, expected_indices)


# 验证 AveragePool 和 LpPool 的 padding、ceil_mode 和 count_include_pad 属性。
def test_pooling_ops_match_onnx_reference_for_padding_and_ceil_mode():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.array(
        [[[[1.0, 2.0, 3.0], [4.0, -5.0, 6.0], [7.0, 8.0, 9.0]]]],
        dtype=np.float32,
    )
    avg_attrs = {
        "kernel_shape": [2, 2],
        "pads": [1, 1, 0, 0],
        "strides": [2, 2],
        "count_include_pad": 1,
        "ceil_mode": 1,
    }
    avg_expected = _onnx_reference(
        "AveragePool",
        [x],
        [TensorProto.FLOAT],
        avg_attrs,
        [(1, 1, 2, 2)],
    )[0]
    avg_actual = AveragePool(["x"], ["y"], dtype="float32", dilations=[1, 1], **avg_attrs).forward(_tensor(x, "float32"))["tensor"]
    _assert_tensor_matches(avg_actual, avg_expected, rtol=1e-5, atol=1e-5)

    lp_attrs = {
        "kernel_shape": [2, 2],
        "pads": [0, 0, 1, 1],
        "strides": [1, 2],
        "p": 3,
        "ceil_mode": 1,
    }
    lp_expected = _onnx_reference(
        "LpPool",
        [x],
        [TensorProto.FLOAT],
        lp_attrs,
        [(1, 1, 3, 2)],
    )[0]
    lp_actual = LpPool(["x"], ["y"], dtype="float32", dilations=[1, 1], **lp_attrs).forward(_tensor(x, "float32"))["tensor"]
    _assert_tensor_matches(lp_actual, lp_expected, rtol=1e-5, atol=1e-5)


# 验证全局池化族在二维空间输入上的官方输出形状和数值语义。
def test_global_pooling_ops_match_onnx_reference_for_4d_input():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = (np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4) - 10.0) / 3.0
    for op_cls, op_name, attrs in [
        (GlobalAveragePool, "GlobalAveragePool", {}),
        (GlobalMaxPool, "GlobalMaxPool", {}),
    ]:
        expected = _onnx_reference(op_name, [x], [TensorProto.FLOAT], attrs, [(2, 2, 1, 1)])[0]
        actual = op_cls(["x"], ["y"], dtype="float32", **attrs).forward(_tensor(x, "float32"))["tensor"]
        _assert_tensor_matches(actual, expected, rtol=1e-5, atol=1e-5)

    global_lp_expected = np.sum(np.abs(x) ** 3, axis=(2, 3), keepdims=True) ** (1.0 / 3.0)
    global_lp_actual = GlobalLpPool(["x"], ["y"], p=3, dtype="float32").forward(_tensor(x, "float32"))["tensor"]
    _assert_tensor_matches(global_lp_actual, global_lp_expected, rtol=1e-5, atol=1e-5)


# 验证 QuantizeLinear 和 DequantizeLinear 的负 axis per-axis 语义。
def test_c_backend_quantize_and_dequantize_negative_axis_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.array(
        [[[0.0, 0.25, -0.5], [1.0, -1.25, 2.0]], [[-0.75, 0.5, 1.5], [2.5, -2.0, 0.75]]],
        dtype=np.float32,
    )
    scale = np.array([0.1, 0.25, 0.5], dtype=np.float32)
    zero_point = np.array([10, -3, 4], dtype=np.int8)
    quant_expected = _onnx_reference(
        "QuantizeLinear",
        [x, scale, zero_point],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT8],
        {"axis": -1},
        [x.shape],
        [TensorProto.INT8],
    )[0]
    quant_actual = QuantizeLinear(["x", "scale", "zp"], ["y"], axis=-1, dtype="int8").forward(
        _tensor(x, "float32"),
        _tensor(scale, "float32"),
        _tensor(zero_point, "int8"),
    )["tensor"]
    _assert_tensor_matches(quant_actual, quant_expected)

    dequant_expected = _onnx_reference(
        "DequantizeLinear",
        [quant_expected, scale, zero_point],
        [TensorProto.INT8, TensorProto.FLOAT, TensorProto.INT8],
        {"axis": -1},
        [x.shape],
        [TensorProto.FLOAT],
        opset=19,
    )[0]
    dequant_actual = DequantizeLinear(["x", "scale", "zp"], ["y"], axis=-1, dtype="float32").forward(
        _tensor(quant_expected, "int8"),
        _tensor(scale, "float32"),
        _tensor(zero_point, "int8"),
    )["tensor"]
    _assert_tensor_matches(dequant_actual, dequant_expected, rtol=1e-6, atol=1e-6)


# 验证 output_dtype 属性在 zero_point 缺省时决定 Q/DQ 输出类型。
def test_c_backend_quantize_dequantize_output_dtype_without_zero_point_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.array(
        [[-12.8, -0.25, 0.25, 12.7], [-6.4, -1.0, 1.0, 6.4]],
        dtype=np.float32,
    )
    scale = np.array([0.1, 0.25, 0.5, 1.25], dtype=np.float32)
    quant_expected = _onnx_reference(
        "QuantizeLinear",
        [x, scale],
        [TensorProto.FLOAT, TensorProto.FLOAT],
        {"axis": -1, "output_dtype": TensorProto.INT8},
        [x.shape],
        [TensorProto.INT8],
        opset=25,
    )[0]
    quant_actual = QuantizeLinear(["x", "scale"], ["y"], axis=-1, output_dtype="int8").forward(
        _tensor(x, "float32"),
        _tensor(scale, "float32"),
    )["tensor"]
    _assert_tensor_matches(quant_actual, quant_expected)

    dequant_expected = (
        quant_expected.astype(np.float32) * scale.reshape(1, scale.size).astype(np.float32)
    ).astype(np.float16)
    dequant_actual = DequantizeLinear(["x", "scale"], ["y"], axis=-1, output_dtype="float16").forward(
        _tensor(quant_expected, "int8"),
        _tensor(scale, "float32"),
    )["tensor"]
    assert dequant_actual.dtype == "float16"
    _assert_tensor_matches(dequant_actual, dequant_expected, rtol=1e-3, atol=1e-3)


# 验证 QuantizeLinear 的 precision=DOUBLE 会真实改变除法精度，而不是只保存属性。
def test_c_backend_quantize_linear_precision_double_uses_double_division():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = _tensor(np.array([-12.75], dtype=np.float32), "float32")
    scale = _tensor(np.array([0.1], dtype=np.float32), "float32")
    zp = _tensor(np.array([0], dtype=np.int8), "int8")

    default_actual = QuantizeLinear(["x", "scale", "zp"], ["y"], dtype="int8").forward(x, scale, zp)["tensor"]
    double_actual = QuantizeLinear(["x", "scale", "zp"], ["y"], dtype="int8", precision=TensorProto.DOUBLE).forward(x, scale, zp)["tensor"]

    np.testing.assert_array_equal(default_actual.data, np.array([-128], dtype=np.int8))
    np.testing.assert_array_equal(double_actual.data, np.array([-127], dtype=np.int8))


# 验证 QuantizeLinear/DequantizeLinear 的 int16 与 uint16 官方 dtype 约束。
def test_c_backend_quantize_and_dequantize_16bit_integer_dtypes_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x_i16 = np.array([-40000.0, -123.4, -0.5, 0.0, 0.5, 123.4, 32767.4, 40000.0], dtype=np.float32)
    scale_i16 = np.array([1.0], dtype=np.float32)
    zp_i16 = np.array([0], dtype=np.int16)
    quant_i16_expected = _onnx_reference(
        "QuantizeLinear",
        [x_i16, scale_i16, zp_i16],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT16],
        {},
        [x_i16.shape],
        [TensorProto.INT16],
        opset=25,
    )[0]
    quant_i16_actual = QuantizeLinear(["x", "scale", "zp"], ["y"], dtype="int16").forward(
        _tensor(x_i16, "float32"),
        _tensor(scale_i16, "float32"),
        _tensor(zp_i16, "int16"),
    )["tensor"]
    _assert_tensor_matches(quant_i16_actual, quant_i16_expected)

    x_u16 = np.array([-10.0, -0.5, 0.0, 0.5, 123.4, 40000.0, 70000.0, 100000.0], dtype=np.float32)
    scale_u16 = np.array([1.0], dtype=np.float32)
    zp_u16 = np.array([5], dtype=np.uint16)
    quant_u16_expected = _onnx_reference(
        "QuantizeLinear",
        [x_u16, scale_u16, zp_u16],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.UINT16],
        {},
        [x_u16.shape],
        [TensorProto.UINT16],
        opset=25,
    )[0]
    quant_u16_actual = QuantizeLinear(["x", "scale", "zp"], ["y"], dtype="uint16").forward(
        _tensor(x_u16, "float32"),
        _tensor(scale_u16, "float32"),
        _tensor(zp_u16, "uint16"),
    )["tensor"]
    _assert_tensor_matches(quant_u16_actual, quant_u16_expected)

    dequant_i16_expected = _onnx_reference(
        "DequantizeLinear",
        [quant_i16_expected, np.array([0.5], dtype=np.float32), np.array([-3], dtype=np.int16)],
        [TensorProto.INT16, TensorProto.FLOAT, TensorProto.INT16],
        {},
        [x_i16.shape],
        [TensorProto.FLOAT],
        opset=25,
    )[0]
    dequant_i16_actual = DequantizeLinear(["x", "scale", "zp"], ["y"], dtype="float32").forward(
        _tensor(quant_i16_expected, "int16"),
        _tensor(np.array([0.5], dtype=np.float32), "float32"),
        _tensor(np.array([-3], dtype=np.int16), "int16"),
    )["tensor"]
    _assert_tensor_matches(dequant_i16_actual, dequant_i16_expected, rtol=1e-6, atol=1e-6)

    dequant_u16_expected = _onnx_reference(
        "DequantizeLinear",
        [quant_u16_expected, np.array([0.25], dtype=np.float32), np.array([5], dtype=np.uint16)],
        [TensorProto.UINT16, TensorProto.FLOAT, TensorProto.UINT16],
        {},
        [x_u16.shape],
        [TensorProto.FLOAT],
        opset=25,
    )[0]
    dequant_u16_actual = DequantizeLinear(["x", "scale", "zp"], ["y"], dtype="float32").forward(
        _tensor(quant_u16_expected, "uint16"),
        _tensor(np.array([0.25], dtype=np.float32), "float32"),
        _tensor(np.array([5], dtype=np.uint16), "uint16"),
    )["tensor"]
    _assert_tensor_matches(dequant_u16_actual, dequant_u16_expected, rtol=1e-6, atol=1e-6)


# 验证 blocked quantization 的 scale/zero_point 块映射语义。
def test_c_backend_quantize_and_dequantize_block_size_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = (np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) - 7.0) / 3.0
    scale = np.array(
        [
            [[0.10, 0.20, 0.25, 0.50], [0.30, 0.40, 0.60, 0.80]],
            [[0.15, 0.35, 0.45, 0.55], [0.25, 0.50, 0.75, 1.00]],
        ],
        dtype=np.float32,
    )
    zero_point = np.array(
        [
            [[-5, -4, -3, -2], [1, 2, 3, 4]],
            [[-8, -6, -4, -2], [2, 4, 6, 8]],
        ],
        dtype=np.int8,
    )
    attrs = {"axis": 1, "block_size": 2}
    quant_expected = _onnx_reference(
        "QuantizeLinear",
        [x, scale, zero_point],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT8],
        attrs,
        [x.shape],
        [TensorProto.INT8],
        opset=25,
    )[0]
    quant_actual = QuantizeLinear(["x", "scale", "zp"], ["y"], axis=1, block_size=2, dtype="int8").forward(
        _tensor(x, "float32"),
        _tensor(scale, "float32"),
        _tensor(zero_point, "int8"),
    )["tensor"]
    _assert_tensor_matches(quant_actual, quant_expected)

    dequant_expected = _onnx_reference(
        "DequantizeLinear",
        [quant_expected, scale, zero_point],
        [TensorProto.INT8, TensorProto.FLOAT, TensorProto.INT8],
        attrs,
        [x.shape],
        [TensorProto.FLOAT],
        opset=25,
    )[0]
    dequant_actual = DequantizeLinear(["x", "scale", "zp"], ["y"], axis=1, block_size=2, dtype="float32").forward(
        _tensor(quant_expected, "int8"),
        _tensor(scale, "float32"),
        _tensor(zero_point, "int8"),
    )["tensor"]
    _assert_tensor_matches(dequant_actual, dequant_expected, rtol=1e-6, atol=1e-6)


# 验证整数矩阵乘法和量化矩阵乘法的零点、scale 广播语义。
def test_c_backend_integer_matmul_ops_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    a = np.array([[130, 132, 134], [136, 138, 140]], dtype=np.uint8)
    b = np.array([[2, -3], [4, 5], [-6, 7]], dtype=np.int8)
    a_zero_point = np.array([128], dtype=np.uint8)
    b_zero_point = np.array([1, -2], dtype=np.int8)
    matmul_integer_expected = _onnx_reference(
        "MatMulInteger",
        [a, b, a_zero_point, b_zero_point],
        [TensorProto.UINT8, TensorProto.INT8, TensorProto.UINT8, TensorProto.INT8],
        {},
        [(2, 2)],
        [TensorProto.INT32],
    )[0]
    matmul_integer_actual = MatMulInteger(["a", "b", "azp", "bzp"], ["y"]).forward(
        _tensor(a, "uint8"),
        _tensor(b, "int8"),
        _tensor(a_zero_point, "uint8"),
        _tensor(b_zero_point, "int8"),
    )["tensor"]
    _assert_tensor_matches(matmul_integer_actual, matmul_integer_expected)

    a_scale = np.array([0.05], dtype=np.float32)
    b_scale = np.array([0.1], dtype=np.float32)
    y_scale = np.array([0.2], dtype=np.float32)
    y_zero_point = np.array([11], dtype=np.uint8)
    b_uint8 = np.array([[122, 117], [124, 125], [114, 127]], dtype=np.uint8)
    b_uint8_zero_point = np.array([120], dtype=np.uint8)
    qlinear_expected = _onnx_reference(
        "QLinearMatMul",
        [a, a_scale, a_zero_point, b_uint8, b_scale, b_uint8_zero_point, y_scale, y_zero_point],
        [
            TensorProto.UINT8,
            TensorProto.FLOAT,
            TensorProto.UINT8,
            TensorProto.UINT8,
            TensorProto.FLOAT,
            TensorProto.UINT8,
            TensorProto.FLOAT,
            TensorProto.UINT8,
        ],
        {},
        [(2, 2)],
        [TensorProto.UINT8],
    )[0]
    qlinear_actual = QLinearMatMul(["a", "as", "azp", "b", "bs", "bzp", "ys", "yzp"], ["y"], dtype="uint8").forward(
        _tensor(a, "uint8"),
        _tensor(a_scale, "float32"),
        _tensor(a_zero_point, "uint8"),
        _tensor(b_uint8, "uint8"),
        _tensor(b_scale, "float32"),
        _tensor(b_uint8_zero_point, "uint8"),
        _tensor(y_scale, "float32"),
        _tensor(y_zero_point, "uint8"),
    )["tensor"]
    _assert_tensor_matches(qlinear_actual, qlinear_expected)


# 验证 Clip 的可选 min/max 输入和 Mod 的 fmod 属性语义。
def test_c_backend_clip_optional_bounds_and_mod_fmod_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.array([[-2.0, -0.5, 0.5, 2.0]], dtype=np.float32)
    min_bound = np.array([-1.0], dtype=np.float32)
    clip_min_expected = _onnx_reference(
        "Clip",
        [x, min_bound],
        [TensorProto.FLOAT, TensorProto.FLOAT],
        {},
        [x.shape],
    )[0]
    clip_min_actual = Clip(["x", "min"], ["y"], dtype="float32").forward(
        _tensor(x, "float32"),
        _tensor(min_bound, "float32"),
    )["tensor"]
    _assert_tensor_matches(clip_min_actual, clip_min_expected)

    int_a = np.array([[-7, -7, 7, 7]], dtype=np.int32)
    int_b = np.array([[3, -3, 3, -3]], dtype=np.int32)
    mod_expected = _onnx_reference(
        "Mod",
        [int_a, int_b],
        [TensorProto.INT32, TensorProto.INT32],
        {"fmod": 0},
        [int_a.shape],
    )[0]
    mod_actual = Mod(["a", "b"], ["y"], dtype="int32", fmod=0).forward(_tensor(int_a, "int32"), _tensor(int_b, "int32"))["tensor"]
    _assert_tensor_matches(mod_actual, mod_expected)

    float_a = np.array([[-7.5, -7.5, 7.5, 7.5]], dtype=np.float32)
    float_b = np.array([[3.0, -3.0, 3.0, -3.0]], dtype=np.float32)
    fmod_expected = _onnx_reference(
        "Mod",
        [float_a, float_b],
        [TensorProto.FLOAT, TensorProto.FLOAT],
        {"fmod": 1},
        [float_a.shape],
    )[0]
    fmod_actual = Mod(["a", "b"], ["y"], dtype="float32", fmod=1).forward(
        _tensor(float_a, "float32"),
        _tensor(float_b, "float32"),
    )["tensor"]
    _assert_tensor_matches(fmod_actual, fmod_expected)


# 验证 bfloat16 在矩阵、池化和量化周边路径中按位读取并写回。
def test_c_backend_core_numeric_bfloat16_decode_and_write_bit_storage():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    left = np.array([[1.0, -2.0], [3.5, 0.5]], dtype=np.float32)
    right = np.array([[2.0, -1.0], [0.25, 4.0]], dtype=np.float32)
    matmul = MatMul(["a", "b"], ["y"], dtype="bfloat16").forward(
        _tensor(_bf16_bits(left), "bfloat16"),
        _tensor(_bf16_bits(right), "bfloat16"),
    )["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(matmul.data), np.matmul(left, right), rtol=2e-2, atol=2e-2)

    pool_input = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    avg = AveragePool(
        ["x"],
        ["y"],
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
        dtype="bfloat16",
        dilations=[1, 1],
    ).forward(_tensor(_bf16_bits(pool_input), "bfloat16"))["tensor"]
    np.testing.assert_array_equal(avg.data, _bf16_bits(np.array([[[[2.5]]]], dtype=np.float32)))

    quant_input = np.array([0.0, 0.25, -0.25], dtype=np.float32)
    scale = np.array([0.25], dtype=np.float32)
    zero_point = np.array([0], dtype=np.int8)
    quantized = QuantizeLinear(["x", "scale", "zp"], ["y"], dtype="int8").forward(
        _tensor(_bf16_bits(quant_input), "bfloat16"),
        _tensor(_bf16_bits(scale), "bfloat16"),
        _tensor(zero_point, "int8"),
    )["tensor"]
    np.testing.assert_array_equal(quantized.data, np.array([0, 1, -1], dtype=np.int8))
