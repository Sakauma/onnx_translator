# /**
#   ******************************************************************************
#   * @file        test_operator_deform_conv_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用 ONNX reference 验证 DeformConv 算子的官方语义和混合精度路径。
#   * @details     2026.06.05  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.op_deform_conv import _deform_conv_implementation

from conftest import _disable_c_backend
from operator_test_context import *  # noqa: F401,F403
from nn.Operators import DeformConv


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


# 将 bfloat16 的 uint16 位模式解码成 float32，便于按官方公式计算参考值。
def _bf16_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


# 构造 Tensor，避免每个断言重复 dtype、shape 和 data 样板。
def _tensor(data, dtype):
    return Tensor(*data.shape, dtype=dtype, data=data)


# 调用 ONNX reference evaluator，获得 DeformConv 官方参考输出。
def _onnx_deform_conv_reference(inputs, attrs, output_shape):
    names = ["x", "w", "offset"]
    protos = [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT]
    if len(inputs) >= 4 and inputs[3] is not None:
        names.append("b")
        protos.append(TensorProto.FLOAT)
    if len(inputs) >= 5 and inputs[4] is not None:
        names.append("mask")
        protos.append(TensorProto.FLOAT)
    graph = helper.make_graph(
        [helper.make_node("DeformConv", names, ["y"], **attrs)],
        "deform_conv_reference",
        [helper.make_tensor_value_info(name, proto, list(value.shape)) for name, proto, value in zip(names, protos, inputs) if value is not None],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(output_shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])
    return ReferenceEvaluator(model).run(None, {name: value for name, value in zip(names, inputs) if value is not None})[0]


# 验证 group、offset_group、bias 和 mask 组合与 ONNX 官方 reference 一致。
def test_c_backend_deform_conv_group_offset_mask_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    attrs = {"strides": [1, 1], "pads": [0, 0, 0, 0], "dilations": [1, 1], "group": 2, "offset_group": 2}
    x = np.linspace(-1.0, 1.0, 1 * 4 * 4 * 4, dtype=np.float32).reshape(1, 4, 4, 4)
    w = np.linspace(-0.5, 0.7, 4 * 2 * 2 * 2, dtype=np.float32).reshape(4, 2, 2, 2)
    offset = np.linspace(-0.2, 0.2, 1 * 16 * 3 * 3, dtype=np.float32).reshape(1, 16, 3, 3)
    b = np.linspace(-0.1, 0.2, 4, dtype=np.float32)
    mask = np.linspace(0.6, 1.0, 1 * 8 * 3 * 3, dtype=np.float32).reshape(1, 8, 3, 3)
    expected = _onnx_deform_conv_reference([x, w, offset, b, mask], attrs, (1, 4, 3, 3))
    actual = DeformConv(["x", "w", "offset", "b", "mask"], ["y"], dtype="float32", **attrs).forward(
        _tensor(x, "float32"),
        _tensor(w, "float32"),
        _tensor(offset, "float32"),
        _tensor(b, "float32"),
        _tensor(mask, "float32"),
    )["tensor"]
    np.testing.assert_allclose(actual.data, expected, rtol=2e-5, atol=2e-5)


# 验证 bfloat16 路径会解码低精度输入，并以 bfloat16 位存储写回输出。
def test_c_backend_deform_conv_bfloat16_decodes_and_writes_bit_storage():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    attrs = {"strides": [1, 1], "pads": [0, 0, 0, 0], "dilations": [1, 1], "group": 1, "offset_group": 1}
    x_values = np.linspace(-1.0, 1.0, 1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
    w_values = np.linspace(-0.5, 0.5, 2 * 2 * 3 * 3, dtype=np.float32).reshape(2, 2, 3, 3)
    offset_values = np.linspace(-0.15, 0.15, 1 * 18 * 2 * 2, dtype=np.float32).reshape(1, 18, 2, 2)
    bias_values = np.array([-0.1, 0.2], dtype=np.float32)
    mask_values = np.linspace(0.7, 1.0, 1 * 9 * 2 * 2, dtype=np.float32).reshape(1, 9, 2, 2)
    x_bits = _bf16_bits(x_values)
    w_bits = _bf16_bits(w_values)
    offset_bits = _bf16_bits(offset_values)
    bias_bits = _bf16_bits(bias_values)
    mask_bits = _bf16_bits(mask_values)
    expected = _deform_conv_implementation(
        _bf16_to_float32(x_bits),
        _bf16_to_float32(w_bits),
        _bf16_to_float32(offset_bits),
        _bf16_to_float32(bias_bits),
        _bf16_to_float32(mask_bits),
        attrs["dilations"],
        attrs["group"],
        [3, 3],
        attrs["offset_group"],
        attrs["pads"],
        attrs["strides"],
    )
    actual = DeformConv(["x", "w", "offset", "b", "mask"], ["y"], dtype="bfloat16", **attrs).forward(
        _tensor(x_bits, "bfloat16"),
        _tensor(w_bits, "bfloat16"),
        _tensor(offset_bits, "bfloat16"),
        _tensor(bias_bits, "bfloat16"),
        _tensor(mask_bits, "bfloat16"),
    )["tensor"]
    assert actual.data.dtype == np.uint16
    np.testing.assert_allclose(_bf16_to_float32(actual.data), _bf16_to_float32(_bf16_bits(expected)), rtol=1e-2, atol=2e-2)


# 验证 Python fallback 复用官方 reference 语义。
def test_python_deform_conv_fallback_matches_onnx_reference(monkeypatch):
    _disable_c_backend(monkeypatch)

    attrs = {"strides": [1, 1], "pads": [0, 0, 0, 0], "dilations": [1, 1], "group": 1, "offset_group": 1}
    x = np.linspace(-1.0, 1.0, 1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
    w = np.linspace(-0.5, 0.5, 2 * 2 * 3 * 3, dtype=np.float32).reshape(2, 2, 3, 3)
    offset = np.linspace(-0.1, 0.1, 1 * 18 * 2 * 2, dtype=np.float32).reshape(1, 18, 2, 2)
    expected = _onnx_deform_conv_reference([x, w, offset], attrs, (1, 2, 2, 2))
    actual = DeformConv(["x", "w", "offset"], ["y"], dtype="float32", **attrs).forward(
        _tensor(x, "float32"),
        _tensor(w, "float32"),
        _tensor(offset, "float32"),
    )["tensor"]
    np.testing.assert_allclose(actual.data, expected, rtol=2e-5, atol=2e-5)


# 验证 ONNX 导入时保留 strides、pads、dilations、group 和 offset_group 属性。
def test_onnx_import_deform_conv_preserves_attributes(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("DeformConv", ["x", "w", "offset"], ["y"], strides=[2, 1], pads=[1, 0, 1, 0], dilations=[1, 2], group=2, offset_group=2)],
        "deform_conv_import",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 6, 6]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [4, 2, 2, 2]),
            helper.make_tensor_value_info("offset", TensorProto.FLOAT, [1, 16, 4, 4]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 4, 4])],
    )
    model_path = tmp_path / "deform_conv.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)]), model_path)

    imported = [op for op in ONNXImport(str(model_path), strict=True) if isinstance(op, DeformConv)]
    assert len(imported) == 1
    assert imported[0].strides == [2, 1]
    assert imported[0].pads == [1, 0, 1, 0]
    assert imported[0].dilations == [1, 2]
    assert imported[0].group == 2
    assert imported[0].offset_group == 2
    assert imported[0].version == "22"
