# /**
#   ******************************************************************************
#   * @file        test_operator_col2im_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用 ONNX reference 验证 Col2Im 算子的官方语义和混合精度路径。
#   * @details     2026.06.05  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.op_col2im import col2im_naive_implementation

from conftest import _disable_c_backend
from operator_test_context import *  # noqa: F401,F403
from nn.Operators import Col2Im


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


# 将 bfloat16 的 uint16 位模式解码成 float32，便于按官方 fold 公式计算期望输出。
def _bf16_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


# 构造 Tensor，避免每个断言重复 dtype、shape 和 data 样板。
def _tensor(data, dtype):
    return Tensor(*data.shape, dtype=dtype, data=data)


# 调用 ONNX reference evaluator，获得 Col2Im 的官方参考输出。
def _onnx_col2im_reference(x, image_shape, block_shape, attrs):
    channels = x.shape[1] // int(np.prod(block_shape, dtype=np.int64))
    output_shape = [x.shape[0], channels, *image_shape.tolist()]
    graph = helper.make_graph(
        [helper.make_node("Col2Im", ["x", "image_shape", "block_shape"], ["y"], **attrs)],
        "col2im_reference",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
            helper.make_tensor_value_info("image_shape", TensorProto.INT64, list(image_shape.shape)),
            helper.make_tensor_value_info("block_shape", TensorProto.INT64, list(block_shape.shape)),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    return ReferenceEvaluator(model).run(None, {"x": x, "image_shape": image_shape, "block_shape": block_shape})[0]


# 按官方 naive implementation 对每个 batch/channel 独立执行 fold，供 fallback 和 bfloat16 路径复用。
def _col2im_formula(x, image_shape, block_shape, attrs):
    pads = attrs.get("pads", [0] * (2 * len(image_shape)))
    strides = attrs.get("strides", [1] * len(image_shape))
    dilations = attrs.get("dilations", [1] * len(image_shape))
    block_size = int(np.prod(block_shape, dtype=np.int64))
    channels = x.shape[1] // block_size
    reshaped = x.reshape(x.shape[0], channels, block_size, x.shape[2])
    out = np.empty((x.shape[0], channels, *image_shape), dtype=x.dtype)
    for n in range(x.shape[0]):
        for c in range(channels):
            out[n, c] = col2im_naive_implementation(
                reshaped[n, c],
                tuple(image_shape),
                tuple(block_shape),
                dilations,
                pads,
                strides,
            )
    return out


# 验证 2D Col2Im 在 pads、strides、dilations 组合下与 ONNX 官方 reference 一致。
def test_c_backend_col2im_with_attrs_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    image_shape = np.array([4, 4], dtype=np.int64)
    block_shape = np.array([2, 2], dtype=np.int64)
    attrs = {"pads": [1, 0, 0, 1], "strides": [2, 1], "dilations": [1, 2]}
    x = np.linspace(-2.0, 2.0, 1 * 8 * 6, dtype=np.float32).reshape(1, 8, 6)
    expected = _onnx_col2im_reference(x, image_shape, block_shape, attrs)
    actual = Col2Im(["x", "image_shape", "block_shape"], ["y"], dtype="float32", **attrs).forward(
        _tensor(x, "float32"),
        _tensor(image_shape, "int64"),
        _tensor(block_shape, "int64"),
    )["tensor"]
    np.testing.assert_allclose(actual.data, expected, rtol=1e-6, atol=1e-6)


# 验证 bfloat16 路径按位解码输入、执行重叠累加，并按位写回输出。
def test_c_backend_col2im_bfloat16_decodes_and_writes_bit_storage():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    image_shape = np.array([3, 3], dtype=np.int64)
    block_shape = np.array([2, 2], dtype=np.int64)
    attrs = {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1]}
    x_values = np.linspace(-1.25, 1.25, 1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    x_bits = _bf16_bits(x_values)
    expected = _bf16_bits(_col2im_formula(_bf16_to_float32(x_bits), image_shape.tolist(), block_shape.tolist(), attrs))
    actual = Col2Im(["x", "image_shape", "block_shape"], ["y"], dtype="bfloat16", **attrs).forward(
        _tensor(x_bits, "bfloat16"),
        _tensor(image_shape, "int64"),
        _tensor(block_shape, "int64"),
    )["tensor"]
    np.testing.assert_array_equal(actual.data, expected)


# 验证 Python fallback 与官方 naive implementation 保持一致。
def test_python_col2im_fallback_matches_formula(monkeypatch):
    _disable_c_backend(monkeypatch)

    image_shape = np.array([3, 3], dtype=np.int64)
    block_shape = np.array([2, 2], dtype=np.int64)
    attrs = {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1]}
    x = np.linspace(-2.0, 2.0, 1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    expected = _col2im_formula(x, image_shape.tolist(), block_shape.tolist(), attrs)
    actual = Col2Im(["x", "image_shape", "block_shape"], ["y"], dtype="float32", **attrs).forward(
        _tensor(x, "float32"),
        _tensor(image_shape, "int64"),
        _tensor(block_shape, "int64"),
    )["tensor"]
    np.testing.assert_allclose(actual.data, expected, rtol=1e-6, atol=1e-6)


# 验证 ONNX 导入时保留 pads、strides 和 dilations 属性。
def test_onnx_import_col2im_preserves_attributes(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Col2Im", ["x", "image_shape", "block_shape"], ["y"], pads=[1, 0, 0, 1], strides=[2, 1], dilations=[1, 2])],
        "col2im_import",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 6]),
            helper.make_tensor_value_info("image_shape", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("block_shape", TensorProto.INT64, [2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4, 4])],
    )
    model_path = tmp_path / "col2im.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), model_path)

    imported = [op for op in ONNXImport(str(model_path), strict=True) if isinstance(op, Col2Im)]
    assert len(imported) == 1
    assert imported[0].pads == [1, 0, 0, 1]
    assert imported[0].strides == [2, 1]
    assert imported[0].dilations == [1, 2]
    assert imported[0].version == "18"
