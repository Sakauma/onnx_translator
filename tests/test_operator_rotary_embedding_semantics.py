# /**
#   ******************************************************************************
#   * @file        test_operator_rotary_embedding_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用 ONNX reference 验证 RotaryEmbedding 算子的官方语义和混合精度路径。
#   * @details     2026.06.05  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from onnx.reference import ReferenceEvaluator

from conftest import _disable_c_backend
from operator_test_context import *  # noqa: F401,F403
from nn.Operators import RotaryEmbedding


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


# 调用 ONNX reference evaluator，获得 RotaryEmbedding 的官方参考输出。
def _onnx_rotary_reference(inputs, protos, attrs, output_shape):
    names = ["x", "cos", "sin"]
    if len(inputs) == 4:
        names.append("pos")
    graph = helper.make_graph(
        [helper.make_node("RotaryEmbedding", names, ["y"], **attrs)],
        "rotary_embedding_reference",
        [helper.make_tensor_value_info(name, proto, list(value.shape)) for name, proto, value in zip(names, protos, inputs)],
        [helper.make_tensor_value_info("y", protos[0], list(output_shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    return ReferenceEvaluator(model).run(None, dict(zip(names, inputs)))[0]


# 按 ONNX schema 中的 RoPE 公式独立计算输出，供 bfloat16 位存储路径复用。
def _rotary_formula(x, cos_cache, sin_cache, position_ids=None, interleaved=0, rotary_embedding_dim=0, num_heads=0):
    original_shape = x.shape
    work = np.transpose(x, (0, 2, 1, 3)) if x.ndim == 4 else x.reshape(x.shape[0], x.shape[1], num_heads, x.shape[2] // num_heads)
    head_size = work.shape[-1]
    rotary_dim = rotary_embedding_dim or head_size
    half = rotary_dim // 2
    x_rotate = work[..., :rotary_dim]
    x_not_rotate = work[..., rotary_dim:]
    if position_ids is not None:
        cos_cache = cos_cache[position_ids]
        sin_cache = sin_cache[position_ids]
    cos_cache = np.expand_dims(cos_cache, axis=2)
    sin_cache = np.expand_dims(sin_cache, axis=2)
    if interleaved:
        x1 = x_rotate[..., 0::2]
        x2 = x_rotate[..., 1::2]
    else:
        x1 = x_rotate[..., :half]
        x2 = x_rotate[..., half:]
    real = cos_cache * x1 - sin_cache * x2
    imag = sin_cache * x1 + cos_cache * x2
    rotated = np.stack((real, imag), axis=-1).reshape(x_rotate.shape) if interleaved else np.concatenate((real, imag), axis=-1)
    out = np.concatenate((rotated, x_not_rotate), axis=-1)
    return out.reshape(original_shape) if x.ndim == 3 else np.transpose(out, (0, 2, 1, 3))


# 验证 4D 输入携带 position_ids 时与 ONNX 官方 reference 一致。
def test_c_backend_rotary_embedding_4d_position_ids_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.linspace(-1.5, 1.5, 2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4)
    angles = np.linspace(0.0, 1.0, 6 * 2, dtype=np.float32).reshape(6, 2)
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    pos = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    expected = _onnx_rotary_reference(
        [x, cos, sin, pos],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64],
        {},
        x.shape,
    )
    actual = RotaryEmbedding(["x", "cos", "sin", "pos"], ["y"], dtype="float32").forward(
        _tensor(x, "float32"),
        _tensor(cos, "float32"),
        _tensor(sin, "float32"),
        _tensor(pos, "int64"),
    )["tensor"]
    np.testing.assert_allclose(actual.data, expected, rtol=1e-6, atol=1e-6)


# 验证 3D 输入、interleaved 和 partial rotary dim 组合与 ONNX 官方 reference 一致。
def test_c_backend_rotary_embedding_3d_interleaved_partial_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.linspace(-1.0, 1.0, 2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8)
    angles = np.linspace(0.0, 0.5, 2 * 3 * 1, dtype=np.float32).reshape(2, 3, 1)
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    attrs = {"num_heads": 2, "rotary_embedding_dim": 2, "interleaved": 1}
    expected = _onnx_rotary_reference(
        [x, cos, sin],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
        attrs,
        x.shape,
    )
    actual = RotaryEmbedding(["x", "cos", "sin"], ["y"], dtype="float32", **attrs).forward(
        _tensor(x, "float32"),
        _tensor(cos, "float32"),
        _tensor(sin, "float32"),
    )["tensor"]
    np.testing.assert_allclose(actual.data, expected, rtol=1e-6, atol=1e-6)


# 验证 bfloat16 路径按位解码输入和 cache，并按位写回旋转结果。
def test_c_backend_rotary_embedding_bfloat16_decodes_and_writes_bit_storage():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x_values = np.linspace(-1.5, 1.5, 2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4)
    angles = np.linspace(0.0, 1.0, 6 * 2, dtype=np.float32).reshape(6, 2)
    cos_values = np.cos(angles).astype(np.float32)
    sin_values = np.sin(angles).astype(np.float32)
    pos = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    decoded_x = _bf16_to_float32(_bf16_bits(x_values))
    decoded_cos = _bf16_to_float32(_bf16_bits(cos_values))
    decoded_sin = _bf16_to_float32(_bf16_bits(sin_values))
    expected = _bf16_bits(_rotary_formula(decoded_x, decoded_cos, decoded_sin, pos))

    actual = RotaryEmbedding(["x", "cos", "sin", "pos"], ["y"], dtype="bfloat16").forward(
        _tensor(_bf16_bits(x_values), "bfloat16"),
        _tensor(_bf16_bits(cos_values), "bfloat16"),
        _tensor(_bf16_bits(sin_values), "bfloat16"),
        _tensor(pos, "int64"),
    )["tensor"]
    np.testing.assert_array_equal(actual.data, expected)


# 验证 Python fallback 也实现同样的 position_ids 和 interleaved 语义。
def test_python_rotary_embedding_fallback_matches_formula(monkeypatch):
    _disable_c_backend(monkeypatch)

    x = np.linspace(-1.0, 1.0, 2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8)
    angles = np.linspace(0.0, 0.5, 2 * 3 * 1, dtype=np.float32).reshape(2, 3, 1)
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    attrs = {"num_heads": 2, "rotary_embedding_dim": 2, "interleaved": 1}
    expected = _rotary_formula(x, cos, sin, **attrs)
    actual = RotaryEmbedding(["x", "cos", "sin"], ["y"], dtype="float32", **attrs).forward(
        _tensor(x, "float32"),
        _tensor(cos, "float32"),
        _tensor(sin, "float32"),
    )["tensor"]
    np.testing.assert_allclose(actual.data, expected, rtol=1e-6, atol=1e-6)


# 验证 ONNX 导入时保留 num_heads、rotary_embedding_dim 和 interleaved 属性。
def test_onnx_import_rotary_embedding_preserves_attributes(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("RotaryEmbedding", ["x", "cos", "sin"], ["y"], num_heads=2, rotary_embedding_dim=2, interleaved=1)],
        "rotary_embedding_import",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 8]),
            helper.make_tensor_value_info("cos", TensorProto.FLOAT, [2, 3, 1]),
            helper.make_tensor_value_info("sin", TensorProto.FLOAT, [2, 3, 1]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 8])],
    )
    model_path = tmp_path / "rotary_embedding.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)]), model_path)

    imported = [op for op in ONNXImport(str(model_path), strict=True) if isinstance(op, RotaryEmbedding)]
    assert len(imported) == 1
    assert imported[0].num_heads == 2
    assert imported[0].rotary_embedding_dim == 2
    assert imported[0].interleaved == 1
    assert imported[0].version == "23"
