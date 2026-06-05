# /**
#   ******************************************************************************
#   * @file        test_operator_attention_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用 ONNX reference 验证 Attention 算子的官方语义和混合精度路径。
#   * @details     2026.06.05  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from onnx.reference.ops.op_attention import _compute_attention

from conftest import _disable_c_backend
from operator_test_context import *  # noqa: F401,F403
from nn.Operators import Attention


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


# 验证 4D GQA、float mask、causal 和 softcap 组合与 ONNX 官方 reference 一致。
def test_c_backend_attention_gqa_float_mask_causal_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    q = np.linspace(-1.0, 1.0, 1 * 4 * 3 * 4, dtype=np.float32).reshape(1, 4, 3, 4)
    k = np.linspace(0.8, -0.9, 1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
    v = np.linspace(-0.6, 0.7, 1 * 2 * 4 * 3, dtype=np.float32).reshape(1, 2, 4, 3)
    mask = np.zeros((1, 1, 3, 4), dtype=np.float32)
    mask[0, 0, 2, 1] = -1e4
    expected = _compute_attention(q, k, v, attn_mask=mask, is_causal=True, softcap=3.0)[0]
    actual = Attention(["q", "k", "v", "mask"], ["y"], dtype="float32", is_causal=1, softcap=3.0).forward(
        _tensor(q, "float32"),
        _tensor(k, "float32"),
        _tensor(v, "float32"),
        _tensor(mask, "float32"),
    )["tensor"]
    np.testing.assert_allclose(actual.data, expected, rtol=2e-5, atol=2e-5)


# 验证 bfloat16 路径会解码低精度输入，并以 bfloat16 位存储写回输出。
def test_c_backend_attention_bfloat16_decodes_and_writes_bit_storage():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    q_values = np.linspace(-1.0, 1.0, 1 * 4 * 3 * 4, dtype=np.float32).reshape(1, 4, 3, 4)
    k_values = np.linspace(0.8, -0.9, 1 * 2 * 5 * 4, dtype=np.float32).reshape(1, 2, 5, 4)
    v_values = np.linspace(-0.6, 0.7, 1 * 2 * 5 * 3, dtype=np.float32).reshape(1, 2, 5, 3)
    q_bits = _bf16_bits(q_values)
    k_bits = _bf16_bits(k_values)
    v_bits = _bf16_bits(v_values)
    expected = _compute_attention(
        _bf16_to_float32(q_bits),
        _bf16_to_float32(k_bits),
        _bf16_to_float32(v_bits),
        is_causal=True,
        softcap=3.0,
    )[0]
    actual = Attention(["q", "k", "v"], ["y"], dtype="bfloat16", is_causal=1, softcap=3.0).forward(
        _tensor(q_bits, "bfloat16"),
        _tensor(k_bits, "bfloat16"),
        _tensor(v_bits, "bfloat16"),
    )["tensor"]
    assert actual.data.dtype == np.uint16
    np.testing.assert_allclose(_bf16_to_float32(actual.data), _bf16_to_float32(_bf16_bits(expected)), rtol=2e-2, atol=2e-2)


# 验证 Python fallback 复用官方 reference，覆盖 3D 输入、KV cache 和 qk_matmul_output。
def test_python_attention_fallback_3d_cache_and_qk_output_matches_reference(monkeypatch):
    _disable_c_backend(monkeypatch)

    q = np.linspace(-0.8, 0.9, 1 * 2 * 16, dtype=np.float32).reshape(1, 2, 16)
    k = np.linspace(0.5, -0.7, 1 * 2 * 8, dtype=np.float32).reshape(1, 2, 8)
    v = np.linspace(-0.4, 0.6, 1 * 2 * 6, dtype=np.float32).reshape(1, 2, 6)
    past_key = np.linspace(-0.3, 0.2, 1 * 2 * 2 * 4, dtype=np.float32).reshape(1, 2, 2, 4)
    past_value = np.linspace(0.25, -0.15, 1 * 2 * 2 * 3, dtype=np.float32).reshape(1, 2, 2, 3)
    expected = _compute_attention(
        q,
        k,
        v,
        past_key=past_key,
        past_value=past_value,
        q_num_heads=4,
        kv_num_heads=2,
        qk_matmul_output_mode=3,
    )
    actual = Attention(
        ["q", "k", "v", "", "past_key", "past_value"],
        ["y", "present_key", "present_value", "qk"],
        q_num_heads=4,
        kv_num_heads=2,
        qk_matmul_output_mode=3,
        dtype="float32",
    ).forward(
        _tensor(q, "float32"),
        _tensor(k, "float32"),
        _tensor(v, "float32"),
        None,
        _tensor(past_key, "float32"),
        _tensor(past_value, "float32"),
    )["tensor"]
    assert len(actual) == 4
    for got, exp in zip(actual, expected):
        np.testing.assert_allclose(got.data, exp, rtol=2e-5, atol=2e-5)


# 验证 ONNX 导入时保留 Attention 的头数、缩放和中间输出属性。
def test_onnx_import_attention_preserves_attributes(tmp_path):
    graph = helper.make_graph(
        [
            helper.make_node(
                "Attention",
                ["q", "k", "v"],
                ["y"],
                q_num_heads=4,
                kv_num_heads=2,
                scale=0.5,
                is_causal=1,
                softmax_precision=TensorProto.FLOAT,
                softcap=2.5,
                qk_matmul_output_mode=3,
            )
        ],
        "attention_import",
        [
            helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 2, 16]),
            helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 3, 8]),
            helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 3, 6]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 12])],
    )
    model_path = tmp_path / "attention.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)]), model_path)

    imported = [op for op in ONNXImport(str(model_path), strict=True) if isinstance(op, Attention)]
    assert len(imported) == 1
    assert imported[0].q_num_heads == 4
    assert imported[0].kv_num_heads == 2
    assert imported[0].scale == 0.5
    assert imported[0].is_causal == 1
    assert imported[0].softmax_precision == TensorProto.FLOAT
    assert imported[0].softcap == 2.5
    assert imported[0].qk_matmul_output_mode == 3
    assert imported[0].version == "24"
