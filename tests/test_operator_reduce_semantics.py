# /**
#   ******************************************************************************
#   * @file        test_operator_reduce_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用 ONNX reference 验证基础 Reduce 算子的官方语义和混合精度路径。
#   * @details     2026.06.05  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from onnx.reference import ReferenceEvaluator

from operator_test_context import *  # noqa: F401,F403
from nn.Operators import ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum


# 构造 Tensor，避免每个断言重复 shape、dtype 和 data 样板。
def _tensor(data, dtype):
    data = np.asarray(data)
    return Tensor(*data.shape, dtype=dtype, data=data)


# 调用 ONNX reference evaluator，验证 Reduce 单节点模型的官方输出。
def _onnx_reference(op_name, inputs, input_protos, attrs, output_shape, output_proto):
    input_names = [f"i{i}" for i in range(len(inputs))]
    graph = helper.make_graph(
        [helper.make_node(op_name, input_names, ["y"], **attrs)],
        f"{op_name}_reference",
        [
            helper.make_tensor_value_info(name, proto, list(np.asarray(value).shape))
            for name, proto, value in zip(input_names, input_protos, inputs)
        ],
        [helper.make_tensor_value_info("y", output_proto, list(output_shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return ReferenceEvaluator(model).run(None, dict(zip(input_names, inputs)))[0]


# 对浮点输出使用容差比较，对整数/布尔输出使用精确比较。
def _assert_tensor_matches(actual, expected, rtol=2e-3, atol=2e-3):
    if np.issubdtype(np.asarray(expected).dtype, np.floating):
        np.testing.assert_allclose(actual.data, expected, rtol=rtol, atol=atol)
    else:
        np.testing.assert_array_equal(actual.data, expected)


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


REDUCE_ATTR_CASES = [
    (ReduceMean, "ReduceMean", [-1], 0, (2, 3)),
    (ReduceMax, "ReduceMax", [0, -1], 1, (1, 3, 1)),
    (ReduceMin, "ReduceMin", [0, -1], 0, (3,)),
    (ReduceProd, "ReduceProd", [1], 1, (2, 1, 4)),
]


# 验证 ReduceMean/Max/Min/Prod 在 opset 17 中使用属性 axes 的官方语义。
@pytest.mark.parametrize("op_cls,op_name,axes,keepdims,output_shape", REDUCE_ATTR_CASES)
def test_c_backend_reduce_attr_axes_float16_match_onnx_reference(
    op_cls,
    op_name,
    axes,
    keepdims,
    output_shape,
):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 8.0 - 1.0).astype(np.float16)
    expected = _onnx_reference(
        op_name,
        [x],
        [TensorProto.FLOAT16],
        {"axes": axes, "keepdims": keepdims},
        output_shape,
        TensorProto.FLOAT16,
    )
    actual = op_cls(["x"], ["y"], axes=axes, keepdims=keepdims, dtype="float16").forward(
        _tensor(x, "float16"),
    )["tensor"]
    _assert_tensor_matches(actual, expected)


# 验证 ReduceSum 在 opset 17 中使用第二个输入 axes 的官方语义。
def test_c_backend_reduce_sum_runtime_axes_float16_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 8.0 - 1.0).astype(np.float16)
    axes = np.array([1, -1], dtype=np.int64)
    expected = _onnx_reference(
        "ReduceSum",
        [x, axes],
        [TensorProto.FLOAT16, TensorProto.INT64],
        {"keepdims": 0},
        (2,),
        TensorProto.FLOAT16,
    )
    actual = ReduceSum(["x", "axes"], ["y"], keepdims=0, dtype="float16").forward(
        _tensor(x, "float16"),
        _tensor(axes, "int64"),
    )["tensor"]
    _assert_tensor_matches(actual, expected)


# 验证 ReduceSum 空 axes 在 noop_with_empty_axes=1 时保持输入不变。
def test_c_backend_reduce_sum_empty_axes_noop_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.arange(6, dtype=np.float32).reshape(2, 3).astype(np.float16)
    axes = np.array([], dtype=np.int64)
    expected = _onnx_reference(
        "ReduceSum",
        [x, axes],
        [TensorProto.FLOAT16, TensorProto.INT64],
        {"noop_with_empty_axes": 1},
        x.shape,
        TensorProto.FLOAT16,
    )
    actual = ReduceSum(
        ["x", "axes"],
        ["y"],
        noop_with_empty_axes=1,
        dtype="float16",
    ).forward(_tensor(x, "float16"), _tensor(axes, "int64"))["tensor"]
    _assert_tensor_matches(actual, expected)


# 验证 ReduceSum 空 axes 在默认 noop=0 时归约所有维度，并保留默认 keepdims=1。
def test_c_backend_reduce_sum_empty_axes_default_reduces_all_axes():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.arange(6, dtype=np.float32).reshape(2, 3).astype(np.float16)
    axes = np.array([], dtype=np.int64)
    expected = _onnx_reference(
        "ReduceSum",
        [x, axes],
        [TensorProto.FLOAT16, TensorProto.INT64],
        {},
        (1, 1),
        TensorProto.FLOAT16,
    )
    actual = ReduceSum(["x", "axes"], ["y"], dtype="float16").forward(
        _tensor(x, "float16"),
        _tensor(axes, "int64"),
    )["tensor"]
    _assert_tensor_matches(actual, expected)


# 验证 Reduce shape 推断与 runtime 空 axes 语义一致。
def test_reduce_sum_empty_axes_shape_inference_matches_runtime_semantics():
    x = Tensor_(2, 3, dtype="float16")
    axes = Tensor(0, dtype="int64", data=np.array([], dtype=np.int64))

    no_op = ReduceSum(["x", "axes"], ["y"], noop_with_empty_axes=1, dtype="float16").forward_(
        x,
        axes,
    )["tensor"]
    assert no_op.size == (2, 3)

    reduced = ReduceSum(["x", "axes"], ["y"], dtype="float16").forward_(x, axes)["tensor"]
    assert reduced.size == (1, 1)


# 验证 bfloat16 输入按位解码，并以 bfloat16 输出写回 Reduce 结果。
def test_c_backend_reduce_bfloat16_decode_and_write_bit_storage():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    values = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 7.0) - 1.0
    tensor = _tensor(_bf16_bits(values), "bfloat16")
    axes = _tensor(np.array([1], dtype=np.int64), "int64")

    sum_out = ReduceSum(["x", "axes"], ["y"], dtype="bfloat16").forward(tensor, axes)["tensor"]
    np.testing.assert_allclose(
        _bf16_to_float32(sum_out.data),
        np.sum(values, axis=1, keepdims=True),
        rtol=2e-2,
        atol=2e-2,
    )

    mean_out = ReduceMean(["x"], ["y"], axes=[1], dtype="bfloat16").forward(tensor)["tensor"]
    np.testing.assert_allclose(
        _bf16_to_float32(mean_out.data),
        np.mean(values, axis=1, keepdims=True),
        rtol=2e-2,
        atol=2e-2,
    )

    max_out = ReduceMax(["x"], ["y"], axes=[1], dtype="bfloat16").forward(tensor)["tensor"]
    np.testing.assert_allclose(
        _bf16_to_float32(max_out.data),
        np.max(values, axis=1, keepdims=True),
        rtol=2e-2,
        atol=2e-2,
    )

    min_out = ReduceMin(["x"], ["y"], axes=[1], dtype="bfloat16").forward(tensor)["tensor"]
    np.testing.assert_allclose(
        _bf16_to_float32(min_out.data),
        np.min(values, axis=1, keepdims=True),
        rtol=2e-2,
        atol=2e-2,
    )

    prod_values = (np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4) / 10.0)
    prod_tensor = _tensor(_bf16_bits(prod_values), "bfloat16")
    prod_out = ReduceProd(["x"], ["y"], axes=[1], dtype="bfloat16").forward(prod_tensor)["tensor"]
    np.testing.assert_allclose(
        _bf16_to_float32(prod_out.data),
        np.prod(prod_values, axis=1, keepdims=True),
        rtol=3e-2,
        atol=3e-2,
    )
