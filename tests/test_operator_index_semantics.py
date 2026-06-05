# /**
#   ******************************************************************************
#   * @file        test_operator_index_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用 ONNX reference 验证索引、Arg、TopK、Scatter 和扫描类算子语义。
#   * @details     2026.06.05  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from onnx.reference import ReferenceEvaluator

from operator_test_context import *  # noqa: F401,F403
from nn.Operators import (
    ArgMax,
    ArgMin,
    CumProd,
    CumSum,
    Gather,
    GatherElements,
    GatherND,
    NonZero,
    ScatterND,
    TensorScatter,
    TopK,
)


# 构造 Tensor，避免每个用例重复 shape、dtype 和 data 样板。
def _tensor(data, dtype):
    data = np.asarray(data)
    return Tensor(*data.shape, dtype=dtype, data=data)


# 调用 ONNX reference evaluator，返回单节点模型的官方输出列表。
def _onnx_reference(op_name, inputs, input_protos, attrs, output_shapes, output_protos, opset=17):
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return ReferenceEvaluator(model).run(None, dict(zip(input_names, inputs)))


# 按 dtype 选择精确比较或容差比较。
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


# 验证 Gather 的负轴、负索引和输出 shape 与 ONNX reference 一致。
def test_c_backend_gather_negative_axis_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    indices = np.array([[0, -1], [2, 1]], dtype=np.int64)
    expected = _onnx_reference(
        "Gather",
        [data, indices],
        [TensorProto.FLOAT16, TensorProto.INT64],
        {"axis": -2},
        [(2, 2, 2, 4)],
        [TensorProto.FLOAT16],
    )[0]
    actual = Gather(["data", "indices"], ["y"], axis=-2, dtype="float16").forward(
        _tensor(data, "float16"),
        _tensor(indices, "int64"),
    )["tensor"]
    _assert_tensor_matches(actual, expected)


# 验证 GatherElements 的负轴和逐元素索引语义。
def test_c_backend_gather_elements_negative_axis_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16)
    indices = np.array([[2, 1, 0], [0, 2, 1]], dtype=np.int64)
    expected = _onnx_reference(
        "GatherElements",
        [data, indices],
        [TensorProto.FLOAT16, TensorProto.INT64],
        {"axis": -1},
        [indices.shape],
        [TensorProto.FLOAT16],
    )[0]
    actual = GatherElements(["data", "indices"], ["y"], axis=-1, dtype="float16").forward(
        _tensor(data, "float16"),
        _tensor(indices, "int64"),
    )["tensor"]
    _assert_tensor_matches(actual, expected)


# 验证 GatherND 的 batch_dims=1 官方语义。
def test_c_backend_gathernd_batch_dims_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    indices = np.array([[[0], [2]], [[1], [0]]], dtype=np.int64)
    expected = _onnx_reference(
        "GatherND",
        [data, indices],
        [TensorProto.FLOAT, TensorProto.INT64],
        {"batch_dims": 1},
        [(2, 2, 4)],
        [TensorProto.FLOAT],
    )[0]
    actual = GatherND(["data", "indices"], ["y"], batch_dims=1, dtype="float32").forward(
        _tensor(data, "float32"),
        _tensor(indices, "int64"),
    )["tensor"]
    _assert_tensor_matches(actual, expected, rtol=1e-6, atol=1e-6)


# 验证 ScatterND none/add/mul 三种 reduction 与 ONNX reference 一致。
@pytest.mark.parametrize("reduction", ["none", "add", "mul"])
def test_c_backend_scatternd_reductions_match_onnx_reference(reduction):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    if reduction == "none":
        indices = np.array([[1], [3]], dtype=np.int64)
        updates = np.array([5.0, 6.0], dtype=np.float16)
    else:
        indices = np.array([[1], [3], [1]], dtype=np.int64)
        updates = np.array([5.0, 6.0, 2.0], dtype=np.float16)
    expected = _onnx_reference(
        "ScatterND",
        [data, indices, updates],
        [TensorProto.FLOAT16, TensorProto.INT64, TensorProto.FLOAT16],
        {"reduction": reduction},
        [data.shape],
        [TensorProto.FLOAT16],
        opset=16,
    )[0]
    actual = ScatterND(["data", "indices", "updates"], ["y"], reduction=reduction, dtype="float16").forward(
        _tensor(data, "float16"),
        _tensor(indices, "int64"),
        _tensor(updates, "float16"),
    )["tensor"]
    _assert_tensor_matches(actual, expected)


# 验证 TensorScatter linear 模式按 batch 级写入起点更新 sequence 轴。
def test_c_backend_tensor_scatter_linear_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    past_cache = np.array(
        [
            [[[1, 2, 3, 4, 5], [5, 6, 7, 8, 9], [8, 7, 6, 5, 4], [4, 3, 2, 1, 0]]],
            [[[1, 2, 3, 4, 5], [5, 6, 7, 8, 9], [8, 7, 6, 5, 4], [4, 3, 2, 1, 0]]],
        ],
        dtype=np.float32,
    )
    update = np.array([[[[5, 5, 5, 5, 5]]], [[[1, 1, 1, 1, 1]]]], dtype=np.float32)
    write_indices = np.array([1, 2], dtype=np.int64)
    expected = _onnx_reference(
        "TensorScatter",
        [past_cache, update, write_indices],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64],
        {"mode": "linear"},
        [past_cache.shape],
        [TensorProto.FLOAT],
        opset=24,
    )[0]
    actual = TensorScatter(["past", "update", "indices"], ["present"], mode="linear", dtype="float32").forward(
        _tensor(past_cache, "float32"),
        _tensor(update, "float32"),
        _tensor(write_indices, "int64"),
    )["tensor"]
    _assert_tensor_matches(actual, expected, rtol=1e-6, atol=1e-6)


# 验证 TensorScatter circular 模式在 sequence 轴上取模写入。
def test_c_backend_tensor_scatter_circular_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    past_cache = np.arange(2 * 1 * 4 * 5, dtype=np.float32).reshape(2, 1, 4, 5)
    update = np.array(
        [
            [[[50, 51, 52, 53, 54], [60, 61, 62, 63, 64]]],
            [[[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]]],
        ],
        dtype=np.float32,
    )
    write_indices = np.array([3, 2], dtype=np.int64)
    expected = _onnx_reference(
        "TensorScatter",
        [past_cache, update, write_indices],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64],
        {"mode": "circular"},
        [past_cache.shape],
        [TensorProto.FLOAT],
        opset=24,
    )[0]
    actual = TensorScatter(["past", "update", "indices"], ["present"], mode="circular", dtype="float32").forward(
        _tensor(past_cache, "float32"),
        _tensor(update, "float32"),
        _tensor(write_indices, "int64"),
    )["tensor"]
    _assert_tensor_matches(actual, expected, rtol=1e-6, atol=1e-6)


# 验证 TensorScatter 省略 write_indices 时默认从每个 batch 的 0 位置写入。
def test_c_backend_tensor_scatter_default_write_indices_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    past_cache = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    update = np.full((3, 2, 5), 7.0, dtype=np.float32)
    expected = _onnx_reference(
        "TensorScatter",
        [past_cache, update],
        [TensorProto.FLOAT, TensorProto.FLOAT],
        {},
        [past_cache.shape],
        [TensorProto.FLOAT],
        opset=24,
    )[0]
    actual = TensorScatter(["past", "update"], ["present"], dtype="float32").forward(
        _tensor(past_cache, "float32"),
        _tensor(update, "float32"),
    )["tensor"]
    _assert_tensor_matches(actual, expected, rtol=1e-6, atol=1e-6)


# 验证 TensorScatter 的低精度路径只搬运目标元素位模式，不做额外数值转换。
def test_c_backend_tensor_scatter_low_precision_preserves_bits():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    write_indices = _tensor(np.array([1, 2], dtype=np.int64), "int64")

    bf16_cache = np.arange(1, 41, dtype=np.uint16).reshape(2, 1, 4, 5)
    bf16_update = np.arange(101, 121, dtype=np.uint16).reshape(2, 1, 2, 5)
    bf16_expected = bf16_cache.copy()
    bf16_expected[0, :, 1:3, :] = bf16_update[0]
    bf16_expected[1, :, 2:4, :] = bf16_update[1]
    bf16_actual = TensorScatter(["past", "update", "indices"], ["present"], dtype="bfloat16").forward(
        _tensor(bf16_cache, "bfloat16"),
        _tensor(bf16_update, "bfloat16"),
        write_indices,
    )["tensor"]
    np.testing.assert_array_equal(bf16_actual.data, bf16_expected)

    fp8_cache = np.arange(1, 41, dtype=np.uint8).reshape(2, 1, 4, 5)
    fp8_update = np.arange(151, 171, dtype=np.uint8).reshape(2, 1, 2, 5)
    fp8_expected = fp8_cache.copy()
    fp8_expected[0, :, 1:3, :] = fp8_update[0]
    fp8_expected[1, :, 2:4, :] = fp8_update[1]
    fp8_actual = TensorScatter(["past", "update", "indices"], ["present"], dtype="float8_e4m3").forward(
        _tensor(fp8_cache, "float8_e4m3"),
        _tensor(fp8_update, "float8_e4m3"),
        write_indices,
    )["tensor"]
    np.testing.assert_array_equal(fp8_actual.data, fp8_expected)


# 验证 TensorScatter 导入时保留 axis 和 mode 属性。
def test_onnx_import_tensor_scatter_preserves_attributes(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("TensorScatter", ["past", "update", "indices"], ["present"], axis=-2, mode="circular")],
        "tensor_scatter_import",
        [
            helper.make_tensor_value_info("past", TensorProto.FLOAT, [2, 1, 4, 5]),
            helper.make_tensor_value_info("update", TensorProto.FLOAT, [2, 1, 2, 5]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, [2]),
        ],
        [helper.make_tensor_value_info("present", TensorProto.FLOAT, [2, 1, 4, 5])],
    )
    model_path = tmp_path / "tensor_scatter.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)]), model_path)

    imported = [op for op in ONNXImport(str(model_path), strict=True) if isinstance(op, TensorScatter)]
    assert len(imported) == 1
    assert imported[0].axis == -2
    assert imported[0].mode == "circular"
    assert imported[0].version == "24"


# 验证 NonZero 按行主序返回非零坐标。
def test_c_backend_nonzero_matches_onnx_reference_order():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], dtype=np.float32)
    expected = _onnx_reference(
        "NonZero",
        [data],
        [TensorProto.FLOAT],
        {},
        [(2, 3)],
        [TensorProto.INT64],
    )[0]
    actual = NonZero(["x"], ["y"]).forward(_tensor(data, "float32"))["tensor"]
    _assert_tensor_matches(actual, expected)


ARG_REFERENCE_CASES = [
    (ArgMax, "ArgMax", {"axis": 1, "keepdims": 0, "select_last_index": 0}, (2,)),
    (ArgMax, "ArgMax", {"axis": 1, "keepdims": 0, "select_last_index": 1}, (2,)),
    (ArgMin, "ArgMin", {"axis": -1, "keepdims": 1, "select_last_index": 1}, (2, 1)),
]


# 验证 ArgMax/ArgMin 的 tie-breaking、负轴和 keepdims 语义。
@pytest.mark.parametrize("op_cls,op_name,attrs,output_shape", ARG_REFERENCE_CASES)
def test_c_backend_arg_ops_match_onnx_reference(op_cls, op_name, attrs, output_shape):
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.array([[1.0, 3.0, 3.0], [2.0, -1.0, -1.0]], dtype=np.float32)
    expected = _onnx_reference(
        op_name,
        [data],
        [TensorProto.FLOAT],
        attrs,
        [output_shape],
        [TensorProto.INT64],
    )[0]
    actual = op_cls(["x"], ["y"], dtype="int64", **attrs).forward(_tensor(data, "float32"))["tensor"]
    _assert_tensor_matches(actual, expected)


# 验证 TopK 的 smallest/sorted 输出和值索引双输出语义。
def test_c_backend_topk_smallest_sorted_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.array([[1.0, 4.0, 2.0, 3.0], [5.0, -1.0, 0.0, 2.0]], dtype=np.float16)
    k = np.array([2], dtype=np.int64)
    expected_values, expected_indices = _onnx_reference(
        "TopK",
        [data, k],
        [TensorProto.FLOAT16, TensorProto.INT64],
        {"axis": -1, "largest": 0, "sorted": 1},
        [(2, 2), (2, 2)],
        [TensorProto.FLOAT16, TensorProto.INT64],
    )
    values, indices = TopK(
        ["x", "k"],
        ["values", "indices"],
        axis=-1,
        largest=0,
        sorted=1,
        dtype="float16",
    ).forward(_tensor(data, "float16"), _tensor(k, "int64"))["tensor"]
    _assert_tensor_matches(values, expected_values)
    _assert_tensor_matches(indices, expected_indices)


# 验证 CumSum 的 exclusive + reverse + 负轴语义。
def test_c_backend_cumsum_exclusive_reverse_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    axis = np.array(-1, dtype=np.int32)
    expected = _onnx_reference(
        "CumSum",
        [data, axis],
        [TensorProto.FLOAT, TensorProto.INT32],
        {"exclusive": 1, "reverse": 1},
        [data.shape],
        [TensorProto.FLOAT],
    )[0]
    actual = CumSum(["x", "axis"], ["y"], exclusive=1, reverse=1, dtype="float32").forward(
        _tensor(data, "float32"),
        _tensor(axis, "int32"),
    )["tensor"]
    _assert_tensor_matches(actual, expected, rtol=1e-6, atol=1e-6)


# 验证 CumProd 的 exclusive + reverse + 负轴语义。
def test_c_backend_cumprod_exclusive_reverse_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.linspace(0.75, 1.3, 12, dtype=np.float32).reshape(3, 4)
    axis = np.array(-1, dtype=np.int64)
    expected = _onnx_reference(
        "CumProd",
        [data, axis],
        [TensorProto.FLOAT, TensorProto.INT64],
        {"exclusive": 1, "reverse": 1},
        [data.shape],
        [TensorProto.FLOAT],
        opset=26,
    )[0]
    actual = CumProd(["x", "axis"], ["y"], exclusive=1, reverse=1, dtype="float32").forward(
        _tensor(data, "float32"),
        _tensor(axis, "int64"),
    )["tensor"]
    _assert_tensor_matches(actual, expected, rtol=1e-6, atol=1e-6)


# 验证 CumProd 导入时保留 exclusive 和 reverse 属性。
def test_onnx_import_cumprod_preserves_scan_attributes(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("CumProd", ["x", "axis"], ["y"], exclusive=1, reverse=1)],
        "cumprod_import",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info("axis", TensorProto.INT64, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4])],
    )
    model_path = tmp_path / "cumprod.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 26)]), model_path)

    imported = [op for op in ONNXImport(str(model_path), strict=True) if isinstance(op, CumProd)]
    assert len(imported) == 1
    assert imported[0].exclusive == 1
    assert imported[0].reverse == 1
    assert imported[0].version == "26"


# 验证 bfloat16 的索引类算子按位读取输入并按位写回输出。
def test_c_backend_bfloat16_index_ops_decode_and_write_bit_storage():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    values = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]], dtype=np.float32)
    data = _tensor(_bf16_bits(values), "bfloat16")
    gather_indices = _tensor(np.array([2, 0], dtype=np.int64), "int64")
    gathered = Gather(["data", "indices"], ["y"], axis=1, dtype="bfloat16").forward(data, gather_indices)["tensor"]
    np.testing.assert_allclose(
        _bf16_to_float32(gathered.data),
        values[:, [2, 0]],
        rtol=1e-2,
        atol=1e-2,
    )

    scatter_data = _tensor(_bf16_bits(np.zeros(4, dtype=np.float32)), "bfloat16")
    scatter_indices = _tensor(np.array([[1], [3]], dtype=np.int64), "int64")
    scatter_updates = _tensor(_bf16_bits(np.array([2.0, -1.5], dtype=np.float32)), "bfloat16")
    scattered = ScatterND(
        ["data", "indices", "updates"],
        ["y"],
        dtype="bfloat16",
    ).forward(scatter_data, scatter_indices, scatter_updates)["tensor"]
    np.testing.assert_allclose(
        _bf16_to_float32(scattered.data),
        np.array([0.0, 2.0, 0.0, -1.5], dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )

    cumprod_values = np.array([1.0, 1.5, 0.5, 2.0], dtype=np.float32)
    cumprod_axis = _tensor(np.array(0, dtype=np.int64), "int64")
    cumprod = CumProd(["x", "axis"], ["y"], exclusive=1, reverse=1, dtype="bfloat16").forward(
        _tensor(_bf16_bits(cumprod_values), "bfloat16"),
        cumprod_axis,
    )["tensor"]
    np.testing.assert_allclose(
        _bf16_to_float32(cumprod.data),
        np.array([1.5, 1.0, 2.0, 1.0], dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
