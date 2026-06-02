"""文件功能：覆盖池化、Arg、归一化、压缩、拆分、类型转换和 ONNX17 官方补充算子。
作者：Egor Izmaylov
时间：2026-06-02
"""

from conftest import _disable_c_backend
from operator_test_context import *  # noqa: F401,F403


def test_pooling_supports_nd_shapes_and_optional_indices(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    x1 = Tensor(1, 1, 5, dtype="float32", data=np.arange(5, dtype=np.float32).reshape(1, 1, 5))
    max_pool = MaxPool(["x"], ["y", "idx"], kernel_shape=[2], pads=[0, 1], strides=[2], dilations=[1], dtype="float32")
    y, idx = max_pool.forward(x1)["tensor"]
    np.testing.assert_array_equal(y.data, np.array([[[1.0, 3.0, 4.0]]], dtype=np.float32))
    np.testing.assert_array_equal(idx.data, np.array([[[1, 3, 4]]], dtype=np.int64))
    inferred_y, inferred_idx = max_pool.forward_(Tensor_(1, 1, 5, dtype="float32"))["tensor"]
    assert inferred_y.size == (1, 1, 3)
    assert inferred_idx.dtype == "int64"

    data3 = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 2, 2)
    x3 = Tensor(*data3.shape, dtype="float32", data=data3)
    avg = AveragePool(
        ["x"], ["y"], kernel_shape=[2, 1, 2], pads=[0, 0, 0, 0, 0, 0], strides=[1, 1, 1],
        dilations=[1, 1, 1], count_include_pad=0, dtype="float32"
    ).forward(x3)["tensor"]
    expected_avg = np.array([[[[[2.5], [4.5]]]]], dtype=np.float32)
    np.testing.assert_array_equal(avg.data, expected_avg)

    lp = LpPool(["x"], ["y"], kernel_shape=[3], pads=[1, 1], strides=[1], dilations=[1], p=2, dtype="float32")
    lp_out = lp.forward(x1)["tensor"]
    expected_lp = np.sqrt(np.array([1, 5, 14, 29, 25], dtype=np.float32)).reshape(1, 1, 5)
    np.testing.assert_allclose(lp_out.data, expected_lp)

    auto_max = MaxPool(
        ["x"], ["y"], kernel_shape=[3], pads=[0, 0], strides=[2], dilations=[1], auto_pad="SAME_UPPER", dtype="float32"
    )
    np.testing.assert_array_equal(auto_max.forward(x1)["tensor"].data, np.array([[[1.0, 3.0, 4.0]]], dtype=np.float32))
    assert auto_max.forward_(Tensor_(1, 1, 5, dtype="float32"))["tensor"].size == (1, 1, 3)

    auto_avg = AveragePool(
        ["x"], ["y"], kernel_shape=[3], pads=[0, 0], strides=[2], dilations=[1], count_include_pad=0,
        auto_pad="SAME_UPPER", dtype="float32"
    ).forward(x1)["tensor"]
    np.testing.assert_allclose(auto_avg.data, np.array([[[0.5, 2.0, 3.5]]], dtype=np.float32))

    auto_lp = LpPool(
        ["x"], ["y"], kernel_shape=[3], pads=[0, 0], strides=[2], dilations=[1], p=2,
        auto_pad="SAME_UPPER", dtype="float32"
    ).forward(x1)["tensor"]
    np.testing.assert_allclose(auto_lp.data, np.sqrt(np.array([1.0, 14.0, 25.0], dtype=np.float32)).reshape(1, 1, 3))

    model_path = tmp_path / "pool_attrs.onnx"
    graph = helper.make_graph(
        [helper.make_node("MaxPool", ["x"], ["y", "idx"], kernel_shape=[2], pads=[0, 1], strides=[2], ceil_mode=1, storage_order=1, auto_pad="SAME_UPPER")],
        "pool_attrs",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 5])],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3]),
            helper.make_tensor_value_info("idx", TensorProto.INT64, [1, 1, 3]),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert isinstance(ops[0], MaxPool)
    assert ops[0].ceil_mode == 1
    assert ops[0].storage_order == 1
    assert ops[0].auto_pad == "SAME_UPPER"

def test_arg_ops_keepdims_and_select_last_match_onnx_reference(monkeypatch):
    _disable_c_backend(monkeypatch)
    from onnx.reference import ReferenceEvaluator

    data = np.array(
        [
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
            [[0.0, -1.0, 1.0], [2.0, 2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    tensor = Tensor(*data.shape, dtype="float32", data=data)

    for op_type, op_cls in (("ArgMax", ArgMax), ("ArgMin", ArgMin)):
        for axis in (0, 1, -1):
            graph = helper.make_graph(
                [
                    helper.make_node(
                        op_type,
                        ["x"],
                        ["y"],
                        axis=axis,
                        keepdims=1,
                        select_last_index=1,
                    )
                ],
                f"{op_type.lower()}_keepdims",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, data.shape)],
                [helper.make_tensor_value_info("y", TensorProto.INT64, None)],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            expected = ReferenceEvaluator(model).run(None, {"x": data})[0]

            actual = op_cls(
                ["x"],
                ["y"],
                axis=axis,
                keepdims=1,
                select_last_index=1,
                dtype="int64",
            ).forward(tensor)["tensor"].data

            np.testing.assert_array_equal(actual, expected)

def test_space_to_depth_and_lp_normalization_match_onnx_reference(monkeypatch):
    _disable_c_backend(monkeypatch)
    from onnx.reference import ReferenceEvaluator

    space_data = np.arange(1 * 2 * 4 * 6, dtype=np.float32).reshape(1, 2, 4, 6)
    space_graph = helper.make_graph(
        [helper.make_node("SpaceToDepth", ["x"], ["y"], blocksize=2)],
        "space_to_depth_ref",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, space_data.shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    space_model = helper.make_model(space_graph, opset_imports=[helper.make_opsetid("", 17)])
    expected_space = ReferenceEvaluator(space_model).run(None, {"x": space_data})[0]
    actual_space = SpaceToDepth(["x"], ["y"], blocksize=2, dtype="float32").forward(
        Tensor(*space_data.shape, dtype="float32", data=space_data)
    )["tensor"].data
    np.testing.assert_array_equal(actual_space, expected_space)

    lp_data = np.array(
        [
            [[1.0, -2.0], [3.0, -4.0]],
            [[0.0, 5.0], [-6.0, 7.0]],
        ],
        dtype=np.float32,
    )
    lp_norm = np.sum(np.abs(lp_data), axis=1, keepdims=True)
    expected_lp = np.where(lp_norm == 0, 0, lp_data / lp_norm).astype(np.float32)
    actual_lp = LpNormalization(["x"], ["y"], axis=1, p=1, dtype="float32").forward(
        Tensor(*lp_data.shape, dtype="float32", data=lp_data)
    )["tensor"].data
    np.testing.assert_array_equal(actual_lp, expected_lp)

def test_lp_normalization_l1_preserves_input_sign_in_c_and_python():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.array([[-1.0, 2.0, -3.0], [4.0, -4.0, 0.0]], dtype=np.float32)
    tensor = Tensor(*data.shape, dtype="float32", data=data)
    norm = np.sum(np.abs(data), axis=1, keepdims=True)
    expected = np.where(norm == 0, 0, data / norm).astype(np.float32)

    c_result = LpNormalization(["x"], ["y"], axis=1, p=1, dtype="float32").forward(tensor)["tensor"].data
    np.testing.assert_allclose(c_result, expected, rtol=1e-6, atol=1e-6)

    py_op = LpNormalization(["x"], ["y"], axis=1, p=1, dtype="float32")
    py_op.lib = None
    py_result = py_op.forward(tensor)["tensor"].data
    np.testing.assert_allclose(py_result, expected, rtol=1e-6, atol=1e-6)

def test_onehot_python_fallback_keeps_out_of_range_negative_indices_off():
    indices = Tensor(3, dtype="int64", data=np.array([-3, -4, 1], dtype=np.int64))
    depth = Tensor(dtype="int64", data=np.array(3, dtype=np.int64))
    values = Tensor(2, dtype="string", data=np.array(["off", "on"], dtype=np.str_))

    out = OneHot(["indices", "depth", "values"], ["y"], axis=-1, dtype="string").forward(
        indices, depth, values
    )["tensor"]

    np.testing.assert_array_equal(
        out.data,
        np.array(
            [
                ["on", "off", "off"],
                ["off", "off", "off"],
                ["off", "on", "off"],
            ],
            dtype=np.str_,
        ),
    )

def test_compress_short_condition_matches_onnx_reference(monkeypatch):
    _disable_c_backend(monkeypatch)
    from onnx.reference import ReferenceEvaluator

    data = np.arange(8, dtype=np.float32).reshape(2, 4)
    condition = np.array([False, True, True, False, True], dtype=np.bool_)
    graph = helper.make_graph(
        [helper.make_node("Compress", ["x", "condition"], ["y"])],
        "compress_short_condition",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, data.shape),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, condition.shape),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    expected = ReferenceEvaluator(model).run(None, {"x": data, "condition": condition})[0]
    actual = Compress(["x", "condition"], ["y"], dtype="float32").forward(
        Tensor(*data.shape, dtype="float32", data=data),
        Tensor(*condition.shape, dtype="bool", data=condition),
    )["tensor"].data
    np.testing.assert_array_equal(actual, expected)

def test_split_uneven_without_split_input_matches_onnx_reference(monkeypatch):
    _disable_c_backend(monkeypatch)
    from onnx.reference import ReferenceEvaluator

    data = np.arange(5, dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Split", ["x"], ["y0", "y1"])],
        "split_uneven",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, data.shape)],
        [
            helper.make_tensor_value_info("y0", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("y1", TensorProto.FLOAT, None),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    expected = ReferenceEvaluator(model).run(None, {"x": data})
    actual = Split(["x"], ["y0", "y1"], dtype="float32").forward(
        Tensor(*data.shape, dtype="float32", data=data)
    )["tensor"]

    assert [item.data.shape for item in actual] == [item.shape for item in expected]
    for actual_tensor, expected_array in zip(actual, expected):
        np.testing.assert_array_equal(actual_tensor.data, expected_array)

    inferred = Split(["x"], ["y0", "y1"], dtype="float32").forward_(
        Tensor_(5, dtype="float32")
    )["tensor"]
    assert [item.size for item in inferred] == [(3,), (2,)]

def test_cast_supports_string_conversions_like_onnx_reference(monkeypatch):
    _disable_c_backend(monkeypatch)
    from onnx.reference import ReferenceEvaluator

    string_data = np.array(["3.14", "+INF", "NaN", "-2"], dtype=np.str_)
    to_float_graph = helper.make_graph(
        [helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT)],
        "cast_string_to_float",
        [helper.make_tensor_value_info("x", TensorProto.STRING, string_data.shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    to_float_model = helper.make_model(to_float_graph, opset_imports=[helper.make_opsetid("", 17)])
    expected_float = ReferenceEvaluator(to_float_model).run(None, {"x": string_data})[0]
    actual_float = Cast(["x"], ["y"], dtype="float32").forward(
        Tensor(*string_data.shape, dtype="string", data=string_data)
    )["tensor"].data
    np.testing.assert_equal(actual_float, expected_float)

    numeric_data = np.array([1.5, -2.0, np.inf, np.nan], dtype=np.float32)
    to_string_graph = helper.make_graph(
        [helper.make_node("Cast", ["x"], ["y"], to=TensorProto.STRING)],
        "cast_float_to_string",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, numeric_data.shape)],
        [helper.make_tensor_value_info("y", TensorProto.STRING, None)],
    )
    to_string_model = helper.make_model(to_string_graph, opset_imports=[helper.make_opsetid("", 17)])
    expected_string = ReferenceEvaluator(to_string_model).run(None, {"x": numeric_data})[0]
    actual_string = Cast(["x"], ["y"], dtype="string").forward(
        Tensor(*numeric_data.shape, dtype="float32", data=numeric_data)
    )["tensor"].data
    np.testing.assert_array_equal(actual_string, expected_string)

def test_additional_onnx17_official_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    a = Tensor(2, 1, dtype="float32", data=np.array([[1.0], [-2.0]], dtype=np.float32))
    b = Tensor(1, 3, dtype="float32", data=np.array([[3.0, 4.0, 5.0]], dtype=np.float32))
    c = Tensor(2, 3, dtype="float32", data=np.ones((2, 3), dtype=np.float32))
    summed = Sum(["a", "b", "c"], ["out"], dtype="float32").forward(a, b, c)["tensor"]
    np.testing.assert_array_equal(summed.data, a.data + b.data + c.data)
    assert Sum(["a", "b"], ["out"], dtype="float32").forward_(Tensor_(2, 1, dtype="float32"), Tensor_(1, 3, dtype="float32"))["tensor"].size == (2, 3)

    slope = Tensor(1, 3, dtype="float32", data=np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
    prelu = PRelu(["x", "slope"], ["out"], dtype="float32").forward(summed, slope)["tensor"]
    np.testing.assert_array_equal(prelu.data, np.where(summed.data >= 0, summed.data, summed.data * slope.data))

    target = Tensor(1, dtype="int64", data=np.array([0], dtype=np.int64))
    cast_like = CastLike(["x", "target"], ["out"]).forward(a, target)["tensor"]
    assert cast_like.dtype == "int64"
    np.testing.assert_array_equal(cast_like.data, a.data.astype(np.int64))

    eye = EyeLike(["x"], ["out"], k=1, dtype="float32").forward(Tensor_(3, 4, dtype="float32"))["tensor"]
    np.testing.assert_array_equal(eye.data, np.eye(3, 4, k=1, dtype=np.float32))

    matrix = Tensor(3, 3, dtype="float32", data=np.arange(9, dtype=np.float32).reshape(3, 3))
    k = Tensor(1, dtype="int64", data=np.array([-1], dtype=np.int64))
    trilu = Trilu(["x", "k"], ["out"], upper=0, dtype="float32").forward(matrix, k)["tensor"]
    np.testing.assert_array_equal(trilu.data, np.tril(matrix.data, k=-1))
    assert RandomUniform([], ["out"], shape=[2, 3], dtype=TensorProto.FLOAT).forward_()["tensor"].size == (2, 3)
    assert RandomUniform([], ["out"], shape=[], dtype=TensorProto.FLOAT).forward_()["tensor"].size == ()
    assert RandomNormal([], ["out"], shape=[], dtype=TensorProto.DOUBLE).forward_()["tensor"].size == ()

    like_input = Tensor(2, 2, dtype="float64", data=np.ones((2, 2), dtype=np.float64))
    random_like = RandomUniformLike(["x"], ["y"], low=-2.0, high=-1.0, seed=7.0).forward(like_input)["tensor"]
    assert random_like.dtype == "float64"
    assert random_like.data.dtype == np.float64
    assert random_like.size == (2, 2)
    assert np.all(random_like.data >= -2.0)
    assert np.all(random_like.data < -1.0)

    explicit_random_like = RandomUniformLike(["x"], ["y"], dtype="float32").forward_(Tensor_(2, 2, dtype="float64"))["tensor"]
    assert explicit_random_like.dtype == "float32"

    bernoulli_uint32 = Bernoulli(["p"], ["y"], dtype=TensorProto.UINT32, seed=5.0).forward(
        Tensor(4, dtype="float32", data=np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32))
    )["tensor"]
    assert bernoulli_uint32.dtype == "uint32"
    np.testing.assert_array_equal(bernoulli_uint32.data, np.array([0, 1, 1, 0], dtype=np.uint32))

    unique_input = Tensor(6, dtype="int64", data=np.array([3, 1, 3, 2, 1, 3], dtype=np.int64))
    y, indices, inverse, counts = Unique(["x"], ["y", "indices", "inverse", "counts"], sorted=0, dtype="int64").forward(unique_input)["tensor"]
    np.testing.assert_array_equal(y.data, np.array([3, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(indices.data, np.array([0, 1, 3], dtype=np.int64))
    np.testing.assert_array_equal(inverse.data, np.array([0, 1, 0, 2, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(counts.data, np.array([3, 2, 1], dtype=np.int64))

    model_path = tmp_path / "onnx17_added_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("RandomUniform", [], ["random"], shape=[2, 3]),
            helper.make_node("RandomUniformLike", ["like_input"], ["random_like"]),
            helper.make_node("RandomUniformLike", ["like_input"], ["random_like_float"], dtype=TensorProto.FLOAT),
            helper.make_node("Sum", ["a", "b"], ["sum"]),
            helper.make_node("PRelu", ["sum", "slope"], ["prelu"]),
            helper.make_node("CastLike", ["prelu", "target"], ["casted"]),
            helper.make_node("EyeLike", ["target"], ["eye"], k=0, dtype=TensorProto.FLOAT),
            helper.make_node("Trilu", ["prelu", "k"], ["trilu"], upper=1),
            helper.make_node("Unique", ["labels"], ["unique_y", "unique_idx", "unique_inv", "unique_counts"], sorted=0),
        ],
        "onnx17_added_ops",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("slope", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("target", TensorProto.INT64, [3, 3]),
            helper.make_tensor_value_info("k", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("labels", TensorProto.INT64, [6]),
            helper.make_tensor_value_info("like_input", TensorProto.DOUBLE, [2, 2]),
        ],
        [
            helper.make_tensor_value_info("random", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("random_like", TensorProto.DOUBLE, [2, 2]),
            helper.make_tensor_value_info("random_like_float", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("casted", TensorProto.INT64, [2, 3]),
            helper.make_tensor_value_info("eye", TensorProto.FLOAT, [3, 3]),
            helper.make_tensor_value_info("trilu", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("unique_y", TensorProto.INT64, [3]),
            helper.make_tensor_value_info("unique_idx", TensorProto.INT64, [3]),
            helper.make_tensor_value_info("unique_inv", TensorProto.INT64, [6]),
            helper.make_tensor_value_info("unique_counts", TensorProto.INT64, [3]),
        ],
        value_info=[
            helper.make_tensor_value_info("sum", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("prelu", TensorProto.FLOAT, [2, 3]),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == [
        "RandomUniform", "RandomUniformLike", "RandomUniformLike",
        "Sum", "PRelu", "CastLike", "EyeLike", "Trilu", "Unique"
    ]
    assert ops[1].dtype is None
    assert ops[2].dtype == "float32"
