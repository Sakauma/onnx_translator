# /**
#   ******************************************************************************
#   * @file        test_operator_shape_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用 ONNX reference 验证形状、索引和张量重排算子的混合精度语义。
#   * @details     2026.06.04  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

from operator_test_context import *  # noqa: F401,F403
from nn.Operators import (
    Compress,
    Constant,
    ConstantOfShape,
    DepthToSpace,
    Expand,
    EyeLike,
    Flatten,
    Identity,
    OneHot,
    Pad,
    Range,
    Reshape,
    ReverseSequence,
    Shape,
    Size,
    SpaceToDepth,
    Split,
    Squeeze,
    Tile,
    Transpose,
    Unsqueeze,
)


# 构造 Tensor，避免每个断言重复 dtype、shape 和 data 样板。
def _tensor(data, dtype):
    return Tensor(*data.shape, dtype=dtype, data=data)


# 调用 ONNX reference evaluator，返回指定节点的参考输出。
def _onnx_reference(op_name, inputs, protos, attrs, output_shapes, output_protos=None):
    output_protos = output_protos or [protos[0]] * len(output_shapes)
    input_names = [f"i{i}" for i in range(len(inputs))]
    output_names = [f"o{i}" for i in range(len(output_shapes))]
    graph = helper.make_graph(
        [helper.make_node(op_name, input_names, output_names, **attrs)],
        f"{op_name}_reference",
        [
            helper.make_tensor_value_info(name, proto, list(value.shape))
            for name, proto, value in zip(input_names, protos, inputs)
        ],
        [
            helper.make_tensor_value_info(name, proto, list(shape))
            for name, proto, shape in zip(output_names, output_protos, output_shapes)
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return ReferenceEvaluator(model).run(None, dict(zip(input_names, inputs)))


# 比较单输出 Tensor 与 ONNX reference 输出。
def _assert_tensor_matches(actual, expected):
    if np.issubdtype(actual.data.dtype, np.floating):
        np.testing.assert_allclose(actual.data, expected, rtol=1e-3, atol=1e-3)
    else:
        np.testing.assert_array_equal(actual.data, expected)


# 验证基础 shape transform 算子在 float16 数据上与 ONNX reference 对齐。
def test_c_backend_shape_transform_ops_float16_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    shape = np.array([0, -1], dtype=np.int64)
    _assert_tensor_matches(
        Reshape(["x", "shape"], ["y"], dtype="float16").forward(_tensor(x, "float16"), _tensor(shape, "int64"))["tensor"],
        _onnx_reference("Reshape", [x, shape], [TensorProto.FLOAT16, TensorProto.INT64], {}, [(2, 12)])[0],
    )
    _assert_tensor_matches(
        Flatten(["x"], ["y"], axis=-1, dtype="float16").forward(_tensor(x, "float16"))["tensor"],
        _onnx_reference("Flatten", [x], [TensorProto.FLOAT16], {"axis": -1}, [(6, 4)])[0],
    )
    _assert_tensor_matches(
        Transpose(["x"], ["y"], perm=[2, 0, 1], dtype="float16").forward(_tensor(x, "float16"))["tensor"],
        _onnx_reference("Transpose", [x], [TensorProto.FLOAT16], {"perm": [2, 0, 1]}, [(4, 2, 3)])[0],
    )

    axes = np.array([1], dtype=np.int64)
    squeezed_input = np.arange(6, dtype=np.float16).reshape(2, 1, 3)
    _assert_tensor_matches(
        Squeeze(["x", "axes"], ["y"], dtype="float16").forward(_tensor(squeezed_input, "float16"), _tensor(axes, "int64"))["tensor"],
        _onnx_reference("Squeeze", [squeezed_input, axes], [TensorProto.FLOAT16, TensorProto.INT64], {}, [(2, 3)])[0],
    )

    unsqueeze_axes = np.array([0, 2], dtype=np.int64)
    unsqueeze_input = np.arange(6, dtype=np.float16).reshape(2, 3)
    _assert_tensor_matches(
        Unsqueeze(["x", "axes"], ["y"], dtype="float16").forward(_tensor(unsqueeze_input, "float16"), _tensor(unsqueeze_axes, "int64"))["tensor"],
        _onnx_reference("Unsqueeze", [unsqueeze_input, unsqueeze_axes], [TensorProto.FLOAT16, TensorProto.INT64], {}, [(1, 2, 1, 3)])[0],
    )


# 验证广播、重复、填充和切片类张量变换与 ONNX reference 对齐。
def test_c_backend_tensor_rearrangement_ops_float16_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    expand_input = np.arange(6, dtype=np.float16).reshape(2, 1, 3)
    expand_shape = np.array([2, 4, 3], dtype=np.int64)
    _assert_tensor_matches(
        Expand(["x", "shape"], ["y"], dtype="float16").forward(_tensor(expand_input, "float16"), _tensor(expand_shape, "int64"))["tensor"],
        _onnx_reference("Expand", [expand_input, expand_shape], [TensorProto.FLOAT16, TensorProto.INT64], {}, [(2, 4, 3)])[0],
    )

    x = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    repeats = np.array([1, 2, 1], dtype=np.int64)
    _assert_tensor_matches(
        Tile(["x", "repeats"], ["y"], dtype="float16").forward(_tensor(x, "float16"), _tensor(repeats, "int64"))["tensor"],
        _onnx_reference("Tile", [x, repeats], [TensorProto.FLOAT16, TensorProto.INT64], {}, [(2, 6, 4)])[0],
    )

    pads = np.array([0, 1, 1, 0, 1, 0], dtype=np.int64)
    pad_value = np.array(-2, dtype=np.float16)
    _assert_tensor_matches(
        Pad(["x", "pads", "value"], ["y"], mode="constant", dtype="float16").forward(_tensor(x, "float16"), _tensor(pads, "int64"), _tensor(pad_value, "float16"))["tensor"],
        _onnx_reference("Pad", [x, pads, pad_value], [TensorProto.FLOAT16, TensorProto.INT64, TensorProto.FLOAT16], {"mode": "constant"}, [(2, 5, 5)])[0],
    )

    starts = np.array([0, 1], dtype=np.int64)
    ends = np.array([2, 4], dtype=np.int64)
    axes = np.array([0, 2], dtype=np.int64)
    steps = np.array([1, 2], dtype=np.int64)
    _assert_tensor_matches(
        Slice(["x", "starts", "ends", "axes", "steps"], ["y"], dtype="float16").forward(
            _tensor(x, "float16"), _tensor(starts, "int64"), _tensor(ends, "int64"), _tensor(axes, "int64"), _tensor(steps, "int64")
        )["tensor"],
        _onnx_reference(
            "Slice",
            [x, starts, ends, axes, steps],
            [TensorProto.FLOAT16, TensorProto.INT64, TensorProto.INT64, TensorProto.INT64, TensorProto.INT64],
            {},
            [(2, 3, 2)],
        )[0],
    )


# 验证 Concat/Split/Where/Compress 等组合与筛选算子。
def test_c_backend_selection_and_join_ops_float16_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    left = np.ones((2, 1), dtype=np.float16)
    right = np.zeros((2, 2), dtype=np.float16)
    _assert_tensor_matches(
        Concat(["a", "b"], ["y"], axis=1, dtype="float16").forward(_tensor(left, "float16"), _tensor(right, "float16"))["tensor"],
        _onnx_reference("Concat", [left, right], [TensorProto.FLOAT16, TensorProto.FLOAT16], {"axis": 1}, [(2, 3)])[0],
    )

    split_input = np.arange(6, dtype=np.float16).reshape(2, 3)
    split = np.array([1, 2], dtype=np.int64)
    actual_split = Split(["x", "split"], ["a", "b"], axis=1, dtype="float16").forward(_tensor(split_input, "float16"), _tensor(split, "int64"))["tensor"]
    expected_split = _onnx_reference("Split", [split_input, split], [TensorProto.FLOAT16, TensorProto.INT64], {"axis": 1}, [(2, 1), (2, 2)])
    for actual, expected in zip(actual_split, expected_split):
        _assert_tensor_matches(actual, expected)

    condition = np.array([[True, False, True], [False, True, False]], dtype=bool)
    where_x = np.arange(6, dtype=np.float16).reshape(2, 3)
    where_y = -where_x
    _assert_tensor_matches(
        Where(["cond", "x", "y"], ["z"], dtype="float16").forward(_tensor(condition, "bool"), _tensor(where_x, "float16"), _tensor(where_y, "float16"))["tensor"],
        _onnx_reference("Where", [condition, where_x, where_y], [TensorProto.BOOL, TensorProto.FLOAT16, TensorProto.FLOAT16], {}, [(2, 3)])[0],
    )

    compress_condition = np.array([True, False, True], dtype=bool)
    _assert_tensor_matches(
        Compress(["x", "condition"], ["y"], axis=1, dtype="float16").forward(_tensor(split_input, "float16"), _tensor(compress_condition, "bool"))["tensor"],
        _onnx_reference("Compress", [split_input, compress_condition], [TensorProto.FLOAT16, TensorProto.BOOL], {"axis": 1}, [(2, 2)])[0],
    )


# 验证常量、形状和索引构造类算子与 ONNX reference 对齐。
def test_c_backend_shape_value_ops_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    base = np.empty((3, 4), dtype=np.float16)
    _assert_tensor_matches(
        EyeLike(["x"], ["y"], k=-1, dtype="float16").forward(_tensor(base, "float16"))["tensor"],
        _onnx_reference("EyeLike", [base], [TensorProto.FLOAT16], {"k": -1}, [(3, 4)])[0],
    )

    shape = np.array([2, 3], dtype=np.int64)
    fill_value = np.array([1.5], dtype=np.float16)
    expected_constant = _onnx_reference(
        "ConstantOfShape",
        [shape],
        [TensorProto.INT64],
        {"value": numpy_helper.from_array(fill_value, name="value")},
        [(2, 3)],
        [TensorProto.FLOAT16],
    )[0]
    _assert_tensor_matches(
        ConstantOfShape(["shape"], ["y"], value=fill_value, dtype="float16").forward(_tensor(shape, "int64"))["tensor"],
        expected_constant,
    )

    start = np.array(1, dtype=np.int64)
    limit = np.array(7, dtype=np.int64)
    delta = np.array(2, dtype=np.int64)
    _assert_tensor_matches(
        Range(["start", "limit", "delta"], ["y"], dtype="int64").forward(_tensor(start, "int64"), _tensor(limit, "int64"), _tensor(delta, "int64"))["tensor"],
        _onnx_reference("Range", [start, limit, delta], [TensorProto.INT64] * 3, {}, [(3,)])[0],
    )

    size_input = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    _assert_tensor_matches(
        Size(["x"], ["y"]).forward(_tensor(size_input, "float16"))["tensor"],
        _onnx_reference("Size", [size_input], [TensorProto.FLOAT16], {}, [()], [TensorProto.INT64])[0],
    )

    indices = np.array([0, 2, -1], dtype=np.int64)
    depth = np.array(3, dtype=np.int64)
    values = np.array([0.5, 2.0], dtype=np.float16)
    _assert_tensor_matches(
        OneHot(["indices", "depth", "values"], ["y"], axis=-1, dtype="float16").forward(_tensor(indices, "int64"), _tensor(depth, "int64"), _tensor(values, "float16"))["tensor"],
        _onnx_reference("OneHot", [indices, depth, values], [TensorProto.INT64, TensorProto.INT64, TensorProto.FLOAT16], {"axis": -1}, [(3, 3)], [TensorProto.FLOAT16])[0],
    )

    identity_input = np.array([[1, 2], [3, 4]], dtype=np.int64)
    _assert_tensor_matches(
        Identity(["x"], ["y"], dtype="int64").forward(_tensor(identity_input, "int64"))["tensor"],
        _onnx_reference("Identity", [identity_input], [TensorProto.INT64], {}, [(2, 2)])[0],
    )

    shape_input = np.zeros((2, 3, 4), dtype=np.float16)
    _assert_tensor_matches(
        Shape(["x"], ["shape"], start=1, end=-1).forward(_tensor(shape_input, "float16"))["tensor"],
        _onnx_reference("Shape", [shape_input], [TensorProto.FLOAT16], {"start": 1, "end": -1}, [(1,)], [TensorProto.INT64])[0],
    )

    constant_values = np.array([1, 2, 3], dtype=np.int64)
    expected_constant_node = _onnx_reference(
        "Constant",
        [],
        [],
        {"value": numpy_helper.from_array(constant_values, name="value")},
        [(3,)],
        [TensorProto.INT64],
    )[0]
    _assert_tensor_matches(
        Constant([], ["y"], value=constant_values, dtype="int64").forward()["tensor"],
        expected_constant_node,
    )


# 验证空间重排、序列反转和三角截取类算子与 ONNX reference 对齐。
def test_c_backend_spatial_and_sequence_shape_ops_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    space_input = np.arange(1 * 1 * 4 * 4, dtype=np.float16).reshape(1, 1, 4, 4)
    _assert_tensor_matches(
        SpaceToDepth(["x"], ["y"], blocksize=2, dtype="float16").forward(_tensor(space_input, "float16"))["tensor"],
        _onnx_reference("SpaceToDepth", [space_input], [TensorProto.FLOAT16], {"blocksize": 2}, [(1, 4, 2, 2)])[0],
    )

    depth_input = np.arange(1 * 4 * 2 * 2, dtype=np.float16).reshape(1, 4, 2, 2)
    _assert_tensor_matches(
        DepthToSpace(["x"], ["y"], blocksize=2, mode="DCR", dtype="float16").forward(_tensor(depth_input, "float16"))["tensor"],
        _onnx_reference("DepthToSpace", [depth_input], [TensorProto.FLOAT16], {"blocksize": 2, "mode": "DCR"}, [(1, 1, 4, 4)])[0],
    )

    sequence_input = np.arange(3 * 2 * 2, dtype=np.float16).reshape(3, 2, 2)
    sequence_lens = np.array([3, 2], dtype=np.int64)
    _assert_tensor_matches(
        ReverseSequence(["x", "sequence_lens"], ["y"], time_axis=0, batch_axis=1, dtype="float16").forward(_tensor(sequence_input, "float16"), _tensor(sequence_lens, "int64"))["tensor"],
        _onnx_reference("ReverseSequence", [sequence_input, sequence_lens], [TensorProto.FLOAT16, TensorProto.INT64], {"time_axis": 0, "batch_axis": 1}, [(3, 2, 2)])[0],
    )

    k = np.array(-1, dtype=np.int64)
    triangular_input = np.arange(12, dtype=np.float16).reshape(3, 4)
    _assert_tensor_matches(
        Trilu(["x", "k"], ["y"], upper=0, dtype="float16").forward(_tensor(triangular_input, "float16"), _tensor(k, "int64"))["tensor"],
        _onnx_reference("Trilu", [triangular_input, k], [TensorProto.FLOAT16, TensorProto.INT64], {"upper": 0}, [(3, 4)])[0],
    )
