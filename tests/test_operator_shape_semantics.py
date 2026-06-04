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

from conftest import _disable_c_backend
from operator_test_context import *  # noqa: F401,F403
from nn.Operators import (
    AffineGrid,
    BitCast,
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
def _onnx_reference(op_name, inputs, protos, attrs, output_shapes, output_protos=None, opset=17):
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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return ReferenceEvaluator(model).run(None, dict(zip(input_names, inputs)))


# 比较单输出 Tensor 与 ONNX reference 输出。
def _assert_tensor_matches(actual, expected):
    if np.issubdtype(actual.data.dtype, np.floating):
        np.testing.assert_allclose(actual.data, expected, rtol=1e-3, atol=1e-3)
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


def _bf16_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


# 复用 ONNX reference 的 AffineGrid 网格构造函数，避免测试中重新手写一套坐标规则。
def _affine_grid_reference(theta, size, align_corners):
    from onnx.reference.ops.op_affine_grid import apply_affine_transform, construct_original_grid

    original_grid = construct_original_grid(list(map(int, size[2:])), align_corners)
    return apply_affine_transform(theta, original_grid)


# 验证 Range 的 Python fallback 会解码 bfloat16 起止和步长，并按 bfloat16 位模式写回。
def test_python_range_fallback_bfloat16_decodes_scalar_inputs(monkeypatch):
    _disable_c_backend(monkeypatch)

    start = _tensor(_bf16_bits(np.array(-1.0, dtype=np.float32)), "bfloat16")
    limit = _tensor(_bf16_bits(np.array(1.0, dtype=np.float32)), "bfloat16")
    delta = _tensor(_bf16_bits(np.array(0.5, dtype=np.float32)), "bfloat16")
    actual = Range(["start", "limit", "delta"], ["y"], dtype="bfloat16").forward(start, limit, delta)["tensor"]
    np.testing.assert_array_equal(actual.data, _bf16_bits(np.array([-1.0, -0.5, 0.0, 0.5], dtype=np.float32)))
    np.testing.assert_allclose(_bf16_to_float32(actual.data), np.array([-1.0, -0.5, 0.0, 0.5], dtype=np.float32))


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


# 验证 BitCast 的 float32 到 int32 路径与 ONNX reference 的原始位重解释一致。
def test_c_backend_bitcast_float32_to_int32_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.array([1.0, -2.5, 0.0, np.float32(np.pi)], dtype=np.float32).reshape(2, 2)
    expected = _onnx_reference(
        "BitCast",
        [x],
        [TensorProto.FLOAT],
        {"to": TensorProto.INT32},
        [(2, 2)],
        [TensorProto.INT32],
        opset=26,
    )[0]
    actual = BitCast(["x"], ["y"], dtype="int32").forward(_tensor(x, "float32"))["tensor"]
    np.testing.assert_array_equal(actual.data, expected)


# 验证 BitCast 的 int32 到 float32 路径不会执行数值转换，只重解释二进制位。
def test_c_backend_bitcast_int32_to_float32_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    raw = np.array([0x3F800000, 0xC0200000, 0x00000000, 0x40490FDB], dtype=np.uint32).view(np.int32).reshape(2, 2)
    expected = _onnx_reference(
        "BitCast",
        [raw],
        [TensorProto.INT32],
        {"to": TensorProto.FLOAT},
        [(2, 2)],
        [TensorProto.FLOAT],
        opset=26,
    )[0]
    actual = BitCast(["x"], ["y"], dtype="float32").forward(_tensor(raw, "int32"))["tensor"]
    np.testing.assert_array_equal(actual.data.view(np.uint32), expected.view(np.uint32))


# 验证 BitCast 对 bfloat16 和 float8 混合精度位存储执行原样复制。
def test_c_backend_bitcast_low_precision_preserves_raw_bits():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    bf16_bits = _bf16_bits(np.array([[1.0, -2.0], [0.5, 3.25]], dtype=np.float32))
    bf16_actual = BitCast(["x"], ["y"], dtype="uint16").forward(_tensor(bf16_bits, "bfloat16"))["tensor"]
    np.testing.assert_array_equal(bf16_actual.data, bf16_bits)

    fp8_bits = np.array([[0x00, 0x3C, 0x80], [0x7E, 0x11, 0xA4]], dtype=np.uint8)
    fp8_actual = BitCast(["x"], ["y"], dtype="uint8").forward(_tensor(fp8_bits, "float8_e4m3"))["tensor"]
    np.testing.assert_array_equal(fp8_actual.data, fp8_bits)


# 验证 BitCast 导入时保留官方必需的 to 属性。
def test_onnx_import_bitcast_preserves_target_dtype(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("BitCast", ["x"], ["y"], to=TensorProto.INT32)],
        "bitcast_import",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.INT32, [2, 2])],
    )
    model_path = tmp_path / "bitcast.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 26)]), model_path)

    imported = [op for op in ONNXImport(str(model_path), strict=True) if isinstance(op, BitCast)]
    assert len(imported) == 1
    assert imported[0].dtype == "int32"
    assert imported[0].version == "26"


# 验证 AffineGrid 的 2D 采样网格生成与 ONNX reference 对齐。
def test_c_backend_affine_grid_2d_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    theta = np.array(
        [
            [[1.0, 0.0, 0.1], [0.0, 1.0, -0.2]],
            [[0.8, 0.1, 0.0], [-0.1, 0.9, 0.2]],
        ],
        dtype=np.float32,
    )
    size = np.array([2, 1, 3, 4], dtype=np.int64)
    expected = _onnx_reference(
        "AffineGrid",
        [theta, size],
        [TensorProto.FLOAT, TensorProto.INT64],
        {"align_corners": 0},
        [(2, 3, 4, 2)],
        [TensorProto.FLOAT],
        opset=20,
    )[0]
    actual = AffineGrid(["theta", "size"], ["grid"], align_corners=0, dtype="float32").forward(
        _tensor(theta, "float32"),
        _tensor(size, "int64"),
    )["tensor"]
    _assert_tensor_matches(actual, expected)


# 验证 AffineGrid 的 3D align_corners 路径与 ONNX reference 对齐。
def test_c_backend_affine_grid_3d_align_corners_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    theta = np.array(
        [[[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, -0.2], [0.0, 0.0, 1.0, 0.3]]],
        dtype=np.float32,
    )
    size = np.array([1, 1, 2, 3, 2], dtype=np.int64)
    expected = _onnx_reference(
        "AffineGrid",
        [theta, size],
        [TensorProto.FLOAT, TensorProto.INT64],
        {"align_corners": 1},
        [(1, 2, 3, 2, 3)],
        [TensorProto.FLOAT],
        opset=20,
    )[0]
    actual = AffineGrid(["theta", "size"], ["grid"], align_corners=1, dtype="float32").forward(
        _tensor(theta, "float32"),
        _tensor(size, "int64"),
    )["tensor"]
    _assert_tensor_matches(actual, expected)


# 验证 AffineGrid 的 bfloat16 路径会解码 theta 位模式并按位写回网格结果。
def test_c_backend_affine_grid_bfloat16_decodes_and_writes_bit_storage():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    theta_values = np.array(
        [
            [[1.0, 0.0, 0.125], [0.0, 1.0, -0.25]],
            [[0.75, 0.125, 0.0], [-0.125, 0.875, 0.25]],
        ],
        dtype=np.float32,
    )
    size = np.array([2, 1, 3, 4], dtype=np.int64)
    theta_bits = _bf16_bits(theta_values)
    decoded_theta = _bf16_to_float32(theta_bits)
    expected = _affine_grid_reference(decoded_theta, size, 0)
    expected_bits = _bf16_bits(expected)

    actual = AffineGrid(["theta", "size"], ["grid"], align_corners=0, dtype="bfloat16").forward(
        _tensor(theta_bits, "bfloat16"),
        _tensor(size, "int64"),
    )["tensor"]
    np.testing.assert_array_equal(actual.data, expected_bits)
    np.testing.assert_allclose(_bf16_to_float32(actual.data), _bf16_to_float32(expected_bits), rtol=1e-2, atol=1e-2)


# 验证 AffineGrid 导入时保留 align_corners 属性。
def test_onnx_import_affine_grid_preserves_align_corners(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("AffineGrid", ["theta", "size"], ["grid"], align_corners=1)],
        "affine_grid_import",
        [
            helper.make_tensor_value_info("theta", TensorProto.FLOAT, [1, 2, 3]),
            helper.make_tensor_value_info("size", TensorProto.INT64, [4]),
        ],
        [helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 2, 3, 2])],
    )
    model_path = tmp_path / "affine_grid.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)]), model_path)

    imported = [op for op in ONNXImport(str(model_path), strict=True) if isinstance(op, AffineGrid)]
    assert len(imported) == 1
    assert imported[0].align_corners == 1
    assert imported[0].version == "20"


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
