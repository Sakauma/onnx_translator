# /**
#   ******************************************************************************
#   * @file        test_operator_complex_attribute_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用 ONNX reference 验证复杂属性算子的官方语义和混合精度路径。
#   * @details     2026.06.05  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from onnx.reference import ReferenceEvaluator

from operator_test_context import *  # noqa: F401,F403
from nn.Operators import Einsum, GridSample, MaxUnpool, Pad, Resize


# 构造 Tensor，统一 shape、dtype 和 data 的样板代码。
def _tensor(data, dtype):
    data = np.asarray(data)
    return Tensor(*data.shape, dtype=dtype, data=data)


# 调用 ONNX reference evaluator，得到单节点模型的官方输出。
def _onnx_reference(op_name, inputs, protos, attrs, output_shapes, output_protos=None, opset=17, input_names=None):
    output_protos = output_protos or [protos[0]] * len(output_shapes)
    input_names = input_names or [f"i{i}" for i in range(len(inputs))]
    output_names = [f"o{i}" for i in range(len(output_shapes))]
    graph_inputs = [
        helper.make_tensor_value_info(name, proto, list(np.asarray(value).shape))
        for name, proto, value in zip(input_names, protos, inputs)
        if name
    ]
    feeds = {
        name: value
        for name, value in zip(input_names, inputs)
        if name
    }
    graph = helper.make_graph(
        [helper.make_node(op_name, input_names, output_names, **attrs)],
        f"{op_name}_reference",
        graph_inputs,
        [
            helper.make_tensor_value_info(name, proto, list(shape))
            for name, proto, shape in zip(output_names, output_protos, output_shapes)
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return ReferenceEvaluator(model).run(None, feeds)


# 浮点输出按容差比较，整数和布尔输出按精确值比较。
def _assert_tensor_matches(actual, expected, rtol=1e-4, atol=1e-4):
    actual_data = np.asarray(actual.data)
    expected_data = np.asarray(expected)
    assert actual_data.shape == expected_data.shape
    if np.issubdtype(expected_data.dtype, np.floating):
        np.testing.assert_allclose(actual_data, expected_data, rtol=rtol, atol=atol)
    else:
        np.testing.assert_array_equal(actual_data, expected_data)


# 将 float32 数值转换成 bfloat16 的 uint16 位模式，匹配 Tensor 内部存储。
def _bf16_bits(values):
    data = np.asarray(values, dtype=np.float32)
    bits = data.view(np.uint32)
    lsb = (bits >> 16) & 1
    guard = (bits >> 15) & 1
    sticky = (bits & 0x7FFF) != 0
    rounded = bits + ((guard & (sticky | lsb)).astype(np.uint32) << 16)
    rounded = np.where(np.isnan(data), bits, rounded)
    return (rounded >> 16).astype(np.uint16)


# 将 bfloat16 的 uint16 位模式还原为 float32，用于低精度数值断言。
def _bf16_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


# 验证 Resize 的线性插值、align_corners 和 nearest ceil 模式与 ONNX reference 一致。
def test_c_backend_resize_coordinate_and_nearest_modes_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.array([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]], dtype=np.float32)
    scales = np.array([1.0, 1.0, 1.5, 2.0], dtype=np.float32)
    expected_linear = _onnx_reference(
        "Resize",
        [x, None, scales],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
        {"mode": "linear", "coordinate_transformation_mode": "align_corners"},
        [(1, 1, 3, 6)],
        input_names=["x", "", "scales"],
    )[0]
    actual_linear = Resize(
        ["x", "roi", "scales"],
        ["y"],
        mode="linear",
        coord_mode="align_corners",
        nearest_mode="floor",
        dtype="float32",
    ).forward(_tensor(x, "float32"), None, _tensor(scales, "float32"))["tensor"]
    _assert_tensor_matches(actual_linear, expected_linear, rtol=1e-5, atol=1e-5)

    sizes = np.array([1, 1, 3, 5], dtype=np.int64)
    expected_nearest = _onnx_reference(
        "Resize",
        [x, None, None, sizes],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64],
        {"mode": "nearest", "coordinate_transformation_mode": "asymmetric", "nearest_mode": "ceil"},
        [(1, 1, 3, 5)],
        input_names=["x", "", "", "sizes"],
    )[0]
    actual_nearest = Resize(
        ["x", "roi", "", "sizes"],
        ["y"],
        mode="nearest",
        coord_mode="asymmetric",
        nearest_mode="ceil",
        dtype="float32",
    ).forward(_tensor(x, "float32"), None, None, _tensor(sizes, "int64"))["tensor"]
    _assert_tensor_matches(actual_nearest, expected_nearest)


# 验证 Resize 的 reference fallback 覆盖 cubic 和 round_prefer_ceil 等 C 快速路径之外的官方属性。
def test_resize_reference_fallback_matches_onnx_for_cubic_and_round_prefer_ceil():
    x = np.array([[[[0.0, 1.0], [2.0, 3.0]]]], dtype=np.float32)
    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    expected_cubic = _onnx_reference(
        "Resize",
        [x, None, scales],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
        {"mode": "cubic", "coordinate_transformation_mode": "half_pixel", "cubic_coeff_a": -0.5},
        [(1, 1, 4, 4)],
        input_names=["x", "", "scales"],
    )[0]
    actual_cubic = Resize(
        ["x", "roi", "scales"],
        ["y"],
        mode="cubic",
        coord_mode="half_pixel",
        cubic_coeff_a=-0.5,
        dtype="float32",
    ).forward(_tensor(x, "float32"), None, _tensor(scales, "float32"))["tensor"]
    _assert_tensor_matches(actual_cubic, expected_cubic, rtol=1e-5, atol=1e-5)

    nearest_expected = _onnx_reference(
        "Resize",
        [x, None, scales],
        [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
        {"mode": "nearest", "coordinate_transformation_mode": "half_pixel", "nearest_mode": "round_prefer_ceil"},
        [(1, 1, 4, 4)],
        input_names=["x", "", "scales"],
    )[0]
    nearest_actual = Resize(
        ["x", "roi", "scales"],
        ["y"],
        mode="nearest",
        coord_mode="half_pixel",
        nearest_mode="round_prefer_ceil",
        dtype="float32",
    ).forward(_tensor(x, "float32"), None, _tensor(scales, "float32"))["tensor"]
    _assert_tensor_matches(nearest_actual, nearest_expected)


# 验证 Pad 的 edge、reflect 以及负 pad 裁剪语义与 ONNX reference 一致。
def test_c_backend_pad_edge_reflect_and_crop_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    pads_edge = np.array([1, 2, 0, 1], dtype=np.int64)
    expected_edge = _onnx_reference(
        "Pad",
        [data, pads_edge],
        [TensorProto.FLOAT, TensorProto.INT64],
        {"mode": "edge"},
        [(3, 6)],
    )[0]
    actual_edge = Pad(["x", "pads"], ["y"], mode="edge", dtype="float32").forward(
        _tensor(data, "float32"),
        _tensor(pads_edge, "int64"),
    )["tensor"]
    _assert_tensor_matches(actual_edge, expected_edge)

    pads_reflect = np.array([1, 1, 1, 1], dtype=np.int64)
    expected_reflect = _onnx_reference(
        "Pad",
        [data, pads_reflect],
        [TensorProto.FLOAT, TensorProto.INT64],
        {"mode": "reflect"},
        [(4, 5)],
    )[0]
    actual_reflect = Pad(["x", "pads"], ["y"], mode="reflect", dtype="float32").forward(
        _tensor(data, "float32"),
        _tensor(pads_reflect, "int64"),
    )["tensor"]
    _assert_tensor_matches(actual_reflect, expected_reflect)

    pads_crop = np.array([0, -1, 0, -1], dtype=np.int64)
    expected_crop = data[:, 1:2]
    actual_crop = Pad(["x", "pads"], ["y"], mode="constant", dtype="float32").forward(
        _tensor(data, "float32"),
        _tensor(pads_crop, "int64"),
    )["tensor"]
    _assert_tensor_matches(actual_crop, expected_crop)


# 验证 GridSample 的 nearest/border 和 bilinear/reflection 属性组合。
def test_c_backend_grid_sample_modes_match_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]], dtype=np.float32)
    grid = np.array(
        [[[[0.0, 0.0], [1.2, -1.2]], [[-1.0, 1.0], [0.5, -0.5]]]],
        dtype=np.float32,
    )
    expected_nearest = _onnx_reference(
        "GridSample",
        [x, grid],
        [TensorProto.FLOAT, TensorProto.FLOAT],
        {"mode": "nearest", "padding_mode": "border", "align_corners": 1},
        [(1, 1, 2, 2)],
    )[0]
    actual_nearest = GridSample(
        ["x", "grid"],
        ["y"],
        mode="nearest",
        padding_mode="border",
        align_corners=1,
        dtype="float32",
    ).forward(_tensor(x, "float32"), _tensor(grid, "float32"))["tensor"]
    _assert_tensor_matches(actual_nearest, expected_nearest)

    expected_linear = _onnx_reference(
        "GridSample",
        [x, grid],
        [TensorProto.FLOAT, TensorProto.FLOAT],
        {"mode": "linear", "padding_mode": "reflection", "align_corners": 0},
        [(1, 1, 2, 2)],
    )[0]
    actual_linear = GridSample(
        ["x", "grid"],
        ["y"],
        mode="linear",
        padding_mode="reflection",
        align_corners=0,
        dtype="float32",
    ).forward(_tensor(x, "float32"), _tensor(grid, "float32"))["tensor"]
    _assert_tensor_matches(actual_linear, expected_linear, rtol=1e-5, atol=1e-5)


# 验证 MaxUnpool 的显式 output_shape 和索引写回语义。
def test_c_backend_max_unpool_explicit_output_shape_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    indices = np.array([[[[5, 7], [13, 15]]]], dtype=np.int64)
    output_shape = np.array([1, 1, 5, 5], dtype=np.int64)
    expected = _onnx_reference(
        "MaxUnpool",
        [x, indices, output_shape],
        [TensorProto.FLOAT, TensorProto.INT64, TensorProto.INT64],
        {"kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2]},
        [(1, 1, 5, 5)],
    )[0]
    actual = MaxUnpool(["x", "indices", "shape"], ["y"], kernel_shape=[2, 2], pads=[0, 0, 0, 0], strides=[2, 2], dtype="float32").forward(
        _tensor(x, "float32"),
        _tensor(indices, "int64"),
        _tensor(output_shape, "int64"),
    )["tensor"]
    _assert_tensor_matches(actual, expected)


# 验证 Einsum 的 ellipsis、重复标签和 bfloat16 位存储路径。
def test_c_backend_einsum_ellipsis_repeated_labels_and_bfloat16_match_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    left = (np.arange(2 * 1 * 3 * 4, dtype=np.float32).reshape(2, 1, 3, 4) - 5.0) / 7.0
    right = (np.arange(1 * 5 * 4 * 2, dtype=np.float32).reshape(1, 5, 4, 2) + 2.0) / 6.0
    expected = _onnx_reference(
        "Einsum",
        [left, right],
        [TensorProto.FLOAT, TensorProto.FLOAT],
        {"equation": "...ij,...jk->...ik"},
        [(2, 5, 3, 2)],
    )[0]
    actual = Einsum(["left", "right"], ["y"], equation="...ij,...jk->...ik", dtype="float32").forward(
        _tensor(left, "float32"),
        _tensor(right, "float32"),
    )["tensor"]
    _assert_tensor_matches(actual, expected, rtol=1e-5, atol=1e-5)

    square = np.array([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    diagonal_expected = _onnx_reference(
        "Einsum",
        [square],
        [TensorProto.FLOAT],
        {"equation": "ii->i"},
        [(3,)],
    )[0]
    diagonal_actual = Einsum(["x"], ["y"], equation="ii->i", dtype="float32").forward(_tensor(square, "float32"))["tensor"]
    _assert_tensor_matches(diagonal_actual, diagonal_expected)

    bf16_values = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    bf16_eye = np.eye(2, dtype=np.float32)
    bf16_actual = Einsum(["x", "eye"], ["y"], equation="ij,jk->ik", dtype="bfloat16").forward(
        _tensor(_bf16_bits(bf16_values), "bfloat16"),
        _tensor(_bf16_bits(bf16_eye), "bfloat16"),
    )["tensor"]
    np.testing.assert_allclose(_bf16_to_float32(bf16_actual.data), bf16_values, rtol=2e-2, atol=2e-2)
