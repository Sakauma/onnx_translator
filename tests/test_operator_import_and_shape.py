# /**
#   ******************************************************************************
#   * @file        test_operator_import_and_shape.py
#   * @author      Egor Izmaylov
#   * @brief       覆盖 ONNX 导入、静态形状推断和基础算子形状行为。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from conftest import _disable_c_backend
from operator_test_context import *  # noqa: F401,F403


def test_quantize_linear_forward_shape_and_optional_zero_point_import(tmp_path, monkeypatch):
    _disable_c_backend(monkeypatch)
    x = Tensor_(2, 3, dtype="float32")
    scale = Tensor_(1, dtype="float32")
    op = QuantizeLinear(["x", "scale"], ["y"], dtype="uint8")

    out = op.forward_(x, scale)["tensor"]

    assert out.size == (2, 3)
    assert out.dtype == "uint8"

    model_path = tmp_path / "optional_quant.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["x", "scale"], ["qx"]),
            helper.make_node("DequantizeLinear", ["qx", "scale"], ["y"]),
        ],
        "optional_quant",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        value_info=[helper.make_tensor_value_info("qx", TensorProto.UINT8, [2, 3])],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == ["QuantizeLinear", "DequantizeLinear"]

    qx_data = np.array([[[[10, 11], [12, 13]], [[20, 21], [22, 23]], [[30, 31], [32, 33]]]], dtype=np.uint8)
    qx = Tensor(*qx_data.shape, dtype="uint8", data=qx_data)
    x_scale = Tensor(3, dtype="float32", data=np.array([0.5, 1.0, 2.0], dtype=np.float32))
    x_zp = Tensor(3, dtype="uint8", data=np.array([10, 20, 30], dtype=np.uint8))
    dequant = DequantizeLinear(["x", "scale", "zp"], ["y"], axis=1, dtype="float32").forward(qx, x_scale, x_zp)["tensor"]
    expected = (qx_data.astype(np.float32) - np.array([10, 20, 30], dtype=np.float32).reshape(1, 3, 1, 1)) * np.array([0.5, 1.0, 2.0], dtype=np.float32).reshape(1, 3, 1, 1)
    np.testing.assert_array_equal(dequant.data, expected)
    assert dequant.size == qx_data.shape

    axis_model_path = tmp_path / "dequant_axis.onnx"
    axis_graph = helper.make_graph(
        [helper.make_node("DequantizeLinear", ["x", "scale", "zp"], ["y"], axis=1)],
        "dequant_axis",
        [
            helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 3, 2, 2]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("zp", TensorProto.UINT8, [3]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 2, 2])],
    )
    onnx.save(helper.make_model(axis_graph, opset_imports=[helper.make_opsetid("", 17)]), axis_model_path)
    imported_dequant = [op for op in ONNXImport(str(axis_model_path), strict=True) if isinstance(op, DequantizeLinear)]
    assert imported_dequant[0].axis == 1

def test_constant_import_supports_onnx17_scalar_and_string_attrs(tmp_path, monkeypatch):
    _disable_c_backend(monkeypatch)

    graph = helper.make_graph(
        [
            helper.make_node("Constant", [], ["float_scalar"], value_float=1.5),
            helper.make_node("Constant", [], ["int_vector"], value_ints=[1, 2, 3]),
            helper.make_node("Constant", [], ["string_vector"], value_strings=[b"alpha", b"beta"]),
        ],
        "constant_attrs",
        [],
        [
            helper.make_tensor_value_info("float_scalar", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("int_vector", TensorProto.INT64, [3]),
            helper.make_tensor_value_info("string_vector", TensorProto.STRING, [2]),
        ],
    )
    model_path = tmp_path / "constant_attrs.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), model_path)

    constants = [op for op in ONNXImport(str(model_path), strict=True) if isinstance(op, Constant)]

    assert [op.dtype for op in constants] == ["float32", "int64", "string"]
    float_out = constants[0].forward()["tensor"]
    np.testing.assert_array_equal(float_out.data, np.asarray(1.5, dtype=np.float32))
    assert float_out.size == ()
    np.testing.assert_array_equal(constants[1].forward()["tensor"].data, np.array([1, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(constants[2].forward()["tensor"].data, np.array(["alpha", "beta"], dtype=np.str_))

def test_shape_forward_does_not_mutate_end(monkeypatch):
    _disable_c_backend(monkeypatch)
    op = Shape(["x"], ["shape"])

    first = op.forward_(Tensor_(2, 3, 4, dtype="float32"))["tensor"]
    second = op.forward_(Tensor_(5, 6, dtype="float32"))["tensor"]

    assert first.size == (3,)
    assert second.size == (2,)

def test_static_shape_inference_uses_constant_inputs(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    const_shape = Tensor(2, dtype="int64", data=np.array([2, 3], dtype=np.int64))
    assert ConstantOfShape(["shape"], ["out"], dtype="float32").forward_(const_shape)["tensor"].size == (2, 3)
    scalar_shape = Tensor(0, dtype="int64", data=np.array([], dtype=np.int64))
    scalar_const = ConstantOfShape(
        ["shape"],
        ["out"],
        value=np.array([7], dtype=np.uint32),
        dtype="uint32",
    )
    scalar_out = scalar_const.forward(scalar_shape)["tensor"]
    assert scalar_out.size == ()
    assert scalar_out.dtype == "uint32"
    np.testing.assert_array_equal(scalar_out.data, np.array(7, dtype=np.uint32))
    assert scalar_const.forward_(scalar_shape)["tensor"].size == ()
    with pytest.raises(ValueError, match="single-element"):
        ConstantOfShape(["shape"], ["out"], value=np.array([1, 2], dtype=np.int64), dtype="int64").forward(const_shape)

    start = Tensor(1, dtype="int64", data=np.array([0], dtype=np.int64))
    limit = Tensor(1, dtype="int64", data=np.array([5], dtype=np.int64))
    delta = Tensor(1, dtype="int64", data=np.array([2], dtype=np.int64))
    assert Range(["start", "limit", "delta"], ["out"], dtype="int64").forward_(start, limit, delta)["tensor"].size == (3,)

    repeats = Tensor(2, dtype="int64", data=np.array([2, 3], dtype=np.int64))
    assert Tile(["x", "repeats"], ["out"], dtype="float32").forward_(Tensor_(2, 1, dtype="float32"), repeats)["tensor"].size == (4, 3)

    sizes = Tensor(4, dtype="int64", data=np.array([1, 3, 16, 16], dtype=np.int64))
    assert Resize(["x", "", "", "sizes"], ["out"], dtype="float32").forward_(Tensor_(1, 3, 8, 8, dtype="float32"), None, None, sizes)["tensor"].size == (1, 3, 16, 16)

    resize_x_data = np.array([[[1.0, 2.0, 4.0]]], dtype=np.float32)
    resize_sizes = Tensor(3, dtype="int64", data=np.array([1, 1, 5], dtype=np.int64))
    resize = Resize(
        ["x", "", "", "sizes"], ["y"], mode="cubic", coord_mode="half_pixel",
        cubic_coeff_a=-0.5, exclude_outside=1, dtype="float32"
    )
    resized = resize.forward(Tensor(*resize_x_data.shape, dtype="float32", data=resize_x_data), None, None, resize_sizes)["tensor"]
    from onnx.reference import ReferenceEvaluator
    ref_graph = helper.make_graph(
        [helper.make_node(
            "Resize",
            ["x", "", "", "sizes"],
            ["y"],
            mode="cubic",
            coordinate_transformation_mode="half_pixel",
            cubic_coeff_a=-0.5,
            exclude_outside=1,
        )],
        "resize_cubic_ref",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 3]),
            helper.make_tensor_value_info("sizes", TensorProto.INT64, [3]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 5])],
    )
    ref_model = helper.make_model(ref_graph, opset_imports=[helper.make_opsetid("", 17)])
    expected_resize = ReferenceEvaluator(ref_model).run(None, {"x": resize_x_data, "sizes": resize_sizes.data})[0]
    np.testing.assert_allclose(resized.data, expected_resize, rtol=1e-6)

    resize_model_path = tmp_path / "resize_cubic_attrs.onnx"
    onnx.save(ref_model, resize_model_path)
    resize_imported = [op for op in ONNXImport(str(resize_model_path), strict=True) if isinstance(op, Resize)]
    assert resize_imported[0].cubic_coeff_a == -0.5
    assert resize_imported[0].exclude_outside == 1

    target = Tensor(3, dtype="int64", data=np.array([2, 3, 4], dtype=np.int64))
    assert Expand(["x", "shape"], ["out"], dtype="float32").forward_(Tensor_(1, 3, 1, dtype="float32"), target)["tensor"].size == (2, 3, 4)
    string_matrix = Tensor(1, 2, dtype="string", data=np.array([["a", "b"]], dtype=np.str_))
    expanded_strings = Expand(["x", "shape"], ["out"], dtype="string").forward(
        string_matrix,
        Tensor(2, dtype="int64", data=np.array([3, 2], dtype=np.int64)),
    )["tensor"]
    np.testing.assert_array_equal(expanded_strings.data, np.array([["a", "b"], ["a", "b"], ["a", "b"]], dtype=np.str_))

    tiled_strings = Tile(["x", "repeats"], ["out"], dtype="string").forward(
        Tensor(2, dtype="string", data=np.array(["x", "y"], dtype=np.str_)),
        Tensor(1, dtype="int64", data=np.array([2], dtype=np.int64)),
    )["tensor"]
    np.testing.assert_array_equal(tiled_strings.data, np.array(["x", "y", "x", "y"], dtype=np.str_))

    selected_strings = Where(["cond", "x", "y"], ["out"], dtype="string").forward(
        Tensor(3, dtype="bool", data=np.array([True, False, True], dtype=np.bool_)),
        Tensor(3, dtype="string", data=np.array(["left0", "left1", "left2"], dtype=np.str_)),
        Tensor(3, dtype="string", data=np.array(["right0", "right1", "right2"], dtype=np.str_)),
    )["tensor"]
    np.testing.assert_array_equal(selected_strings.data, np.array(["left0", "right1", "left2"], dtype=np.str_))

    one_hot_strings = OneHot(["idx", "depth", "values"], ["out"], dtype="string").forward(
        Tensor(3, dtype="int64", data=np.array([0, 2, -1], dtype=np.int64)),
        Tensor(dtype="int64", data=np.array(3, dtype=np.int64)),
        Tensor(2, dtype="string", data=np.array(["off", "on"], dtype=np.str_)),
    )["tensor"]
    np.testing.assert_array_equal(
        one_hot_strings.data,
        np.array([
            ["on", "off", "off"],
            ["off", "off", "on"],
            ["off", "off", "on"],
        ], dtype=np.str_),
    )

    clipped_uint32 = Clip(["x", "min", "max"], ["out"], dtype="uint32").forward(
        Tensor(4, dtype="uint32", data=np.array([0, 5, 10, 20], dtype=np.uint32)),
        Tensor(dtype="uint32", data=np.array(3, dtype=np.uint32)),
        Tensor(dtype="uint32", data=np.array(12, dtype=np.uint32)),
    )["tensor"]
    assert clipped_uint32.dtype == "uint32"
    np.testing.assert_array_equal(clipped_uint32.data, np.array([3, 5, 10, 12], dtype=np.uint32))

    matmul_vec_mat_shape = MatMul(["a", "b"], ["out"], dtype="float32").forward_(
        Tensor_(3, dtype="float32"),
        Tensor_(3, 4, dtype="float32"),
    )["tensor"]
    assert matmul_vec_mat_shape.size == (4,)
    matmul_mat_vec_shape = MatMul(["a", "b"], ["out"], dtype="float32").forward_(
        Tensor_(2, 3, dtype="float32"),
        Tensor_(3, dtype="float32"),
    )["tensor"]
    assert matmul_mat_vec_shape.size == (2,)
    matmul_dot_shape = MatMul(["a", "b"], ["out"], dtype="float32").forward_(
        Tensor_(3, dtype="float32"),
        Tensor_(3, dtype="float32"),
    )["tensor"]
    assert matmul_dot_shape.size == ()
    matmul_uint32 = MatMul(["a", "b"], ["out"], dtype="uint32").forward(
        Tensor(2, 2, dtype="uint32", data=np.array([[1, 2], [3, 4]], dtype=np.uint32)),
        Tensor(2, 2, dtype="uint32", data=np.array([[5, 6], [7, 8]], dtype=np.uint32)),
    )["tensor"]
    assert matmul_uint32.dtype == "uint32"
    np.testing.assert_array_equal(matmul_uint32.data, np.array([[19, 22], [43, 50]], dtype=np.uint32))

    mod_uint32 = Mod(["a", "b"], ["out"], dtype="uint32").forward(
        Tensor(4, dtype="uint32", data=np.array([5, 6, 7, 8], dtype=np.uint32)),
        Tensor(1, dtype="uint32", data=np.array([3], dtype=np.uint32)),
    )["tensor"]
    assert mod_uint32.dtype == "uint32"
    np.testing.assert_array_equal(mod_uint32.data, np.array([2, 0, 1, 2], dtype=np.uint32))

    identity_strings = Identity(["x"], ["y"], dtype="string").forward(
        Tensor(2, dtype="string", data=np.array(["same", "value"], dtype=np.str_))
    )["tensor"]
    assert identity_strings.dtype == "string"
    np.testing.assert_array_equal(identity_strings.data, np.array(["same", "value"], dtype=np.str_))
    size_of_strings = Size(["x"], ["size"]).forward(
        Tensor(2, 3, dtype="string", data=np.array([["a", "b", "c"], ["d", "e", "f"]], dtype=np.str_))
    )["tensor"]
    assert size_of_strings.size == ()
    np.testing.assert_array_equal(size_of_strings.data, np.array(6, dtype=np.int64))
    numeric_size = Size(["x"], ["size"]).forward(
        Tensor(2, 3, dtype="float32", data=np.zeros((2, 3), dtype=np.float32))
    )["tensor"]
    assert numeric_size.size == ()
    np.testing.assert_array_equal(numeric_size.data, np.array(6, dtype=np.int64))
    assert Size(["x"], ["size"]).forward_(Tensor_(2, 3, dtype="float32"))["tensor"].size == ()

    cos_model_path = tmp_path / "constant_of_shape_uint32_scalar.onnx"
    cos_graph = helper.make_graph(
        [helper.make_node("ConstantOfShape", ["shape_empty"], ["y"], value=helper.make_tensor("v", TensorProto.UINT32, [1], [9]))],
        "constant_of_shape_uint32_scalar",
        [],
        [helper.make_tensor_value_info("y", TensorProto.UINT32, [])],
        initializer=[helper.make_tensor("shape_empty", TensorProto.INT64, [0], [])],
    )
    onnx.save(helper.make_model(cos_graph, opset_imports=[helper.make_opsetid("", 17)]), cos_model_path)
    imported_cos = [op for op in ONNXImport(str(cos_model_path), strict=True) if isinstance(op, ConstantOfShape)]
    assert imported_cos[0].dtype == "uint32"

def test_conv_supports_onnx17_auto_pad_and_kernel_shape(tmp_path, monkeypatch):
    _disable_c_backend(monkeypatch)

    x_data = np.arange(1, 26, dtype=np.float32).reshape(1, 1, 5, 5)
    w_data = np.ones((1, 1, 3, 3), dtype=np.float32)
    x = Tensor(*x_data.shape, dtype="float32", data=x_data)
    w = Tensor(*w_data.shape, dtype="float32", data=w_data)

    conv = Conv(
        ["x", "w"], ["y"], pads=None, strides=[2, 2], dilations=[1, 1], group=1,
        kernel_shape=[3, 3], auto_pad="SAME_UPPER", dtype="float32"
    )
    out = conv.forward(x, w)["tensor"]

    padded = np.pad(x_data, [(0, 0), (0, 0), (1, 1), (1, 1)], mode="constant")
    expected = np.empty((1, 1, 3, 3), dtype=np.float32)
    for oy in range(3):
        for ox in range(3):
            expected[0, 0, oy, ox] = np.sum(padded[0, 0, oy * 2:oy * 2 + 3, ox * 2:ox * 2 + 3])
    np.testing.assert_array_equal(out.data, expected)
    assert conv.forward_(Tensor_(1, 1, 5, 5, dtype="float32"), Tensor_(1, 1, 3, 3, dtype="float32"))["tensor"].size == (1, 1, 3, 3)

    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"], strides=[2, 2], kernel_shape=[3, 3], auto_pad="SAME_UPPER")],
        "conv_auto_pad",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 5, 5]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 1, 3, 3]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3, 3])],
    )
    model_path = tmp_path / "conv_auto_pad.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), model_path)
    imported = [op for op in ONNXImport(str(model_path), strict=True) if isinstance(op, Conv)]
    assert imported[0].auto_pad == "SAME_UPPER"
    assert imported[0].kernel_shape == [3, 3]

def test_conv_transpose_auto_pad_matches_onnx_reference(monkeypatch):
    _disable_c_backend(monkeypatch)
    from onnx.reference import ReferenceEvaluator

    x_data = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    w_data = np.ones((1, 1, 2, 2), dtype=np.float32)
    b_data = np.array([1.0], dtype=np.float32)
    x = Tensor(*x_data.shape, dtype="float32", data=x_data)
    w = Tensor(*w_data.shape, dtype="float32", data=w_data)
    bias = Tensor(*b_data.shape, dtype="float32", data=b_data)

    for auto_pad in ("SAME_UPPER", "SAME_LOWER", "VALID"):
        graph = helper.make_graph(
            [
                helper.make_node(
                    "ConvTranspose",
                    ["x", "w", "b"],
                    ["y"],
                    auto_pad=auto_pad,
                    kernel_shape=[2, 2],
                    strides=[1, 1],
                )
            ],
            f"conv_transpose_{auto_pad.lower()}",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, x_data.shape),
                helper.make_tensor_value_info("w", TensorProto.FLOAT, w_data.shape),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, b_data.shape),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        expected = ReferenceEvaluator(model).run(None, {"x": x_data, "w": w_data, "b": b_data})[0]
        actual = ConvTranspose(
            ["x", "w", "b"],
            ["y"],
            pads=None,
            strides=[1, 1],
            dilations=None,
            group=1,
            kernel_shape=[2, 2],
            auto_pad=auto_pad,
            dtype="float32",
        ).forward(x, w, bias)["tensor"].data

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

def test_reshape_and_pad_shape_inference_cover_onnx_edge_cases(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    reshape_shape = Tensor(2, dtype="int64", data=np.array([0, -1], dtype=np.int64))
    reshaped = Reshape(["x", "shape"], ["out"], dtype="float32").forward_(
        Tensor_(2, 3, 4, dtype="float32"), reshape_shape
    )["tensor"]
    assert reshaped.size == (2, 12)

    zero_shape = Tensor(2, dtype="int64", data=np.array([0, 3], dtype=np.int64))
    zero_reshaped = Reshape(["x", "shape"], ["out"], dtype="float32", allowzero=1).forward_(
        Tensor_(0, 3, dtype="float32"), zero_shape
    )["tensor"]
    assert zero_reshaped.size == (0, 3)

    pads = Tensor(4, dtype="int64", data=np.array([1, -1, 0, 2], dtype=np.int64))
    padded = Pad(["x", "pads"], ["out"], dtype="float32").forward_(Tensor_(2, 3, dtype="float32"), pads)["tensor"]
    assert padded.size == (3, 4)

    from onnx.reference import ReferenceEvaluator

    pad_data = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    positive_pads = np.array([1, 2, 1, 0], dtype=np.int64)
    pad_const = np.array(-5.0, dtype=np.float32)
    for mode in ["constant", "edge", "reflect"]:
        node_inputs = ["x", "pads", "value"] if mode == "constant" else ["x", "pads"]
        graph_inputs = [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 3]),
            helper.make_tensor_value_info("pads", TensorProto.INT64, [4]),
        ]
        if mode == "constant":
            graph_inputs.append(helper.make_tensor_value_info("value", TensorProto.FLOAT, []))
        graph = helper.make_graph(
            [helper.make_node("Pad", node_inputs, ["y"], mode=mode)],
            f"pad_{mode}",
            graph_inputs,
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [5, 5])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        feeds = {"x": pad_data, "pads": positive_pads}
        inputs = [
            Tensor(*pad_data.shape, dtype="float32", data=pad_data),
            Tensor(4, dtype="int64", data=positive_pads),
        ]
        if mode == "constant":
            feeds["value"] = pad_const
            inputs.append(Tensor(dtype="float32", data=pad_const))
        expected = ReferenceEvaluator(model).run(None, feeds)[0]
        actual = Pad(["x", "pads"], ["out"], mode=mode, dtype="float32").forward(*inputs)["tensor"]
        np.testing.assert_array_equal(actual.data, expected)

    bool_data = np.array([[True, False]], dtype=np.bool_)
    bool_pads = np.array([0, 1, 0, 1], dtype=np.int64)
    bool_out = Pad(["x", "pads"], ["out"], dtype="bool").forward(
        Tensor(*bool_data.shape, dtype="bool", data=bool_data),
        Tensor(4, dtype="int64", data=bool_pads),
    )["tensor"]
    np.testing.assert_array_equal(bool_out.data, np.pad(bool_data, [(0, 0), (1, 1)], mode="constant", constant_values=False))

    string_data = np.array(["a", "b"], dtype=np.str_)
    string_out = Pad(["x", "pads"], ["out"], dtype="string").forward(
        Tensor(2, dtype="string", data=string_data),
        Tensor(2, dtype="int64", data=np.array([1, 1], dtype=np.int64)),
    )["tensor"]
    np.testing.assert_array_equal(string_out.data, np.array(["", "a", "b", ""], dtype=np.str_))

    model_path = tmp_path / "reshape_allowzero.onnx"
    graph = helper.make_graph(
        [helper.make_node("Reshape", ["x", "shape"], ["y"], allowzero=1)],
        "reshape_allowzero",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [0, 3]),
            helper.make_tensor_value_info("shape", TensorProto.INT64, [2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [0, 3])],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert isinstance(ops[0], Reshape)
    assert ops[0].allowzero == 1

def test_squeeze_and_slice_shape_inference(monkeypatch):
    _disable_c_backend(monkeypatch)

    squeezed = Squeeze(["x"], ["out"], dtype="float32").forward_(Tensor_(1, 3, 1, 4, dtype="float32"))["tensor"]
    assert squeezed.size == (3, 4)
    with pytest.raises(ValueError, match="Cannot squeeze"):
        Squeeze(["x"], ["out"], axes=[1], dtype="float32").forward_(Tensor_(1, 3, dtype="float32"))

    unsqueezed = Unsqueeze(["x"], ["out"], axes=[-1, -3], dtype="float32").forward_(Tensor_(3, 4, dtype="float32"))["tensor"]
    assert unsqueezed.size == (3, 1, 4, 1)
    with pytest.raises(ValueError, match="appears more than once"):
        Unsqueeze(["x"], ["out"], axes=[0, 0], dtype="float32").forward_(Tensor_(3, dtype="float32"))

    starts = Tensor(2, dtype="int64", data=np.array([1, 0], dtype=np.int64))
    ends = Tensor(2, dtype="int64", data=np.array([4, 6], dtype=np.int64))
    axes = Tensor(2, dtype="int64", data=np.array([0, 1], dtype=np.int64))
    steps = Tensor(2, dtype="int64", data=np.array([2, 3], dtype=np.int64))
    sliced = Slice(["x", "starts", "ends", "axes", "steps"], ["out"], dtype="float32").forward_(
        Tensor_(5, 10, dtype="float32"), starts, ends, axes, steps
    )["tensor"]

    assert sliced.size == (2, 2)

def test_shape_copy_ops_support_default_transpose_and_string_payloads(tmp_path):
    text = np.array([["a", "b", "c"], ["d", "e", "f"]], dtype=np.str_)
    text_tensor = Tensor(*text.shape, dtype="string", data=text)

    transposed = Transpose(["x"], ["y"], dtype="string").forward(text_tensor)["tensor"]
    assert transposed.size == (3, 2)
    np.testing.assert_array_equal(transposed.data, text.T)
    assert Transpose(["x"], ["y"], dtype="string").forward_(Tensor_(2, 3, 4, dtype="string"))["tensor"].size == (4, 3, 2)

    flattened = Flatten(["x"], ["y"], axis=0, dtype="string").forward(text_tensor)["tensor"]
    assert flattened.size == (1, 6)
    np.testing.assert_array_equal(flattened.data, text.reshape(1, 6))

    shape = Tensor(2, dtype="int64", data=np.array([3, 2], dtype=np.int64))
    reshaped = Reshape(["x", "shape"], ["y"], dtype="string").forward(text_tensor, shape)["tensor"]
    assert reshaped.size == (3, 2)
    np.testing.assert_array_equal(reshaped.data, text.reshape(3, 2))

    boxed = Tensor(1, 2, 1, dtype="string", data=np.array([[["left"], ["right"]]], dtype=np.str_))
    axes = Tensor(1, dtype="int64", data=np.array([0], dtype=np.int64))
    squeezed = Squeeze(["x", "axes"], ["y"], dtype="string").forward(boxed, axes)["tensor"]
    assert squeezed.size == (2, 1)
    np.testing.assert_array_equal(squeezed.data, boxed.data.reshape(2, 1))

    unsqueeze_axes = Tensor(2, dtype="int64", data=np.array([0, -1], dtype=np.int64))
    unsqueezed = Unsqueeze(["x", "axes"], ["y"], dtype="string").forward(text_tensor, unsqueeze_axes)["tensor"]
    assert unsqueezed.size == (1, 2, 3, 1)
    np.testing.assert_array_equal(unsqueezed.data, text.reshape(1, 2, 3, 1))

    model_path = tmp_path / "transpose_default_perm.onnx"
    graph = helper.make_graph(
        [helper.make_node("Transpose", ["x"], ["y"])],
        "transpose_default_perm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 3, 2])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), model_path)

    imported = [op for op in ONNXImport(str(model_path), strict=True) if isinstance(op, Transpose)]
    assert imported[0].forward_(Tensor_(2, 3, 4, dtype="float32"))["tensor"].size == (4, 3, 2)

def test_indexing_ops_support_string_payloads():
    text = np.array([["a", "b", "c"], ["d", "e", "f"]], dtype=np.str_)
    text_tensor = Tensor(*text.shape, dtype="string", data=text)

    sliced = Slice(["x", "starts", "ends", "axes", "steps"], ["y"], dtype="string").forward(
        text_tensor,
        Tensor(2, dtype="int64", data=np.array([0, 0], dtype=np.int64)),
        Tensor(2, dtype="int64", data=np.array([2, 2], dtype=np.int64)),
        Tensor(2, dtype="int64", data=np.array([0, 1], dtype=np.int64)),
        Tensor(2, dtype="int64", data=np.array([1, 1], dtype=np.int64)),
    )["tensor"]
    np.testing.assert_array_equal(sliced.data, text[:, :2])

    gathered = Gather(["x", "indices"], ["y"], axis=1, dtype="string").forward(
        text_tensor,
        Tensor(2, dtype="int64", data=np.array([2, 0], dtype=np.int64)),
    )["tensor"]
    np.testing.assert_array_equal(gathered.data, text[:, [2, 0]])

    element_indices = np.array([[2, 1, 0], [0, 2, 1]], dtype=np.int64)
    gathered_elements = GatherElements(["x", "indices"], ["y"], axis=1, dtype="string").forward(
        text_tensor,
        Tensor(*element_indices.shape, dtype="int64", data=element_indices),
    )["tensor"]
    np.testing.assert_array_equal(gathered_elements.data, np.take_along_axis(text, element_indices, axis=1))

    nd_indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    gathered_nd = GatherND(["x", "indices"], ["y"], dtype="string").forward(
        text_tensor,
        Tensor(*nd_indices.shape, dtype="int64", data=nd_indices),
    )["tensor"]
    np.testing.assert_array_equal(gathered_nd.data, np.array(["b", "d"], dtype=np.str_))

def test_multi_output_shape_inference_preserves_rank(monkeypatch):
    _disable_c_backend(monkeypatch)

    split = Tensor(3, dtype="int64", data=np.array([2, 3, 1], dtype=np.int64))
    split_out = Split(["x", "split"], ["a", "b", "c"], axis=1, dtype="float32").forward_(Tensor_(4, 6, dtype="float32"), split)["tensor"]
    assert [out.size for out in split_out] == [(4, 2), (4, 3), (4, 1)]

    k = Tensor(1, dtype="int64", data=np.array([5], dtype=np.int64))
    topk_out = TopK(["x", "k"], ["values", "indices"], axis=-1, dtype="float32").forward_(Tensor_(2, 9, dtype="float32"), k)["tensor"]
    assert [out.size for out in topk_out] == [(2, 5), (2, 5)]
    assert topk_out[1].dtype == "int64"

def test_variadic_elementwise_ops_and_einsum_cover_full_onnx_forms(monkeypatch):
    _disable_c_backend(monkeypatch)

    a = Tensor(2, 1, dtype="float32", data=np.array([[1.0], [4.0]], dtype=np.float32))
    b = Tensor(1, 3, dtype="float32", data=np.array([[2.0, 3.0, 0.0]], dtype=np.float32))
    c = Tensor(2, 3, dtype="float32", data=np.array([[0.0, 5.0, 6.0], [7.0, 1.0, 8.0]], dtype=np.float32))

    max_out = Max(["a", "b", "c"], ["out"], dtype="float32").forward(a, b, c)["tensor"]
    np.testing.assert_array_equal(max_out.data, np.maximum(np.maximum(a.data, b.data), c.data))
    assert Max(["a", "b", "c"], ["out"], dtype="float32").forward_(
        Tensor_(2, 1, dtype="float32"), Tensor_(1, 3, dtype="float32"), Tensor_(2, 3, dtype="float32")
    )["tensor"].size == (2, 3)

    min_out = Min(["a", "b", "c"], ["out"], dtype="float32").forward(a, b, c)["tensor"]
    np.testing.assert_array_equal(min_out.data, np.minimum(np.minimum(a.data, b.data), c.data))

    mean_out = Mean(["a", "b", "c"], ["out"], dtype="float32").forward(a, b, c)["tensor"]
    np.testing.assert_allclose(mean_out.data, np.mean(np.stack(np.broadcast_arrays(a.data, b.data, c.data)), axis=0))
    assert Mean(["a", "b", "c"], ["out"], dtype="float32").forward_(
        Tensor_(2, 1, dtype="float32"), Tensor_(1, 3, dtype="float32"), Tensor_(2, 3, dtype="float32")
    )["tensor"].size == (2, 3)

    string_left = Tensor(2, dtype="string", data=np.array(["a", "b"], dtype=np.str_))
    string_right = Tensor(1, dtype="string", data=np.array(["c"], dtype=np.str_))
    string_concat = Concat(["left", "right"], ["out"], axis=0, dtype="string").forward(string_left, string_right)["tensor"]
    np.testing.assert_array_equal(string_concat.data, np.array(["a", "b", "c"], dtype=np.str_))
    assert string_concat.dtype == "string"
    with pytest.raises(ValueError, match="dimension mismatch"):
        Concat(["a", "b"], ["out"], axis=0, dtype="float32").forward_(
            Tensor_(2, 3, dtype="float32"),
            Tensor_(4, 4, dtype="float32"),
        )

    left = Tensor(2, 3, dtype="float32", data=np.arange(6, dtype=np.float32).reshape(2, 3))
    right = Tensor(3, 4, dtype="float32", data=np.arange(12, dtype=np.float32).reshape(3, 4))
    implicit = Einsum(["left", "right"], ["out"], equation="ij,jk", dtype="float32")
    np.testing.assert_array_equal(implicit.forward(left, right)["tensor"].data, np.einsum("ij,jk", left.data, right.data))
    assert implicit.forward_(Tensor_(2, 3, dtype="float32"), Tensor_(3, 4, dtype="float32"))["tensor"].size == (2, 4)
    labels, limits, input_strides, output_strides, out_shape = implicit._parse_equation([(2, 3), (3, 4)])
    assert labels == ["i", "j", "k"]
    assert limits == [2, 3, 4]
    assert input_strides == [3, 1, 0, 0, 4, 1]
    assert output_strides == [4, 0, 1]
    assert out_shape == (2, 4)

    trace = Einsum(["x"], ["out"], equation="ii", dtype="float32")
    labels, limits, input_strides, output_strides, out_shape = trace._parse_equation([(3, 3)])
    assert labels == ["i"]
    assert limits == [3]
    assert input_strides == [4]
    assert output_strides == [0]
    assert out_shape == ()
    with pytest.raises(ValueError, match="inconsistent dimensions"):
        Einsum(["left", "right"], ["out"], equation="ij,kj->ik", dtype="float32")._parse_equation([(2, 3), (4, 5)])
