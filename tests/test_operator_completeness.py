"""文件功能：覆盖 ONNX 算子导入、形状推断、C 后端路径和 Python fallback 的回归测试。
作者：Egor Izmaylov
时间：2026-06-02
"""

import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import nn
from nn import Graph, Ops, Tensor, Tensor_
from nn.ONNXImport import ONNXImport
from nn.Operators import (
    CastLike,
    Clip,
    Constant,
    Concat,
    ConcatFromSequence,
    Compress,
    ConstantOfShape,
    ConvInteger,
    ConvTranspose,
    DequantizeLinear,
    Det,
    DFT,
    Dropout,
    Einsum,
    EyeLike,
    Expand,
    Flatten,
    AveragePool,
    ArgMax,
    ArgMin,
    BatchNormalization,
    BlackmanWindow,
    Cast,
    Conv,
    Gather,
    GatherElements,
    GatherND,
    GlobalAveragePool,
    GlobalLpPool,
    GlobalMaxPool,
    GridSample,
    GRU,
    HammingWindow,
    HannWindow,
    If,
    Identity,
    LSTM,
    LayerNormalization,
    LpNormalization,
    LpPool,
    LRN,
    MatMul,
    MatMulInteger,
    MelWeightMatrix,
    Max,
    MaxPool,
    MaxRoiPool,
    MaxUnpool,
    Mean,
    MeanVarianceNormalization,
    Min,
    Mod,
    Multinomial,
    NegativeLogLikelihoodLoss,
    NonMaxSuppression,
    OneHot,
    Optional,
    OptionalGetElement,
    OptionalHasElement,
    Pad,
    PRelu,
    QLinearConv,
    QLinearMatMul,
    QuantizeLinear,
    Range,
    RandomNormal,
    RandomUniform,
    RandomUniformLike,
    Bernoulli,
    Reshape,
    ReduceSum,
    Resize,
    RoiAlign,
    RNN,
    Loop,
    SequenceAt,
    SequenceConstruct,
    SequenceEmpty,
    SequenceErase,
    SequenceInsert,
    SequenceLength,
    SequenceMap,
    Shape,
    Size,
    Slice,
    Scan,
    Split,
    SplitToSequence,
    SpaceToDepth,
    Squeeze,
    SoftmaxCrossEntropyLoss,
    STFT,
    StringNormalizer,
    Sum,
    TfIdfVectorizer,
    Tile,
    TopK,
    Transpose,
    Trilu,
    NonZero,
    Unsqueeze,
    Unique,
    Where,
)


# 封装 `_disable_c_backend` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
def _disable_c_backend(monkeypatch):
    monkeypatch.setattr(Ops, "_get_lib", classmethod(lambda cls: None))


# 验证 `test_quantize_linear_forward_shape_and_optional_zero_point_import` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_constant_import_supports_onnx17_scalar_and_string_attrs` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_shape_forward_does_not_mutate_end` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_shape_forward_does_not_mutate_end(monkeypatch):
    _disable_c_backend(monkeypatch)
    op = Shape(["x"], ["shape"])

    first = op.forward_(Tensor_(2, 3, 4, dtype="float32"))["tensor"]
    second = op.forward_(Tensor_(5, 6, dtype="float32"))["tensor"]

    assert first.size == (3,)
    assert second.size == (2,)


# 验证 `test_static_shape_inference_uses_constant_inputs` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_conv_supports_onnx17_auto_pad_and_kernel_shape` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_conv_transpose_auto_pad_matches_onnx_reference` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_reshape_and_pad_shape_inference_cover_onnx_edge_cases` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_squeeze_and_slice_shape_inference` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_shape_copy_ops_support_default_transpose_and_string_payloads` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_indexing_ops_support_string_payloads` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_multi_output_shape_inference_preserves_rank` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_multi_output_shape_inference_preserves_rank(monkeypatch):
    _disable_c_backend(monkeypatch)

    split = Tensor(3, dtype="int64", data=np.array([2, 3, 1], dtype=np.int64))
    split_out = Split(["x", "split"], ["a", "b", "c"], axis=1, dtype="float32").forward_(Tensor_(4, 6, dtype="float32"), split)["tensor"]
    assert [out.size for out in split_out] == [(4, 2), (4, 3), (4, 1)]

    k = Tensor(1, dtype="int64", data=np.array([5], dtype=np.int64))
    topk_out = TopK(["x", "k"], ["values", "indices"], axis=-1, dtype="float32").forward_(Tensor_(2, 9, dtype="float32"), k)["tensor"]
    assert [out.size for out in topk_out] == [(2, 5), (2, 5)]
    assert topk_out[1].dtype == "int64"


# 验证 `test_variadic_elementwise_ops_and_einsum_cover_full_onnx_forms` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_global_pooling_and_dropout_optional_mask` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_global_pooling_and_dropout_optional_mask(monkeypatch):
    _disable_c_backend(monkeypatch)

    data = np.arange(2 * 3 * 2 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2, 2)
    x = Tensor(*data.shape, dtype="float32", data=data)

    avg = GlobalAveragePool(["x"], ["out"], dtype="float32").forward(x)["tensor"]
    np.testing.assert_array_equal(avg.data, np.mean(data, axis=(2, 3, 4), keepdims=True))
    assert avg.size == (2, 3, 1, 1, 1)

    max_pool = GlobalMaxPool(["x"], ["out"], dtype="float32").forward(x)["tensor"]
    np.testing.assert_array_equal(max_pool.data, np.max(data, axis=(2, 3, 4), keepdims=True))

    lp = GlobalLpPool(["x"], ["out"], p=2, dtype="float32").forward(x)["tensor"]
    np.testing.assert_allclose(lp.data, np.sum(np.abs(data) ** 2, axis=(2, 3, 4), keepdims=True) ** 0.5)
    assert GlobalLpPool(["x"], ["out"], p=2, dtype="float32").forward_(Tensor_(2, 3, 2, 2, 2, dtype="float32"))["tensor"].size == (2, 3, 1, 1, 1)

    drop_input = Tensor(4, dtype="float32", data=np.ones(4, dtype=np.float32))
    dropout = Dropout(["x"], ["y", "mask"], seed=123, ratio=0.5, training_mode=1)
    y, mask = dropout.forward(drop_input)["tensor"]
    assert y.size == (4,)
    assert mask.size == (4,)
    assert mask.dtype == "bool"
    np.testing.assert_array_equal(y.data, mask.data.astype(np.float32) * 2.0)

    seeded_data = np.arange(6, dtype=np.float32).reshape(2, 3)
    seeded_dropout = Dropout(["x"], ["y", "mask"], seed=0, ratio=0.5, training_mode=1).forward(
        Tensor(*seeded_data.shape, dtype="float32", data=seeded_data)
    )["tensor"]
    np.random.seed(0)
    expected_mask = np.random.uniform(0.0, 1.0, seeded_data.shape) >= 0.5
    np.testing.assert_array_equal(seeded_dropout[1].data, expected_mask)
    np.testing.assert_allclose(seeded_dropout[0].data, seeded_data * expected_mask.astype(np.float32) * 2.0)

    inferred_y, inferred_mask = dropout.forward_(Tensor_(4, dtype="float32"))["tensor"]
    assert inferred_y.size == (4,)
    assert inferred_mask.dtype == "bool"


# 验证 `test_c_backend_pool_mean_and_norm_numeric_paths` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_c_backend_pool_mean_and_norm_numeric_paths():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    big_int = np.array([2**60, 2**60 + 3], dtype=np.int64)
    same_cast = Cast(["x"], ["y"], dtype="int64").forward(Tensor(2, dtype="int64", data=big_int))["tensor"]
    np.testing.assert_array_equal(same_cast.data, big_int)

    float_values = Tensor(4, dtype="float32", data=np.array([1.9, -2.2, 0.0, 127.0], dtype=np.float32))
    int_cast = Cast(["x"], ["y"], dtype="int32").forward(float_values)["tensor"]
    np.testing.assert_array_equal(int_cast.data, float_values.data.astype(np.int32))

    cast_target = Tensor(1, dtype="float64", data=np.array([0.0], dtype=np.float64))
    cast_like = CastLike(["x", "target"], ["y"]).forward(float_values, cast_target)["tensor"]
    assert cast_like.dtype == "float64"
    np.testing.assert_array_equal(cast_like.data, float_values.data.astype(np.float64))

    sum_left = Tensor(2, 1, dtype="float32", data=np.array([[1.0], [-2.0]], dtype=np.float32))
    sum_right = Tensor(1, 3, dtype="float32", data=np.array([[3.0, 4.0, 5.0]], dtype=np.float32))
    sum_bias = Tensor(2, 3, dtype="float32", data=np.ones((2, 3), dtype=np.float32))
    summed = Sum(["left", "right", "bias"], ["out"], dtype="float32").forward(sum_left, sum_right, sum_bias)["tensor"]
    expected_sum = sum_left.data + sum_right.data + sum_bias.data
    np.testing.assert_allclose(summed.data, expected_sum, rtol=1e-6)

    slope = Tensor(1, 3, dtype="float32", data=np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
    prelu = PRelu(["x", "slope"], ["out"], dtype="float32").forward(summed, slope)["tensor"]
    np.testing.assert_allclose(prelu.data, np.where(expected_sum >= 0, expected_sum, expected_sum * slope.data), rtol=1e-6)

    lrn_data = np.arange(1, 1 + 2 * 4 * 2 * 3, dtype=np.float32).reshape(2, 4, 2, 3) / 5.0
    lrn = LRN(["x"], ["y"], size=3, alpha=0.3, beta=0.5, bias=1.0, dtype="float32").forward(
        Tensor(*lrn_data.shape, dtype="float32", data=lrn_data)
    )["tensor"]
    expected_lrn = np.empty_like(lrn_data)
    for n in range(lrn_data.shape[0]):
        for c in range(lrn_data.shape[1]):
            begin, end = max(0, c - 1), min(lrn_data.shape[1], c + 2)
            square_sum = np.sum(lrn_data[n:n + 1, begin:end, ...] ** 2, axis=1)
            expected_lrn[n, c, ...] = lrn_data[n, c, ...] / np.sqrt(1.0 + 0.3 / 3 * square_sum)
    np.testing.assert_allclose(lrn.data, expected_lrn, rtol=1e-6, atol=1e-6)

    mvn_data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    mvn = MeanVarianceNormalization(["x"], ["y"], axes=[0, 2], dtype="float32").forward(
        Tensor(*mvn_data.shape, dtype="float32", data=mvn_data)
    )["tensor"]
    mvn_mean = np.mean(mvn_data, axis=(0, 2), keepdims=True)
    expected_mvn = (mvn_data - mvn_mean) / np.sqrt(np.mean((mvn_data - mvn_mean) ** 2, axis=(0, 2), keepdims=True))
    np.testing.assert_allclose(mvn.data, expected_mvn, rtol=1e-6, atol=1e-6)

    eye = EyeLike(["x"], ["out"], k=-1, dtype="int64").forward(Tensor_(4, 3, dtype="float32"))["tensor"]
    np.testing.assert_array_equal(eye.data, np.eye(4, 3, k=-1, dtype=np.int64))

    data = np.arange(2 * 3 * 2 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2, 2) - 5.0
    x = Tensor(*data.shape, dtype="float32", data=data)

    avg = GlobalAveragePool(["x"], ["out"], dtype="float32").forward(x)["tensor"]
    np.testing.assert_allclose(avg.data, np.mean(data, axis=(2, 3, 4), keepdims=True), rtol=1e-6)
    assert avg.size == (2, 3, 1, 1, 1)

    max_pool = GlobalMaxPool(["x"], ["out"], dtype="float32").forward(x)["tensor"]
    np.testing.assert_allclose(max_pool.data, np.max(data, axis=(2, 3, 4), keepdims=True), rtol=1e-6)

    lp = GlobalLpPool(["x"], ["out"], p=2, dtype="float32").forward(x)["tensor"]
    np.testing.assert_allclose(lp.data, np.sum(np.abs(data) ** 2, axis=(2, 3, 4), keepdims=True) ** 0.5, rtol=1e-6)

    left = Tensor(2, 3, dtype="float32", data=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    right = Tensor(1, 3, dtype="float32", data=np.array([[3, 2, 1]], dtype=np.float32))
    mean = Mean(["left", "right"], ["out"], dtype="float32").forward(left, right)["tensor"]
    expected_mean = np.mean(np.stack(np.broadcast_arrays(left.data, right.data), axis=0), axis=0)
    np.testing.assert_allclose(mean.data, expected_mean, rtol=1e-6)

    bn_data = np.linspace(-1, 1, 12, dtype=np.float32).reshape(2, 3, 2)
    bn_x = Tensor(*bn_data.shape, dtype="float32", data=bn_data)
    bn_scale = Tensor(3, dtype="float32", data=np.array([1.0, 1.5, 0.5], dtype=np.float32))
    bn_bias = Tensor(3, dtype="float32", data=np.array([0.0, 0.1, -0.2], dtype=np.float32))
    bn_mean = Tensor(3, dtype="float32", data=np.array([0.1, -0.2, 0.3], dtype=np.float32))
    bn_var = Tensor(3, dtype="float32", data=np.array([0.8, 1.1, 0.6], dtype=np.float32))
    bn = BatchNormalization(["x", "scale", "bias", "mean", "var"], ["y"], epsilon=1e-5, dtype="float32")
    bn_y = bn.forward(bn_x, bn_scale, bn_bias, bn_mean, bn_var)["tensor"]
    expected_bn = (
        bn_scale.data.reshape(1, 3, 1)
        * (bn_data - bn_mean.data.reshape(1, 3, 1))
        / np.sqrt(bn_var.data.reshape(1, 3, 1) + 1e-5)
        + bn_bias.data.reshape(1, 3, 1)
    )
    np.testing.assert_allclose(bn_y.data, expected_bn, rtol=1e-6, atol=1e-6)

    ln_data = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float32)
    ln_x = Tensor(*ln_data.shape, dtype="float32", data=ln_data)
    ln_scale = Tensor(4, dtype="float32", data=np.array([1.0, 0.5, 1.5, 2.0], dtype=np.float32))
    ln_bias = Tensor(4, dtype="float32", data=np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32))
    ln_y = LayerNormalization(["x", "scale", "bias"], ["y"], axis=-1, epsilon=1e-5, dtype="float32").forward(
        ln_x, ln_scale, ln_bias
    )["tensor"]
    ln_mean = ln_data.mean(axis=1, keepdims=True)
    ln_inv_std = np.reciprocal(np.sqrt(((ln_data - ln_mean) ** 2).mean(axis=1, keepdims=True) + 1e-5))
    expected_ln = (ln_data - ln_mean) * ln_inv_std * ln_scale.data + ln_bias.data
    np.testing.assert_allclose(ln_y.data, expected_ln, rtol=1e-6, atol=1e-6)


# 验证 `test_c_backend_quantized_matmul_paths` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_c_backend_quantized_matmul_paths():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    a_data = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.uint8)
    b_data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int8)
    a = Tensor(*a_data.shape, dtype="uint8", data=a_data)
    b = Tensor(*b_data.shape, dtype="int8", data=b_data)
    a_zp = Tensor(2, dtype="uint8", data=np.array([2, 5], dtype=np.uint8))
    b_zp = Tensor(2, dtype="int8", data=np.array([1, -1], dtype=np.int8))
    matmul_int = MatMulInteger(["a", "b", "azp", "bzp"], ["y"]).forward(a, b, a_zp, b_zp)["tensor"]
    expected_int = np.matmul(
        a_data.astype(np.int32) - a_zp.data.astype(np.int32).reshape(2, 1),
        b_data.astype(np.int32) - b_zp.data.astype(np.int32).reshape(1, 2),
    ).astype(np.int32)
    np.testing.assert_array_equal(matmul_int.data, expected_int)

    batch_a = np.array([[[2, 3, 4], [5, 6, 7]], [[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    batch_b = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.uint8)
    batched = MatMulInteger(["a", "b", "azp", "bzp"], ["y"]).forward(
        Tensor(*batch_a.shape, dtype="uint8", data=batch_a),
        Tensor(*batch_b.shape, dtype="uint8", data=batch_b),
        Tensor(2, 2, 1, dtype="uint8", data=np.array([[[1], [2]], [[0], [1]]], dtype=np.uint8)),
        Tensor(1, 1, 2, dtype="uint8", data=np.array([[[1, 2]]], dtype=np.uint8)),
    )["tensor"]
    expected_batched = np.matmul(
        batch_a.astype(np.int32) - np.array([[[1], [2]], [[0], [1]]], dtype=np.int32),
        batch_b.astype(np.int32) - np.array([[[1, 2]]], dtype=np.int32),
    ).astype(np.int32)
    np.testing.assert_array_equal(batched.data, expected_batched)

    a_scale = Tensor(2, dtype="float32", data=np.array([0.5, 0.25], dtype=np.float32))
    b_scale = Tensor(2, dtype="float32", data=np.array([0.2, 0.4], dtype=np.float32))
    y_scale = Tensor(1, dtype="float32", data=np.array([0.1], dtype=np.float32))
    y_zp = Tensor(1, dtype="uint8", data=np.array([100], dtype=np.uint8))
    qlinear = QLinearMatMul(["a", "as", "azp", "b", "bs", "bzp", "ys", "yzp"], ["y"], dtype="uint8").forward(
        a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
    )["tensor"]
    a_real = (a_data.astype(np.float64) - a_zp.data.reshape(2, 1)) * a_scale.data.reshape(2, 1)
    b_real = (b_data.astype(np.float64) - b_zp.data.reshape(1, 2)) * b_scale.data.reshape(1, 2)
    expected_q = np.rint(np.matmul(a_real, b_real) / y_scale.data.item() + y_zp.data.item())
    expected_q = np.clip(expected_q, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(qlinear.data, expected_q)


# 验证 `test_c_backend_conv_integer_path` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_c_backend_conv_integer_path():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x_data = np.arange(1, 1 + 1 * 2 * 4 * 4, dtype=np.uint8).reshape(1, 2, 4, 4)
    w_data = np.array(
        [
            [[[1, 0], [0, 1]], [[1, 1], [0, 0]]],
            [[[0, 1], [1, 0]], [[1, 0], [1, 0]]],
        ],
        dtype=np.int8,
    )
    x = Tensor(*x_data.shape, dtype="uint8", data=x_data)
    w = Tensor(*w_data.shape, dtype="int8", data=w_data)
    x_zp = Tensor(1, dtype="uint8", data=np.array([2], dtype=np.uint8))
    w_zp = Tensor(2, dtype="int8", data=np.array([0, 1], dtype=np.int8))
    conv_int = ConvInteger(
        ["x", "w", "xz", "wz"], ["y"], pads=[1, 1, 1, 1], strides=[2, 2], dilations=[1, 1], group=1
    ).forward(x, w, x_zp, w_zp)["tensor"]
    expected = np.zeros((1, 2, 3, 3), dtype=np.int32)
    x_centered = x_data.astype(np.int32) - 2
    w_centered = w_data.astype(np.int32) - w_zp.data.astype(np.int32).reshape(2, 1, 1, 1)
    x_padded = np.pad(x_centered, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    for oc in range(2):
        for oh in range(3):
            for ow in range(3):
                patch = x_padded[0, :, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2]
                expected[0, oc, oh, ow] = np.sum(patch * w_centered[oc])
    np.testing.assert_array_equal(conv_int.data, expected)

    grouped_x = Tensor(1, 2, 3, 3, dtype="uint8", data=np.arange(1, 19, dtype=np.uint8).reshape(1, 2, 3, 3))
    grouped_w_data = np.array([[[[1, 0], [0, 1]]], [[[1, 1], [1, 0]]]], dtype=np.uint8)
    grouped = ConvInteger(["x", "w"], ["y"], pads=[0, 0, 0, 0], strides=[1, 1], group=2).forward(
        grouped_x, Tensor(*grouped_w_data.shape, dtype="uint8", data=grouped_w_data)
    )["tensor"]
    expected_grouped = np.empty((1, 2, 2, 2), dtype=np.int32)
    for oc in range(2):
        for oh in range(2):
            for ow in range(2):
                patch = grouped_x.data[0, oc:oc + 1, oh:oh + 2, ow:ow + 2].astype(np.int32)
                expected_grouped[0, oc, oh, ow] = np.sum(patch * grouped_w_data[oc].astype(np.int32))
    np.testing.assert_array_equal(grouped.data, expected_grouped)


# 验证 `test_c_backend_conv_transpose_path` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_c_backend_conv_transpose_path():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x_data = np.arange(1, 1 + 1 * 2 * 2 * 3, dtype=np.float32).reshape(1, 2, 2, 3)
    w_data = np.array(
        [
            [[[1.0, -1.0], [0.5, 2.0]], [[0.0, 1.0], [1.5, -0.5]]],
            [[[-0.5, 1.0], [2.0, 0.0]], [[1.0, 0.5], [-1.0, 1.0]]],
        ],
        dtype=np.float32,
    )
    b_data = np.array([0.25, -0.75], dtype=np.float32)
    x = Tensor(*x_data.shape, dtype="float32", data=x_data)
    w = Tensor(*w_data.shape, dtype="float32", data=w_data)
    b = Tensor(*b_data.shape, dtype="float32", data=b_data)
    op = ConvTranspose(
        ["x", "w", "b"],
        ["y"],
        pads=[1, 0, 0, 1],
        strides=[2, 1],
        dilations=[1, 1],
        output_padding=[1, 0],
        dtype="float32",
    )
    out = op.forward(x, w, b)["tensor"]

    expected = np.zeros((1, 2, 4, 3), dtype=np.float32)
    for n in range(x_data.shape[0]):
        for ic in range(x_data.shape[1]):
            for ih in range(x_data.shape[2]):
                for iw in range(x_data.shape[3]):
                    for oc in range(w_data.shape[1]):
                        for kh in range(w_data.shape[2]):
                            for kw in range(w_data.shape[3]):
                                oh = ih * 2 + kh - 1
                                ow = iw + kw
                                if 0 <= oh < expected.shape[2] and 0 <= ow < expected.shape[3]:
                                    expected[n, oc, oh, ow] += x_data[n, ic, ih, iw] * w_data[ic, oc, kh, kw]
    expected += b_data.reshape(1, 2, 1, 1)

    np.testing.assert_allclose(out.data, expected, rtol=1e-6, atol=1e-6)


# 验证 `test_c_backend_qlinear_conv_path` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_c_backend_qlinear_conv_path():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x_data = np.arange(1, 1 + 1 * 2 * 4 * 4, dtype=np.uint8).reshape(1, 2, 4, 4)
    w_data = np.array(
        [
            [[[1, 2], [3, 4]], [[2, 1], [0, 3]]],
            [[[4, 1], [2, 0]], [[1, 3], [2, 4]]],
        ],
        dtype=np.uint8,
    )
    x = Tensor(*x_data.shape, dtype="uint8", data=x_data)
    w = Tensor(*w_data.shape, dtype="uint8", data=w_data)
    x_scale = Tensor(1, dtype="float32", data=np.array([0.2], dtype=np.float32))
    x_zp = Tensor(1, dtype="uint8", data=np.array([3], dtype=np.uint8))
    w_scale = Tensor(2, dtype="float32", data=np.array([0.25, 0.5], dtype=np.float32))
    w_zp = Tensor(2, dtype="uint8", data=np.array([1, 2], dtype=np.uint8))
    y_scale = Tensor(1, dtype="float32", data=np.array([0.1], dtype=np.float32))
    y_zp = Tensor(1, dtype="uint8", data=np.array([7], dtype=np.uint8))
    bias = Tensor(2, dtype="int32", data=np.array([3, -4], dtype=np.int32))

    qconv = QLinearConv(
        ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz", "b"],
        ["y"],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
        dtype="uint8",
    ).forward(x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, bias)["tensor"]

    expected = np.zeros((1, 2, 3, 3), dtype=np.uint8)
    x_centered = x_data.astype(np.int32) - int(x_zp.data[0])
    w_centered = w_data.astype(np.int32) - w_zp.data.astype(np.int32).reshape(2, 1, 1, 1)
    x_padded = np.pad(x_centered, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    for oc in range(2):
        for oh in range(3):
            for ow in range(3):
                patch = x_padded[0, :, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2]
                acc = int(np.sum(patch * w_centered[oc])) + int(bias.data[oc])
                scaled = acc * float(x_scale.data[0]) * float(w_scale.data[oc]) / float(y_scale.data[0])
                expected[0, oc, oh, ow] = np.clip(np.rint(scaled + int(y_zp.data[0])), 0, 255).astype(np.uint8)

    np.testing.assert_array_equal(qconv.data, expected)


# 验证 `test_c_backend_max_unpool_path` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_c_backend_max_unpool_path():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    pooled = Tensor(1, 1, 2, 2, dtype="float32", data=np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32))
    indices = Tensor(1, 1, 2, 2, dtype="int64", data=np.array([[[[5, 7], [13, 15]]]], dtype=np.int64))
    output_shape = Tensor(4, dtype="int64", data=np.array([1, 1, 5, 5], dtype=np.int64))

    unpooled = MaxUnpool(["x", "i", "shape"], ["y"], kernel_shape=[2, 2], strides=[2, 2], dtype="float32").forward(
        pooled, indices, output_shape
    )["tensor"]

    expected = np.zeros((1, 1, 5, 5), dtype=np.float32)
    expected.reshape(-1)[[6, 8, 16, 18]] = [1.0, 2.0, 3.0, 4.0]
    np.testing.assert_array_equal(unpooled.data, expected)


# 验证 `test_c_backend_unique_and_mel_weight_matrix_paths` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_c_backend_unique_and_mel_weight_matrix_paths():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    unique_input = Tensor(6, dtype="int64", data=np.array([3, 1, 3, 2, 1, 3], dtype=np.int64))
    y, indices, inverse, counts = Unique(
        ["x"], ["y", "indices", "inverse", "counts"], sorted=0, dtype="int64"
    ).forward(unique_input)["tensor"]
    np.testing.assert_array_equal(y.data, np.array([3, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(indices.data, np.array([0, 1, 3], dtype=np.int64))
    np.testing.assert_array_equal(inverse.data, np.array([0, 1, 0, 2, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(counts.data, np.array([3, 2, 1], dtype=np.int64))

    mel = MelWeightMatrix([], ["mel"], output_datatype=TensorProto.FLOAT).forward(
        Tensor(dtype="int64", data=np.array(3, dtype=np.int64)),
        Tensor(dtype="int64", data=np.array(8, dtype=np.int64)),
        Tensor(dtype="int64", data=np.array(16000, dtype=np.int64)),
        Tensor(dtype="float32", data=np.array(0.0, dtype=np.float32)),
        Tensor(dtype="float32", data=np.array(8000.0, dtype=np.float32)),
    )["tensor"]
    expected_mel = np.array(
        [[1.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(mel.data, expected_mel, rtol=1e-6, atol=1e-6)


# 验证 `test_c_backend_dft_and_stft_paths_against_reference` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_c_backend_dft_and_stft_paths_against_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    from onnx.reference import ReferenceEvaluator

    signal_data = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    dft_len = np.array(4, dtype=np.int64)
    signal = Tensor(*signal_data.shape, dtype="float32", data=signal_data)
    dft = DFT(["x", "dft_len"], ["y"], axis=1, onesided=1, dtype="float32").forward(
        signal, Tensor(dtype="int64", data=dft_len)
    )["tensor"]
    dft_graph = helper.make_graph(
        [helper.make_node("DFT", ["x", "dft_len"], ["y"], axis=1, onesided=1)],
        "dft_reference",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(signal_data.shape)),
            helper.make_tensor_value_info("dft_len", TensorProto.INT64, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 2])],
    )
    dft_expected = ReferenceEvaluator(helper.make_model(dft_graph, opset_imports=[helper.make_opsetid("", 17)])).run(
        None, {"x": signal_data, "dft_len": dft_len}
    )[0]
    np.testing.assert_allclose(dft.data, dft_expected, rtol=1e-6, atol=1e-6)

    inverse = DFT(["x", "dft_len"], ["y"], axis=1, inverse=1, onesided=1, dtype="float32").forward(
        dft, Tensor(dtype="int64", data=dft_len)
    )["tensor"]
    inv_graph = helper.make_graph(
        [helper.make_node("DFT", ["x", "dft_len"], ["y"], axis=1, inverse=1, onesided=1)],
        "dft_inverse_reference",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(dft.data.shape)),
            helper.make_tensor_value_info("dft_len", TensorProto.INT64, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(signal_data.shape))],
    )
    inv_expected = ReferenceEvaluator(helper.make_model(inv_graph, opset_imports=[helper.make_opsetid("", 17)])).run(
        None, {"x": dft.data, "dft_len": dft_len}
    )[0]
    np.testing.assert_allclose(inverse.data, inv_expected, rtol=1e-6, atol=1e-6)

    frame_step = np.array(2, dtype=np.int64)
    frame_length = np.array(2, dtype=np.int64)
    window_data = np.ones((2,), dtype=np.float32)
    stft = STFT(["x", "step", "window", "length"], ["y"], onesided=1, dtype="float32").forward(
        signal,
        Tensor(dtype="int64", data=frame_step),
        Tensor(*window_data.shape, dtype="float32", data=window_data),
        Tensor(dtype="int64", data=frame_length),
    )["tensor"]
    stft_graph = helper.make_graph(
        [helper.make_node("STFT", ["x", "step", "window", "length"], ["y"], onesided=1)],
        "stft_reference",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(signal_data.shape)),
            helper.make_tensor_value_info("step", TensorProto.INT64, []),
            helper.make_tensor_value_info("window", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("length", TensorProto.INT64, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])],
    )
    stft_expected = ReferenceEvaluator(helper.make_model(stft_graph, opset_imports=[helper.make_opsetid("", 17)])).run(
        None, {"x": signal_data, "step": frame_step, "window": window_data, "length": frame_length}
    )[0]
    np.testing.assert_allclose(stft.data, stft_expected, rtol=1e-6, atol=1e-6)


# 验证 `test_c_backend_recurrent_paths_match_python_semantics` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_c_backend_recurrent_paths_match_python_semantics():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = Tensor(3, 2, 2, dtype="float32", data=np.array(
        [
            [[0.5, -0.2], [0.1, 0.4]],
            [[1.0, 0.3], [-0.3, 0.2]],
            [[0.2, 0.7], [0.8, -0.5]],
        ],
        dtype=np.float32,
    ))
    sequence_lens = Tensor(2, dtype="int32", data=np.array([3, 2], dtype=np.int32))

    rnn_w = Tensor(2, 2, 2, dtype="float32", data=np.array(
        [[[0.1, 0.2], [-0.2, 0.3]], [[-0.3, 0.4], [0.2, 0.1]]], dtype=np.float32
    ))
    rnn_r = Tensor(2, 2, 2, dtype="float32", data=np.array(
        [[[0.5, 0.1], [0.2, 0.4]], [[0.2, -0.1], [0.3, 0.2]]], dtype=np.float32
    ))
    rnn_b = Tensor(2, 4, dtype="float32", data=np.array([[0.1, -0.1, 0.05, 0.02], [0.0, 0.1, -0.03, 0.04]], dtype=np.float32))
    rnn_initial = Tensor(2, 2, 2, dtype="float32", data=np.array(
        [[[0.1, 0.0], [0.0, 0.2]], [[-0.1, 0.1], [0.2, -0.2]]], dtype=np.float32
    ))
    rnn_c = RNN(["x", "w", "r", "b", "seq", "init"], ["y", "yh"], hidden_size=2, direction="bidirectional", dtype="float32")
    rnn_py = RNN(["x", "w", "r", "b", "seq", "init"], ["y", "yh"], hidden_size=2, direction="bidirectional", dtype="float32")
    rnn_py.lib = None
    c_y, c_h = rnn_c.forward(x, rnn_w, rnn_r, rnn_b, sequence_lens, rnn_initial)["tensor"]
    py_y, py_h = rnn_py.forward(x, rnn_w, rnn_r, rnn_b, sequence_lens, rnn_initial)["tensor"]
    np.testing.assert_allclose(c_y.data, py_y.data, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c_h.data, py_h.data, rtol=1e-6, atol=1e-6)

    gru_w = Tensor(1, 6, 2, dtype="float32", data=np.array([[
        [0.2, -0.1], [0.1, 0.3],
        [-0.2, 0.4], [0.3, -0.2],
        [0.4, 0.1], [-0.1, 0.2],
    ]], dtype=np.float32))
    gru_r = Tensor(1, 6, 2, dtype="float32", data=np.array([[
        [0.1, 0.2], [0.2, -0.1],
        [0.3, 0.1], [-0.2, 0.4],
        [0.2, 0.3], [0.1, -0.3],
    ]], dtype=np.float32))
    gru_b = Tensor(1, 12, dtype="float32", data=np.linspace(-0.2, 0.2, 12, dtype=np.float32).reshape(1, 12))
    gru_initial = Tensor(1, 2, 2, dtype="float32", data=np.array([[[0.1, -0.1], [0.2, 0.0]]], dtype=np.float32))
    gru_c = GRU(["x", "w", "r", "b", "seq", "init"], ["y", "yh"], hidden_size=2, linear_before_reset=1, dtype="float32")
    gru_py = GRU(["x", "w", "r", "b", "seq", "init"], ["y", "yh"], hidden_size=2, linear_before_reset=1, dtype="float32")
    gru_py.lib = None
    c_y, c_h = gru_c.forward(x, gru_w, gru_r, gru_b, sequence_lens, gru_initial)["tensor"]
    py_y, py_h = gru_py.forward(x, gru_w, gru_r, gru_b, sequence_lens, gru_initial)["tensor"]
    np.testing.assert_allclose(c_y.data, py_y.data, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c_h.data, py_h.data, rtol=1e-6, atol=1e-6)

    lstm_w = Tensor(1, 8, 2, dtype="float32", data=np.linspace(-0.3, 0.4, 16, dtype=np.float32).reshape(1, 8, 2))
    lstm_r = Tensor(1, 8, 2, dtype="float32", data=np.linspace(0.2, -0.2, 16, dtype=np.float32).reshape(1, 8, 2))
    lstm_b = Tensor(1, 16, dtype="float32", data=np.linspace(-0.1, 0.1, 16, dtype=np.float32).reshape(1, 16))
    lstm_initial_h = Tensor(1, 2, 2, dtype="float32", data=np.array([[[0.1, 0.0], [-0.1, 0.2]]], dtype=np.float32))
    lstm_initial_c = Tensor(1, 2, 2, dtype="float32", data=np.array([[[0.0, 0.2], [0.1, -0.1]]], dtype=np.float32))
    peepholes = Tensor(1, 6, dtype="float32", data=np.linspace(-0.05, 0.05, 6, dtype=np.float32).reshape(1, 6))
    lstm_c = LSTM(["x", "w", "r", "b", "seq", "h", "c", "p"], ["y", "yh", "yc"], hidden_size=2, input_forget=1, dtype="float32")
    lstm_py = LSTM(["x", "w", "r", "b", "seq", "h", "c", "p"], ["y", "yh", "yc"], hidden_size=2, input_forget=1, dtype="float32")
    lstm_py.lib = None
    c_y, c_h, c_c = lstm_c.forward(x, lstm_w, lstm_r, lstm_b, sequence_lens, lstm_initial_h, lstm_initial_c, peepholes)["tensor"]
    py_y, py_h, py_c = lstm_py.forward(x, lstm_w, lstm_r, lstm_b, sequence_lens, lstm_initial_h, lstm_initial_c, peepholes)["tensor"]
    np.testing.assert_allclose(c_y.data, py_y.data, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c_h.data, py_h.data, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c_c.data, py_c.data, rtol=1e-6, atol=1e-6)


# 验证 `test_c_backend_probability_and_loss_paths` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_c_backend_probability_and_loss_paths():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    probabilities = Tensor(
        2, 3, dtype="float32",
        data=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )
    samples = Multinomial(["p"], ["y"], dtype=TensorProto.INT64, sample_size=4, seed=7.0).forward(probabilities)["tensor"]
    np.testing.assert_array_equal(samples.data, np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.int64))

    log_probs = Tensor(
        2, 3, 2, dtype="float32",
        data=np.array(
            [[[-0.1, -0.2], [-1.0, -1.1], [-2.0, -2.1]],
             [[-0.3, -0.4], [-1.2, -1.3], [-2.2, -2.3]]],
            dtype=np.float32,
        ),
    )
    labels = Tensor(2, 2, dtype="int64", data=np.array([[0, 2], [1, -1]], dtype=np.int64))
    weights = Tensor(3, dtype="float32", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    nll = NegativeLogLikelihoodLoss(
        ["x", "target", "w"], ["loss"], reduction="mean", ignore_index=-1, dtype="float32"
    ).forward(log_probs, labels, weights)["tensor"]
    expected_weighted = np.array([[0.1, 2.1 * 3.0], [1.2 * 2.0, 0.0]], dtype=np.float32)
    expected_denom = np.array([[1.0, 3.0], [2.0, 0.0]], dtype=np.float32).sum()
    np.testing.assert_allclose(nll.data, expected_weighted.sum() / expected_denom, rtol=1e-6)

    scores = Tensor(2, 3, dtype="float32", data=np.array([[1.0, 2.0, 4.0], [0.5, 0.0, -1.0]], dtype=np.float32))
    labels_1d = Tensor(2, dtype="int64", data=np.array([2, 0], dtype=np.int64))
    sce_loss, log_prob = SoftmaxCrossEntropyLoss(
        ["scores", "labels"], ["loss", "log_prob"], reduction="none", dtype="float32"
    ).forward(scores, labels_1d)["tensor"]
    shifted = scores.data - np.max(scores.data, axis=1, keepdims=True)
    expected_log_prob = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    np.testing.assert_allclose(log_prob.data, expected_log_prob, rtol=1e-6)
    np.testing.assert_allclose(sce_loss.data, -expected_log_prob[np.arange(2), labels_1d.data], rtol=1e-6)


# 验证 `test_c_backend_non_max_suppression_path` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_c_backend_non_max_suppression_path():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    boxes = Tensor(
        1, 3, 4, dtype="float32",
        data=np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1], [0.0, 10.0, 1.0, 11.0]]], dtype=np.float32),
    )
    scores = Tensor(1, 1, 3, dtype="float32", data=np.array([[[0.9, 0.8, 0.7]]], dtype=np.float32))
    selected = NonMaxSuppression(["boxes", "scores", "max", "iou"], ["selected"]).forward(
        boxes,
        scores,
        Tensor(1, dtype="int64", data=np.array([2], dtype=np.int64)),
        Tensor(1, dtype="float32", data=np.array([0.5], dtype=np.float32)),
    )["tensor"]
    np.testing.assert_array_equal(selected.data, np.array([[0, 0, 0], [0, 0, 2]], dtype=np.int64))

    center_boxes = Tensor(
        1, 3, 4, dtype="float32",
        data=np.array([[[0.5, 0.5, 1.0, 1.0], [0.55, 0.5, 1.0, 1.0], [10.5, 0.5, 1.0, 1.0]]], dtype=np.float32),
    )
    center_selected = NonMaxSuppression(
        ["boxes", "scores", "max", "iou"], ["selected"], center_point_box=1
    ).forward(
        center_boxes,
        scores,
        Tensor(1, dtype="int64", data=np.array([2], dtype=np.int64)),
        Tensor(1, dtype="float32", data=np.array([0.5], dtype=np.float32)),
    )["tensor"]
    np.testing.assert_array_equal(center_selected.data, np.array([[0, 0, 0], [0, 0, 2]], dtype=np.int64))


# 验证 `test_c_backend_grid_sample_matches_onnx_reference` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_c_backend_grid_sample_matches_onnx_reference():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")
    from onnx.reference import ReferenceEvaluator

    x_data = np.arange(1, 1 + 1 * 1 * 3 * 3, dtype=np.float32).reshape(1, 1, 3, 3)
    grid_data = np.array(
        [[[[-1.0, -1.0], [0.0, 0.0], [1.2, 1.2]], [[0.5, -0.5], [-1.5, 0.5], [0.2, 1.5]]]],
        dtype=np.float32,
    )
    for mode, padding_mode, align_corners in [
        ("linear", "zeros", 0),
        ("nearest", "border", 0),
        ("cubic", "reflection", 1),
    ]:
        graph = helper.make_graph(
            [helper.make_node("GridSample", ["x", "grid"], ["y"], mode=mode, padding_mode=padding_mode, align_corners=align_corners)],
            "grid_sample_ref",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, x_data.shape),
                helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_data.shape),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        )
        expected = ReferenceEvaluator(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])).run(
            None, {"x": x_data, "grid": grid_data}
        )[0]
        actual = GridSample(
            ["x", "grid"], ["y"], mode=mode, padding_mode=padding_mode, align_corners=align_corners, dtype="float32"
        ).forward(
            Tensor(*x_data.shape, dtype="float32", data=x_data),
            Tensor(*grid_data.shape, dtype="float32", data=grid_data),
        )["tensor"].data
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    alias = GridSample(["x", "grid"], ["y"], mode="bilinear", dtype="float32").forward(
        Tensor(*x_data.shape, dtype="float32", data=x_data),
        Tensor(*grid_data.shape, dtype="float32", data=grid_data),
    )["tensor"].data
    linear = GridSample(["x", "grid"], ["y"], mode="linear", dtype="float32").forward(
        Tensor(*x_data.shape, dtype="float32", data=x_data),
        Tensor(*grid_data.shape, dtype="float32", data=grid_data),
    )["tensor"].data
    np.testing.assert_allclose(alias, linear, rtol=1e-6, atol=1e-6)


# 验证 `test_pooling_supports_nd_shapes_and_optional_indices` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_arg_ops_keepdims_and_select_last_match_onnx_reference` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_space_to_depth_and_lp_normalization_match_onnx_reference` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_lp_normalization_l1_preserves_input_sign_in_c_and_python` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_onehot_python_fallback_keeps_out_of_range_negative_indices_off` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_compress_short_condition_matches_onnx_reference` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_split_uneven_without_split_input_matches_onnx_reference` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_cast_supports_string_conversions_like_onnx_reference` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_additional_onnx17_official_ops` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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


# 验证 `test_independent_onnx17_gap_ops` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_independent_onnx17_gap_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    det_input = Tensor(2, 2, 2, dtype="float32", data=np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 0.0], [0.0, 5.0]]], dtype=np.float32
    ))
    det = Det(["x"], ["y"], dtype="float32").forward(det_input)["tensor"]
    np.testing.assert_allclose(det.data, np.linalg.det(det_input.data).astype(np.float32))
    assert Det(["x"], ["y"], dtype="float32").forward_(Tensor_(2, 2, 2, dtype="float32"))["tensor"].size == (2,)

    unique_input = Tensor(6, dtype="int64", data=np.array([3, 1, 3, 2, 1, 3], dtype=np.int64))
    unique_y, unique_idx, unique_inv, unique_counts = Unique(
        ["x"], ["y", "indices", "inverse", "counts"], sorted=0, dtype="int64"
    ).forward(unique_input)["tensor"]
    np.testing.assert_array_equal(unique_y.data, np.array([3, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(unique_idx.data, np.array([0, 1, 3], dtype=np.int64))
    np.testing.assert_array_equal(unique_inv.data, np.array([0, 1, 0, 2, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(unique_counts.data, np.array([3, 2, 1], dtype=np.int64))

    lrn_data = np.arange(1, 1 + 1 * 4 * 1 * 1, dtype=np.float32).reshape(1, 4, 1, 1)
    lrn = LRN(["x"], ["y"], size=3, alpha=0.3, beta=0.5, bias=1.0, dtype="float32").forward(
        Tensor(*lrn_data.shape, dtype="float32", data=lrn_data)
    )["tensor"]
    expected_lrn = np.empty_like(lrn_data)
    for c in range(4):
        begin, end = max(0, c - 1), min(4, c + 2)
        square_sum = np.sum(lrn_data[:, begin:end, ...] ** 2, axis=1)
        expected_lrn[:, c, ...] = lrn_data[:, c, ...] / np.sqrt(1.0 + 0.3 / 3 * square_sum)
    np.testing.assert_allclose(lrn.data, expected_lrn)

    mvn_input = Tensor(2, 2, dtype="float32", data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    mvn = MeanVarianceNormalization(["x"], ["y"], axes=[0], dtype="float32").forward(mvn_input)["tensor"]
    mean = np.mean(mvn_input.data, axis=(0,), keepdims=True)
    expected_mvn = (mvn_input.data - mean) / np.sqrt(np.mean((mvn_input.data - mean) ** 2, axis=(0,), keepdims=True))
    np.testing.assert_allclose(mvn.data, expected_mvn)

    bn_data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    bn_x = Tensor(*bn_data.shape, dtype="float32", data=bn_data)
    bn_scale = Tensor(2, dtype="float32", data=np.array([1.5, 0.5], dtype=np.float32))
    bn_bias = Tensor(2, dtype="float32", data=np.array([0.25, -0.75], dtype=np.float32))
    bn_mean = Tensor(2, dtype="float32", data=np.array([10.0, 20.0], dtype=np.float32))
    bn_var = Tensor(2, dtype="float32", data=np.array([4.0, 9.0], dtype=np.float32))
    bn = BatchNormalization(
        ["x", "scale", "bias", "mean", "var"], ["y", "running_mean", "running_var"],
        epsilon=1e-5, momentum=0.8, training_mode=1, dtype="float32"
    )
    bn_y, bn_running_mean, bn_running_var = bn.forward(bn_x, bn_scale, bn_bias, bn_mean, bn_var)["tensor"]
    saved_mean = np.mean(bn_data, axis=(0, 2))
    saved_var = np.var(bn_data, axis=(0, 2))
    expected_bn = (
        bn_scale.data.reshape(1, 2, 1)
        * (bn_data - saved_mean.reshape(1, 2, 1))
        / np.sqrt(saved_var.reshape(1, 2, 1) + 1e-5)
        + bn_bias.data.reshape(1, 2, 1)
    )
    np.testing.assert_allclose(bn_y.data, expected_bn, rtol=1e-6)
    np.testing.assert_allclose(bn_running_mean.data, bn_mean.data * 0.8 + saved_mean * 0.2)
    np.testing.assert_allclose(bn_running_var.data, bn_var.data * 0.8 + saved_var * 0.2)
    inferred_bn = bn.forward_(
        Tensor_(2, 2, 2, dtype="float32"),
        Tensor_(2, dtype="float32"),
        Tensor_(2, dtype="float32"),
        Tensor_(2, dtype="float32"),
        Tensor_(2, dtype="float32"),
    )["tensor"]
    assert [out.size for out in inferred_bn] == [(2, 2, 2), (2,), (2,)]

    bn_model_path = tmp_path / "batch_norm_training.onnx"
    bn_graph = helper.make_graph(
        [helper.make_node(
            "BatchNormalization",
            ["x", "scale", "bias", "mean", "var"],
            ["y", "running_mean", "running_var"],
            epsilon=1e-5,
            momentum=0.8,
            training_mode=1,
        )],
        "batch_norm_training",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2, 2]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("var", TensorProto.FLOAT, [2]),
        ],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2, 2]),
            helper.make_tensor_value_info("running_mean", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("running_var", TensorProto.FLOAT, [2]),
        ],
    )
    onnx.save(helper.make_model(bn_graph, opset_imports=[helper.make_opsetid("", 17)]), bn_model_path)
    bn_imported = [op for op in ONNXImport(str(bn_model_path), strict=True) if isinstance(op, BatchNormalization)]
    assert bn_imported[0].training_mode == 1

    ln_data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    ln_scale = np.linspace(0.5, 1.6, 12, dtype=np.float32).reshape(3, 4)
    ln_bias = np.linspace(-0.3, 0.8, 12, dtype=np.float32).reshape(3, 4)
    ln = LayerNormalization(
        ["x", "scale", "bias"], ["y", "mean", "inv_std"],
        axis=1, epsilon=1e-5, stash_type=1, dtype="float32"
    )
    ln_y, ln_mean, ln_inv_std = ln.forward(
        Tensor(*ln_data.shape, dtype="float32", data=ln_data),
        Tensor(*ln_scale.shape, dtype="float32", data=ln_scale),
        Tensor(*ln_bias.shape, dtype="float32", data=ln_bias),
    )["tensor"]
    ln_mat = ln_data.reshape(2, 12)
    expected_mean = np.mean(ln_mat, axis=1, keepdims=True).reshape(2, 1, 1)
    expected_inv_std = np.reciprocal(np.sqrt(np.mean((ln_mat - expected_mean.reshape(2, 1)) ** 2, axis=1, keepdims=True) + 1e-5)).reshape(2, 1, 1)
    expected_ln = ((ln_data - expected_mean) * expected_inv_std) * ln_scale + ln_bias
    np.testing.assert_allclose(ln_y.data, expected_ln, rtol=1e-6)
    np.testing.assert_allclose(ln_mean.data, expected_mean)
    np.testing.assert_allclose(ln_inv_std.data, expected_inv_std)
    inferred_ln = ln.forward_(
        Tensor_(2, 3, 4, dtype="float32"),
        Tensor_(3, 4, dtype="float32"),
        Tensor_(3, 4, dtype="float32"),
    )["tensor"]
    assert [out.size for out in inferred_ln] == [(2, 3, 4), (2, 1, 1), (2, 1, 1)]

    ln_model_path = tmp_path / "layer_norm_stash.onnx"
    ln_graph = helper.make_graph(
        [helper.make_node("LayerNormalization", ["x", "scale", "bias"], ["y", "mean", "inv_std"], axis=1, epsilon=1e-5, stash_type=1)],
        "layer_norm_stash",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3, 4]),
        ],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4]),
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [2, 1, 1]),
            helper.make_tensor_value_info("inv_std", TensorProto.FLOAT, [2, 1, 1]),
        ],
    )
    onnx.save(helper.make_model(ln_graph, opset_imports=[helper.make_opsetid("", 17)]), ln_model_path)
    ln_imported = [op for op in ONNXImport(str(ln_model_path), strict=True) if isinstance(op, LayerNormalization)]
    assert ln_imported[0].stash_type == 1

    a = Tensor(2, 3, dtype="uint8", data=np.array([[2, 3, 4], [5, 6, 7]], dtype=np.uint8))
    b = Tensor(3, 2, dtype="int8", data=np.array([[1, -2], [3, 4], [-1, 2]], dtype=np.int8))
    a_zp = Tensor(1, dtype="uint8", data=np.array([2], dtype=np.uint8))
    b_zp = Tensor(2, dtype="int8", data=np.array([1, -1], dtype=np.int8))
    matmul_int = MatMulInteger(["a", "b", "azp", "bzp"], ["y"]).forward(a, b, a_zp, b_zp)["tensor"]
    expected_int = np.matmul(a.data.astype(np.int32) - 2, b.data.astype(np.int32) - np.array([1, -1], dtype=np.int32))
    np.testing.assert_array_equal(matmul_int.data, expected_int.astype(np.int32))

    qlinear = QLinearMatMul(["a", "as", "azp", "b", "bs", "bzp", "ys", "yzp"], ["y"], dtype="uint8").forward(
        a,
        Tensor(1, dtype="float32", data=np.array([0.5], dtype=np.float32)),
        a_zp,
        Tensor(3, 2, dtype="uint8", data=np.array([[3, 4], [5, 6], [7, 8]], dtype=np.uint8)),
        Tensor(1, dtype="float32", data=np.array([0.25], dtype=np.float32)),
        Tensor(1, dtype="uint8", data=np.array([3], dtype=np.uint8)),
        Tensor(1, dtype="float32", data=np.array([0.5], dtype=np.float32)),
        Tensor(1, dtype="uint8", data=np.array([10], dtype=np.uint8)),
    )["tensor"]
    assert qlinear.dtype == "uint8"
    assert qlinear.size == (2, 2)

    conv_x = Tensor(1, 1, 3, 3, dtype="uint8", data=np.arange(1, 10, dtype=np.uint8).reshape(1, 1, 3, 3))
    conv_w = Tensor(1, 1, 2, 2, dtype="int8", data=np.array([[[[1, 0], [0, 1]]]], dtype=np.int8))
    conv_int = ConvInteger(["x", "w", "xzp", "wzp"], ["y"], pads=[0, 0, 0, 0], strides=[1, 1]).forward(
        conv_x,
        conv_w,
        Tensor(1, dtype="uint8", data=np.array([1], dtype=np.uint8)),
        Tensor(1, dtype="int8", data=np.array([0], dtype=np.int8)),
    )["tensor"]
    expected_conv_int = np.array([[[[4, 6], [10, 12]]]], dtype=np.int32)
    np.testing.assert_array_equal(conv_int.data, expected_conv_int)
    assert ConvInteger(["x", "w"], ["y"], pads=[0, 0, 0, 0], strides=[1, 1]).forward_(
        Tensor_(1, 1, 3, 3, dtype="uint8"), Tensor_(1, 1, 2, 2, dtype="int8")
    )["tensor"].size == (1, 1, 2, 2)

    qconv = QLinearConv(["x", "xs", "xzp", "w", "ws", "wzp", "ys", "yzp"], ["y"], pads=[0, 0, 0, 0], dtype="uint8").forward(
        conv_x,
        Tensor(1, dtype="float32", data=np.array([0.5], dtype=np.float32)),
        Tensor(1, dtype="uint8", data=np.array([0], dtype=np.uint8)),
        Tensor(1, 1, 2, 2, dtype="uint8", data=np.array([[[[1, 0], [0, 1]]]], dtype=np.uint8)),
        Tensor(1, dtype="float32", data=np.array([0.25], dtype=np.float32)),
        Tensor(1, dtype="uint8", data=np.array([0], dtype=np.uint8)),
        Tensor(1, dtype="float32", data=np.array([0.125], dtype=np.float32)),
        Tensor(1, dtype="uint8", data=np.array([0], dtype=np.uint8)),
    )["tensor"]
    np.testing.assert_array_equal(qconv.data, np.array([[[[6, 8], [12, 14]]]], dtype=np.uint8))

    deconv_x = Tensor(1, 1, 2, 2, dtype="float32", data=np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32))
    deconv_w = Tensor(1, 1, 2, 2, dtype="float32", data=np.ones((1, 1, 2, 2), dtype=np.float32))
    deconv = ConvTranspose(["x", "w"], ["y"], strides=[2, 2], pads=[0, 0, 0, 0], dtype="float32").forward(
        deconv_x, deconv_w
    )["tensor"]
    np.testing.assert_array_equal(
        deconv.data,
        np.array([[[[1.0, 1.0, 2.0, 2.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [3.0, 3.0, 4.0, 4.0],
                    [3.0, 3.0, 4.0, 4.0]]]], dtype=np.float32),
    )
    assert ConvTranspose(["x", "w"], ["y"], strides=[2, 2], pads=[0, 0, 0, 0]).forward_(
        Tensor_(1, 1, 2, 2, dtype="float32"), Tensor_(1, 1, 2, 2, dtype="float32")
    )["tensor"].size == (1, 1, 4, 4)

    boxes = Tensor(1, 3, 4, dtype="float32", data=np.array([[[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, 10, 1, 11]]], dtype=np.float32))
    scores = Tensor(1, 1, 3, dtype="float32", data=np.array([[[0.9, 0.8, 0.7]]], dtype=np.float32))
    nms = NonMaxSuppression(["boxes", "scores", "max", "iou"], ["selected"]).forward(
        boxes,
        scores,
        Tensor(1, dtype="int64", data=np.array([2], dtype=np.int64)),
        Tensor(1, dtype="float32", data=np.array([0.5], dtype=np.float32)),
    )["tensor"]
    np.testing.assert_array_equal(nms.data, np.array([[0, 0, 0], [0, 0, 2]], dtype=np.int64))

    model_path = tmp_path / "onnx17_independent_gap_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("Det", ["det_x"], ["det_y"]),
            helper.make_node("LRN", ["lrn_x"], ["lrn_y"], size=3),
            helper.make_node("MeanVarianceNormalization", ["mvn_x"], ["mvn_y"], axes=[0]),
            helper.make_node("MatMulInteger", ["mma", "mmb", "mma_zp", "mmb_zp"], ["mmi_y"]),
            helper.make_node("QLinearMatMul", ["qa", "qa_s", "qa_zp", "qb", "qb_s", "qb_zp", "qy_s", "qy_zp"], ["qmm_y"]),
            helper.make_node("ConvTranspose", ["deconv_x", "deconv_w"], ["deconv_y"], strides=[2, 2], pads=[0, 0, 0, 0]),
            helper.make_node("ConvInteger", ["conv_x", "conv_w", "conv_x_zp", "conv_w_zp"], ["conv_int_y"], pads=[0, 0, 0, 0], strides=[1, 1]),
            helper.make_node("QLinearConv", ["qconv_x", "qconv_x_s", "qconv_x_zp", "qconv_w", "qconv_w_s", "qconv_w_zp", "qconv_y_s", "qconv_y_zp"], ["qconv_y"], pads=[0, 0, 0, 0]),
            helper.make_node("NonMaxSuppression", ["boxes", "scores", "max_boxes", "iou"], ["selected"], center_point_box=0),
        ],
        "onnx17_independent_gap_ops",
        [
            helper.make_tensor_value_info("det_x", TensorProto.FLOAT, [2, 2, 2]),
            helper.make_tensor_value_info("lrn_x", TensorProto.FLOAT, [1, 4, 1, 1]),
            helper.make_tensor_value_info("mvn_x", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("mma", TensorProto.UINT8, [2, 3]),
            helper.make_tensor_value_info("mmb", TensorProto.INT8, [3, 2]),
            helper.make_tensor_value_info("mma_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("mmb_zp", TensorProto.INT8, [2]),
            helper.make_tensor_value_info("qa", TensorProto.UINT8, [2, 3]),
            helper.make_tensor_value_info("qa_s", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("qa_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("qb", TensorProto.UINT8, [3, 2]),
            helper.make_tensor_value_info("qb_s", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("qb_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("qy_s", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("qy_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("deconv_x", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("deconv_w", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("conv_x", TensorProto.UINT8, [1, 1, 3, 3]),
            helper.make_tensor_value_info("conv_w", TensorProto.INT8, [1, 1, 2, 2]),
            helper.make_tensor_value_info("conv_x_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("conv_w_zp", TensorProto.INT8, [1]),
            helper.make_tensor_value_info("qconv_x", TensorProto.UINT8, [1, 1, 3, 3]),
            helper.make_tensor_value_info("qconv_x_s", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("qconv_x_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("qconv_w", TensorProto.UINT8, [1, 1, 2, 2]),
            helper.make_tensor_value_info("qconv_w_s", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("qconv_w_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("qconv_y_s", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("qconv_y_zp", TensorProto.UINT8, [1]),
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 3, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 3]),
            helper.make_tensor_value_info("max_boxes", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("iou", TensorProto.FLOAT, [1]),
        ],
        [
            helper.make_tensor_value_info("det_y", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("lrn_y", TensorProto.FLOAT, [1, 4, 1, 1]),
            helper.make_tensor_value_info("mvn_y", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("mmi_y", TensorProto.INT32, [2, 2]),
            helper.make_tensor_value_info("qmm_y", TensorProto.UINT8, [2, 2]),
            helper.make_tensor_value_info("deconv_y", TensorProto.FLOAT, [1, 1, 4, 4]),
            helper.make_tensor_value_info("conv_int_y", TensorProto.INT32, [1, 1, 2, 2]),
            helper.make_tensor_value_info("qconv_y", TensorProto.UINT8, [1, 1, 2, 2]),
            helper.make_tensor_value_info("selected", TensorProto.INT64, [2, 3]),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == [
        "Det", "LRN", "MeanVarianceNormalization", "MatMulInteger", "QLinearMatMul", "ConvTranspose",
        "ConvInteger", "QLinearConv", "NonMaxSuppression"
    ]


# 验证 `test_onnx17_probability_loss_and_spectral_ops` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_onnx17_probability_loss_and_spectral_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    probabilities = Tensor(
        2, 3, dtype="float32",
        data=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )
    samples = Multinomial(["p"], ["y"], dtype=TensorProto.INT64, sample_size=4, seed=7.0).forward(probabilities)["tensor"]
    assert samples.dtype == "int64"
    np.testing.assert_array_equal(samples.data, np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.int64))
    assert Multinomial(["p"], ["y"], sample_size=3).forward_(Tensor_(2, 5, dtype="float32"))["tensor"].size == (2, 3)

    log_probs = Tensor(
        2, 3, 2, dtype="float32",
        data=np.array(
            [[[-0.1, -0.2], [-1.0, -1.1], [-2.0, -2.1]],
             [[-0.3, -0.4], [-1.2, -1.3], [-2.2, -2.3]]],
            dtype=np.float32,
        ),
    )
    labels = Tensor(2, 2, dtype="int64", data=np.array([[0, 2], [1, -1]], dtype=np.int64))
    weights = Tensor(3, dtype="float32", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    nll_none = NegativeLogLikelihoodLoss(["x", "target"], ["loss"], reduction="none", ignore_index=-1, dtype="float32").forward(
        log_probs, labels
    )["tensor"]
    np.testing.assert_allclose(nll_none.data, np.array([[0.1, 2.1], [1.2, 0.0]], dtype=np.float32))
    nll_mean = NegativeLogLikelihoodLoss(["x", "target", "w"], ["loss"], reduction="mean", ignore_index=-1, dtype="float32").forward(
        log_probs, labels, weights
    )["tensor"]
    expected_weighted = (0.1 * 1.0 + 2.1 * 3.0 + 1.2 * 2.0) / (1.0 + 3.0 + 2.0)
    np.testing.assert_allclose(nll_mean.data, np.array(expected_weighted, dtype=np.float32))
    assert NegativeLogLikelihoodLoss(["x", "target"], ["loss"], reduction="none").forward_(
        Tensor_(2, 3, 2, dtype="float32"), Tensor_(2, 2, dtype="int64")
    )["tensor"].size == (2, 2)

    scores = Tensor(2, 3, dtype="float32", data=np.array([[1.0, 2.0, 4.0], [0.5, 0.0, -1.0]], dtype=np.float32))
    labels_1d = Tensor(2, dtype="int64", data=np.array([2, 0], dtype=np.int64))
    sce_loss, log_prob = SoftmaxCrossEntropyLoss(
        ["scores", "labels"], ["loss", "log_prob"], reduction="none", dtype="float32"
    ).forward(scores, labels_1d)["tensor"]
    shifted = scores.data - np.max(scores.data, axis=1, keepdims=True)
    expected_log_prob = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    np.testing.assert_allclose(log_prob.data, expected_log_prob, rtol=1e-6)
    np.testing.assert_allclose(sce_loss.data, -expected_log_prob[np.arange(2), labels_1d.data], rtol=1e-6)

    mel = MelWeightMatrix([], ["mel"], output_datatype=TensorProto.FLOAT).forward(
        Tensor(dtype="int64", data=np.array(3, dtype=np.int64)),
        Tensor(dtype="int64", data=np.array(8, dtype=np.int64)),
        Tensor(dtype="int64", data=np.array(16000, dtype=np.int64)),
        Tensor(dtype="float32", data=np.array(0.0, dtype=np.float32)),
        Tensor(dtype="float32", data=np.array(8000.0, dtype=np.float32)),
    )["tensor"]
    assert mel.size == (5, 3)
    assert np.max(mel.data) <= 1.0
    assert np.count_nonzero(mel.data) > 0

    signal = Tensor(1, 4, 1, dtype="float32", data=np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32))
    dft = DFT(["x"], ["y"], axis=1, onesided=1, dtype="float32").forward(
        signal, Tensor(dtype="int64", data=np.array(4, dtype=np.int64))
    )["tensor"]
    expected_fft = np.fft.fft(signal.data.squeeze(-1), n=4, axis=1)[:, :3]
    expected_dft = np.stack([expected_fft.real, expected_fft.imag], axis=-1).astype(np.float32)
    np.testing.assert_allclose(dft.data, expected_dft, rtol=1e-6, atol=1e-6)
    assert DFT(["x"], ["y"], axis=1, onesided=1).forward_(Tensor_(1, 4, 1, dtype="float32"))["tensor"].size == (1, 3, 2)

    stft = STFT(["x", "step", "window", "length"], ["y"], onesided=1, dtype="float32").forward(
        signal,
        Tensor(dtype="int64", data=np.array(2, dtype=np.int64)),
        Tensor(2, dtype="float32", data=np.ones((2,), dtype=np.float32)),
        Tensor(dtype="int64", data=np.array(2, dtype=np.int64)),
    )["tensor"]
    expected_frames = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    expected_stft_complex = np.fft.fft(expected_frames.squeeze(-1), n=2, axis=-1)[..., :2]
    expected_stft = np.stack([expected_stft_complex.real, expected_stft_complex.imag], axis=-1).astype(np.float32)
    np.testing.assert_allclose(stft.data, expected_stft, rtol=1e-6, atol=1e-6)

    window_size = Tensor(dtype="int64", data=np.array(5, dtype=np.int64))
    assert HannWindow(["size"], ["hann"]).forward_(window_size)["tensor"].size == (5,)
    hamming_shape = HammingWindow(["size"], ["hamming"], output_datatype=TensorProto.DOUBLE).forward_(window_size)["tensor"]
    assert hamming_shape.size == (5,)
    assert hamming_shape.dtype == "float64"
    blackman_shape = BlackmanWindow(["size"], ["blackman"], output_datatype=TensorProto.INT32).forward_(window_size)["tensor"]
    assert blackman_shape.size == (5,)
    assert blackman_shape.dtype == "int32"
    from onnx.reference import ReferenceEvaluator

    for op_name, op_cls in [("HannWindow", HannWindow), ("HammingWindow", HammingWindow), ("BlackmanWindow", BlackmanWindow)]:
        window_graph = helper.make_graph(
            [helper.make_node(op_name, ["size"], ["y"], periodic=0, output_datatype=TensorProto.DOUBLE)],
            f"{op_name}_ref",
            [helper.make_tensor_value_info("size", TensorProto.INT64, [])],
            [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [5])],
        )
        window_model = helper.make_model(window_graph, opset_imports=[helper.make_opsetid("", 17)])
        expected_window = ReferenceEvaluator(window_model).run(None, {"size": np.array(5, dtype=np.int64)})[0]
        actual_window = op_cls(["size"], ["y"], periodic=0, output_datatype=TensorProto.DOUBLE).forward(window_size)["tensor"]
        np.testing.assert_allclose(actual_window.data, expected_window, rtol=1e-12, atol=1e-12)

    uint_window = HannWindow(["size"], ["hann"], output_datatype=TensorProto.UINT32).forward(window_size)["tensor"]
    assert uint_window.dtype == "uint32"
    assert uint_window.data.dtype == np.uint32

    model_path = tmp_path / "onnx17_probability_loss_spectral_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("Multinomial", ["prob"], ["sample"], dtype=TensorProto.INT64, sample_size=2, seed=1.0),
            helper.make_node("NegativeLogLikelihoodLoss", ["log_probs", "labels", "class_weights"], ["nll"], reduction="mean", ignore_index=-1),
            helper.make_node("SoftmaxCrossEntropyLoss", ["scores", "labels_1d"], ["sce", "logp"], reduction="none"),
            helper.make_node("MelWeightMatrix", ["num_mel", "dft_len", "sample_rate", "lower", "upper"], ["mel"]),
            helper.make_node("DFT", ["signal", "dft_len"], ["dft"], axis=1, onesided=1),
            helper.make_node("STFT", ["signal", "frame_step", "window", "frame_length"], ["stft"], onesided=1),
            helper.make_node("HannWindow", ["window_size"], ["hann"], periodic=1),
            helper.make_node("HammingWindow", ["window_size"], ["hamming"], output_datatype=TensorProto.DOUBLE),
            helper.make_node("BlackmanWindow", ["window_size"], ["blackman"], output_datatype=TensorProto.INT32),
        ],
        "onnx17_probability_loss_spectral_ops",
        [
            helper.make_tensor_value_info("prob", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("log_probs", TensorProto.FLOAT, [2, 3, 2]),
            helper.make_tensor_value_info("labels", TensorProto.INT64, [2, 2]),
            helper.make_tensor_value_info("class_weights", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("labels_1d", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("num_mel", TensorProto.INT64, []),
            helper.make_tensor_value_info("dft_len", TensorProto.INT64, []),
            helper.make_tensor_value_info("sample_rate", TensorProto.INT64, []),
            helper.make_tensor_value_info("lower", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("upper", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("signal", TensorProto.FLOAT, [1, 4, 1]),
            helper.make_tensor_value_info("frame_step", TensorProto.INT64, []),
            helper.make_tensor_value_info("window", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("frame_length", TensorProto.INT64, []),
            helper.make_tensor_value_info("window_size", TensorProto.INT64, []),
        ],
        [
            helper.make_tensor_value_info("sample", TensorProto.INT64, [2, 2]),
            helper.make_tensor_value_info("nll", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("sce", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("logp", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("mel", TensorProto.FLOAT, [5, 3]),
            helper.make_tensor_value_info("dft", TensorProto.FLOAT, [1, 3, 2]),
            helper.make_tensor_value_info("stft", TensorProto.FLOAT, [1, 2, 2, 2]),
            helper.make_tensor_value_info("hann", TensorProto.FLOAT, [5]),
            helper.make_tensor_value_info("hamming", TensorProto.DOUBLE, [5]),
            helper.make_tensor_value_info("blackman", TensorProto.INT32, [5]),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == [
        "Multinomial", "NegativeLogLikelihoodLoss", "SoftmaxCrossEntropyLoss", "MelWeightMatrix",
        "DFT", "STFT", "HannWindow", "HammingWindow", "BlackmanWindow"
    ]


# 验证 `test_onnx17_recurrent_ops` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_onnx17_recurrent_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    x = Tensor(2, 1, 1, dtype="float32", data=np.array([[[1.0]], [[2.0]]], dtype=np.float32))
    rnn_w = Tensor(1, 1, 1, dtype="float32", data=np.ones((1, 1, 1), dtype=np.float32))
    rnn_r = Tensor(1, 1, 1, dtype="float32", data=np.ones((1, 1, 1), dtype=np.float32))
    rnn_y, rnn_h = RNN(["x", "w", "r"], ["y", "yh"], hidden_size=1, dtype="float32").forward(x, rnn_w, rnn_r)["tensor"]
    h0 = np.tanh(1.0)
    h1 = np.tanh(2.0 + h0)
    np.testing.assert_allclose(rnn_y.data, np.array([[[[h0]]], [[[h1]]]], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(rnn_h.data, np.array([[[h1]]], dtype=np.float32), rtol=1e-6)
    assert RNN(["x", "w", "r"], ["y", "yh"], hidden_size=1).forward_(
        Tensor_(2, 1, 1, dtype="float32"), Tensor_(1, 1, 1, dtype="float32"), Tensor_(1, 1, 1, dtype="float32")
    )["tensor"][0].size == (2, 1, 1, 1)

    reverse_x_data = np.arange(8, dtype=np.float32).reshape(4, 2, 1) / 10.0
    reverse_x = Tensor(*reverse_x_data.shape, dtype="float32", data=reverse_x_data)
    reverse_w = Tensor(1, 1, 1, dtype="float32", data=np.ones((1, 1, 1), dtype=np.float32))
    reverse_r = Tensor(1, 1, 1, dtype="float32", data=np.full((1, 1, 1), 0.5, dtype=np.float32))
    sequence_lens = Tensor(2, dtype="int64", data=np.array([2, 4], dtype=np.int64))
    reverse_y, reverse_h = RNN(
        ["x", "w", "r", "", "sequence_lens"],
        ["y", "yh"],
        hidden_size=1,
        direction="reverse",
        dtype="float32",
    ).forward(reverse_x, reverse_w, reverse_r, None, sequence_lens)["tensor"]
    expected_reverse = np.zeros((4, 1, 2, 1), dtype=np.float32)
    b0_t1 = np.tanh(reverse_x_data[1, 0, 0])
    b0_t0 = np.tanh(reverse_x_data[0, 0, 0] + 0.5 * b0_t1)
    b1_t3 = np.tanh(reverse_x_data[3, 1, 0])
    b1_t2 = np.tanh(reverse_x_data[2, 1, 0] + 0.5 * b1_t3)
    b1_t1 = np.tanh(reverse_x_data[1, 1, 0] + 0.5 * b1_t2)
    b1_t0 = np.tanh(reverse_x_data[0, 1, 0] + 0.5 * b1_t1)
    expected_reverse[1, 0, 0, 0] = b0_t1
    expected_reverse[0, 0, 0, 0] = b0_t0
    expected_reverse[3, 0, 1, 0] = b1_t3
    expected_reverse[2, 0, 1, 0] = b1_t2
    expected_reverse[1, 0, 1, 0] = b1_t1
    expected_reverse[0, 0, 1, 0] = b1_t0
    np.testing.assert_allclose(reverse_y.data, expected_reverse, rtol=1e-6)
    np.testing.assert_allclose(reverse_h.data, np.array([[[b0_t0], [b1_t0]]], dtype=np.float32), rtol=1e-6)

    one_step = Tensor(1, 1, 1, dtype="float32", data=np.array([[[1.0]]], dtype=np.float32))
    gru_w = Tensor(1, 3, 1, dtype="float32", data=np.array([[[0.0], [0.0], [1.0]]], dtype=np.float32))
    gru_r = Tensor(1, 3, 1, dtype="float32", data=np.zeros((1, 3, 1), dtype=np.float32))
    gru_y, gru_h = GRU(["x", "w", "r"], ["y", "yh"], hidden_size=1, dtype="float32").forward(one_step, gru_w, gru_r)["tensor"]
    expected_gru = 0.5 * np.tanh(1.0)
    np.testing.assert_allclose(gru_y.data, np.array([[[[expected_gru]]]], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(gru_h.data, np.array([[[expected_gru]]], dtype=np.float32), rtol=1e-6)

    lstm_w = Tensor(1, 4, 1, dtype="float32", data=np.array([[[0.0], [0.0], [0.0], [1.0]]], dtype=np.float32))
    lstm_r = Tensor(1, 4, 1, dtype="float32", data=np.zeros((1, 4, 1), dtype=np.float32))
    lstm_y, lstm_h, lstm_c = LSTM(["x", "w", "r"], ["y", "yh", "yc"], hidden_size=1, dtype="float32").forward(
        one_step, lstm_w, lstm_r
    )["tensor"]
    expected_c = 0.5 * np.tanh(1.0)
    expected_h = 0.5 * np.tanh(expected_c)
    np.testing.assert_allclose(lstm_y.data, np.array([[[[expected_h]]]], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(lstm_h.data, np.array([[[expected_h]]], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(lstm_c.data, np.array([[[expected_c]]], dtype=np.float32), rtol=1e-6)

    model_path = tmp_path / "onnx17_recurrent_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("RNN", ["rnn_x", "rnn_w", "rnn_r"], ["rnn_y", "rnn_h"], hidden_size=1),
            helper.make_node("GRU", ["gru_x", "gru_w", "gru_r"], ["gru_y", "gru_h"], hidden_size=1),
            helper.make_node("LSTM", ["lstm_x", "lstm_w", "lstm_r"], ["lstm_y", "lstm_h", "lstm_c"], hidden_size=1),
        ],
        "onnx17_recurrent_ops",
        [
            helper.make_tensor_value_info("rnn_x", TensorProto.FLOAT, [2, 1, 1]),
            helper.make_tensor_value_info("rnn_w", TensorProto.FLOAT, [1, 1, 1]),
            helper.make_tensor_value_info("rnn_r", TensorProto.FLOAT, [1, 1, 1]),
            helper.make_tensor_value_info("gru_x", TensorProto.FLOAT, [1, 1, 1]),
            helper.make_tensor_value_info("gru_w", TensorProto.FLOAT, [1, 3, 1]),
            helper.make_tensor_value_info("gru_r", TensorProto.FLOAT, [1, 3, 1]),
            helper.make_tensor_value_info("lstm_x", TensorProto.FLOAT, [1, 1, 1]),
            helper.make_tensor_value_info("lstm_w", TensorProto.FLOAT, [1, 4, 1]),
            helper.make_tensor_value_info("lstm_r", TensorProto.FLOAT, [1, 4, 1]),
        ],
        [
            helper.make_tensor_value_info("rnn_y", TensorProto.FLOAT, [2, 1, 1, 1]),
            helper.make_tensor_value_info("rnn_h", TensorProto.FLOAT, [1, 1, 1]),
            helper.make_tensor_value_info("gru_y", TensorProto.FLOAT, [1, 1, 1, 1]),
            helper.make_tensor_value_info("gru_h", TensorProto.FLOAT, [1, 1, 1]),
            helper.make_tensor_value_info("lstm_y", TensorProto.FLOAT, [1, 1, 1, 1]),
            helper.make_tensor_value_info("lstm_h", TensorProto.FLOAT, [1, 1, 1]),
            helper.make_tensor_value_info("lstm_c", TensorProto.FLOAT, [1, 1, 1]),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == ["RNN", "GRU", "LSTM"]


# 验证 `test_onnx17_unpool_and_string_normalizer_ops` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_onnx17_unpool_and_string_normalizer_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    image = Tensor(1, 1, 2, 2, dtype="float32", data=np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32))
    grid = Tensor(
        1, 2, 2, 2, dtype="float32",
        data=np.array([[[[-1.0, -1.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, -1.0]]]], dtype=np.float32),
    )
    sampled = GridSample(["x", "grid"], ["y"], mode="bilinear", align_corners=1, dtype="float32").forward(image, grid)["tensor"]
    np.testing.assert_allclose(sampled.data, np.array([[[[1.0, 4.0], [2.5, 2.0]]]], dtype=np.float32))
    assert GridSample(["x", "grid"], ["y"]).forward_(Tensor_(1, 3, 4, 5, dtype="float32"), Tensor_(1, 6, 7, 2, dtype="float32"))["tensor"].size == (1, 3, 6, 7)

    roi_input = Tensor(1, 1, 4, 4, dtype="float32", data=np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4))
    max_roi = MaxRoiPool(["x", "rois"], ["y"], pooled_shape=[2, 2], spatial_scale=1.0, dtype="float32").forward(
        roi_input, Tensor(1, 5, dtype="float32", data=np.array([[0, 0, 0, 3, 3]], dtype=np.float32))
    )["tensor"]
    np.testing.assert_array_equal(max_roi.data, np.array([[[[5.0, 7.0], [13.0, 15.0]]]], dtype=np.float32))

    aligned = RoiAlign(
        ["x", "rois", "batch"], ["y"], output_height=1, output_width=1, sampling_ratio=1, dtype="float32"
    ).forward(
        roi_input,
        Tensor(1, 4, dtype="float32", data=np.array([[0, 0, 3, 3]], dtype=np.float32)),
        Tensor(1, dtype="int64", data=np.array([0], dtype=np.int64)),
    )["tensor"]
    np.testing.assert_allclose(aligned.data, np.array([[[[5.0]]]], dtype=np.float32), rtol=1e-6)
    aligned_max = RoiAlign(
        ["x", "rois", "batch"], ["y"], output_height=1, output_width=1, sampling_ratio=2, mode="max", dtype="float32"
    ).forward(
        roi_input,
        Tensor(1, 4, dtype="float32", data=np.array([[0, 0, 3, 3]], dtype=np.float32)),
        Tensor(1, dtype="int64", data=np.array([0], dtype=np.int64)),
    )["tensor"]
    np.testing.assert_allclose(aligned_max.data, np.array([[[[5.625]]]], dtype=np.float32), rtol=1e-6)

    pooled = Tensor(1, 1, 2, 2, dtype="float32", data=np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32))
    indices = Tensor(1, 1, 2, 2, dtype="int64", data=np.array([[[[5, 7], [13, 15]]]], dtype=np.int64))
    unpooled = MaxUnpool(["x", "i"], ["y"], kernel_shape=[2, 2], strides=[2, 2], dtype="float32").forward(
        pooled, indices
    )["tensor"]
    expected_unpool = np.zeros((1, 1, 4, 4), dtype=np.float32)
    expected_unpool.reshape(-1)[[5, 7, 13, 15]] = [1.0, 2.0, 3.0, 4.0]
    np.testing.assert_array_equal(unpooled.data, expected_unpool)
    assert MaxUnpool(["x", "i"], ["y"], kernel_shape=[2, 2], strides=[2, 2]).forward_(
        Tensor_(1, 1, 2, 2, dtype="float32"), Tensor_(1, 1, 2, 2, dtype="int64")
    )["tensor"].size == (1, 1, 4, 4)

    strings = Tensor(3, dtype="string", data=np.array(["The Café", "stop WORD", ""], dtype=np.str_))
    normalized = StringNormalizer(
        ["x"], ["y"], case_change_action="LOWER", is_case_sensitive=0, stopwords=["the", "stop"]
    ).forward(strings)["tensor"]
    np.testing.assert_array_equal(normalized.data, np.array(["cafe", "word"], dtype=np.str_))

    matrix_strings = Tensor(1, 3, dtype="string", data=np.array([["Keep", "THE item", ""]], dtype=np.str_))
    normalized_matrix = StringNormalizer(
        ["x"], ["y"], case_change_action="LOWER", stopwords=["the"]
    ).forward(matrix_strings)["tensor"]
    assert normalized_matrix.size == (1, 2)
    np.testing.assert_array_equal(normalized_matrix.data, np.array([["keep", "item"]], dtype=np.str_))

    tfidf = TfIdfVectorizer(
        ["tokens"],
        ["features"],
        mode="TFIDF",
        ngram_counts=[0, 0],
        ngram_indexes=[1, 0],
        max_skip_count=0,
        min_gram_length=2,
        max_gram_length=2,
        pool_int64s=[94, 17, 17, 36],
        weights=[0.5, 2.0],
    ).forward(Tensor(3, dtype="int64", data=np.array([94, 17, 36], dtype=np.int64)))["tensor"]
    np.testing.assert_array_equal(tfidf.data, np.array([0.5, 2.0], dtype=np.float32))

    string_tfidf = TfIdfVectorizer(
        ["tokens"],
        ["features"],
        mode="TF",
        ngram_counts=[0],
        ngram_indexes=[0, 1],
        max_skip_count=0,
        min_gram_length=1,
        max_gram_length=1,
        pool_strings=["a", "b"],
    ).forward(Tensor(2, 2, dtype="string", data=np.array([["a", "x"], ["b", "a"]], dtype=np.str_)))["tensor"]
    np.testing.assert_array_equal(string_tfidf.data, np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32))

    model_path = tmp_path / "onnx17_unpool_string_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("GridSample", ["image", "grid"], ["sampled"], mode="bilinear", align_corners=1),
            helper.make_node("MaxRoiPool", ["roi_input", "max_rois"], ["max_roi"], pooled_shape=[2, 2]),
            helper.make_node("RoiAlign", ["roi_input", "align_rois", "batch_indices"], ["aligned"], output_height=1, output_width=1, sampling_ratio=1),
            helper.make_node("MaxUnpool", ["pooled", "indices"], ["unpooled"], kernel_shape=[2, 2], strides=[2, 2]),
            helper.make_node("StringNormalizer", ["tokens"], ["normalized"], case_change_action="LOWER", stopwords=["the", "stop"]),
            helper.make_node(
                "TfIdfVectorizer",
                ["ids"],
                ["tfidf"],
                mode="TF",
                ngram_counts=[0, 0],
                ngram_indexes=[1, 0],
                max_skip_count=0,
                min_gram_length=2,
                max_gram_length=2,
                pool_int64s=[94, 17, 17, 36],
            ),
        ],
        "onnx17_unpool_string_ops",
        [
            helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 2, 2, 2]),
            helper.make_tensor_value_info("roi_input", TensorProto.FLOAT, [1, 1, 4, 4]),
            helper.make_tensor_value_info("max_rois", TensorProto.FLOAT, [1, 5]),
            helper.make_tensor_value_info("align_rois", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("batch_indices", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("pooled", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 1, 2, 2]),
            helper.make_tensor_value_info("tokens", TensorProto.STRING, [3]),
            helper.make_tensor_value_info("ids", TensorProto.INT64, [3]),
        ],
        [
            helper.make_tensor_value_info("sampled", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("max_roi", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("aligned", TensorProto.FLOAT, [1, 1, 1, 1]),
            helper.make_tensor_value_info("unpooled", TensorProto.FLOAT, [1, 1, 4, 4]),
            helper.make_tensor_value_info("normalized", TensorProto.STRING, [2]),
            helper.make_tensor_value_info("tfidf", TensorProto.FLOAT, [2]),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == [
        "GridSample", "MaxRoiPool", "RoiAlign", "MaxUnpool", "StringNormalizer", "TfIdfVectorizer"
    ]


# 验证 `test_roi_pool_ops_use_c_backend_against_reference` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_roi_pool_ops_use_c_backend_against_reference():
    x_data = np.arange(32, dtype=np.float32).reshape(2, 1, 4, 4)
    roi_input = Tensor(*x_data.shape, dtype="float32", data=x_data)

    max_rois_data = np.array(
        [
            [0, 0, 0, 3, 3],
            [1, 1, 1, 3, 3],
        ],
        dtype=np.float32,
    )
    max_rois = Tensor(*max_rois_data.shape, dtype="float32", data=max_rois_data)
    pooled = MaxRoiPool(["x", "rois"], ["y"], pooled_shape=[2, 2], dtype="float32").forward(
        roi_input, max_rois
    )["tensor"]
    np.testing.assert_array_equal(
        pooled.data,
        np.array(
            [
                [[[5.0, 7.0], [13.0, 15.0]]],
                [[[26.0, 27.0], [30.0, 31.0]]],
            ],
            dtype=np.float32,
        ),
    )

    align_rois_data = np.array([[0, 0, 3, 3], [1, 1, 3, 3]], dtype=np.float32)
    batch_indices_data = np.array([0, 1], dtype=np.int64)
    align_rois = Tensor(*align_rois_data.shape, dtype="float32", data=align_rois_data)
    batch_indices = Tensor(*batch_indices_data.shape, dtype="int64", data=batch_indices_data)

    aligned = RoiAlign(
        ["x", "rois", "batch"],
        ["y"],
        output_height=2,
        output_width=2,
        sampling_ratio=2,
        mode="avg",
        dtype="float32",
    ).forward(roi_input, align_rois, batch_indices)["tensor"]

    from onnx.reference import ReferenceEvaluator

    graph = helper.make_graph(
        [
            helper.make_node(
                "RoiAlign",
                ["x", "rois", "batch"],
                ["y"],
                output_height=2,
                output_width=2,
                sampling_ratio=2,
                mode="avg",
            )
        ],
        "roi_align_reference",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_data.shape)),
            helper.make_tensor_value_info("rois", TensorProto.FLOAT, list(align_rois_data.shape)),
            helper.make_tensor_value_info("batch", TensorProto.INT64, list(batch_indices_data.shape)),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 1, 2, 2])],
    )
    ref = ReferenceEvaluator(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
    expected = ref.run(None, {"x": x_data, "rois": align_rois_data, "batch": batch_indices_data})[0]
    np.testing.assert_allclose(aligned.data, expected, rtol=1e-6, atol=1e-6)

    aligned_max = RoiAlign(
        ["x", "rois", "batch"],
        ["y"],
        output_height=1,
        output_width=1,
        sampling_ratio=2,
        mode="max",
        coordinate_transformation_mode="output_half_pixel",
        dtype="float32",
    ).forward(roi_input, align_rois, batch_indices)["tensor"]
    graph_max = helper.make_graph(
        [
            helper.make_node(
                "RoiAlign",
                ["x", "rois", "batch"],
                ["y"],
                output_height=1,
                output_width=1,
                sampling_ratio=2,
                mode="max",
                coordinate_transformation_mode="output_half_pixel",
            )
        ],
        "roi_align_max_reference",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_data.shape)),
            helper.make_tensor_value_info("rois", TensorProto.FLOAT, list(align_rois_data.shape)),
            helper.make_tensor_value_info("batch", TensorProto.INT64, list(batch_indices_data.shape)),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 1, 1, 1])],
    )
    ref_max = ReferenceEvaluator(helper.make_model(graph_max, opset_imports=[helper.make_opsetid("", 17)]))
    expected_max = ref_max.run(None, {"x": x_data, "rois": align_rois_data, "batch": batch_indices_data})[0]
    np.testing.assert_allclose(aligned_max.data, expected_max, rtol=1e-6, atol=1e-6)


# 验证 `test_onnx17_sequence_ops` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_onnx17_sequence_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    a = Tensor(2, dtype="float32", data=np.array([1.0, 2.0], dtype=np.float32))
    b = Tensor(2, dtype="float32", data=np.array([3.0, 4.0], dtype=np.float32))
    c = Tensor(2, dtype="float32", data=np.array([5.0, 6.0], dtype=np.float32))
    seq = SequenceConstruct(["a", "b"], ["seq"], dtype="float32").forward(a, b)["tensor"]
    assert len(seq) == 2

    inserted = SequenceInsert(["seq", "c"], ["out"], dtype="float32").forward(seq, c)["tensor"]
    assert len(inserted) == 3
    picked = SequenceAt(["seq", "pos"], ["out"], dtype="float32").forward(
        inserted, Tensor(1, dtype="int64", data=np.array([-1], dtype=np.int64))
    )["tensor"]
    np.testing.assert_array_equal(picked.data, c.data)

    erased = SequenceErase(["seq"], ["out"], dtype="float32").forward(inserted)["tensor"]
    assert len(erased) == 2
    length = SequenceLength(["seq"], ["len"]).forward(erased)["tensor"]
    assert length.size == ()
    np.testing.assert_array_equal(length.data, np.array(2, dtype=np.int64))

    stacked = ConcatFromSequence(["seq"], ["out"], axis=0, new_axis=1, dtype="float32").forward(erased)["tensor"]
    np.testing.assert_array_equal(stacked.data, np.stack([a.data, b.data], axis=0))

    split_input = Tensor(2, 3, dtype="float32", data=np.arange(6, dtype=np.float32).reshape(2, 3))
    split = Tensor(2, dtype="int64", data=np.array([1, 2], dtype=np.int64))
    pieces = SplitToSequence(["x", "split"], ["seq"], axis=1, keepdims=1, dtype="float32").forward(split_input, split)["tensor"]
    assert [piece.size for piece in pieces] == [(2, 1), (2, 2)]
    np.testing.assert_array_equal(pieces[0].data, split_input.data[:, :1])
    squeezed_pieces = SplitToSequence(["x"], ["seq"], axis=1, keepdims=0, dtype="float32").forward(split_input)["tensor"]
    assert [piece.size for piece in squeezed_pieces] == [(2,), (2,), (2,)]

    empty = SequenceEmpty([], ["seq"], dtype="float32").forward()["tensor"]
    assert empty == []

    model_path = tmp_path / "onnx17_sequence_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("SequenceEmpty", [], ["empty"], dtype=TensorProto.FLOAT),
            helper.make_node("SequenceConstruct", ["a", "b"], ["seq"]),
            helper.make_node("SequenceLength", ["seq"], ["seq_len"]),
            helper.make_node("SequenceAt", ["seq", "pos"], ["at"]),
            helper.make_node("SequenceInsert", ["seq", "c"], ["seq_inserted"]),
            helper.make_node("SequenceErase", ["seq_inserted"], ["seq_erased"]),
            helper.make_node("ConcatFromSequence", ["seq_erased"], ["concat"], axis=0, new_axis=0),
            helper.make_node("SplitToSequence", ["matrix", "split"], ["split_seq"], axis=1),
        ],
        "onnx17_sequence_ops",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("pos", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("matrix", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("split", TensorProto.INT64, [2]),
        ],
        [
            helper.make_tensor_sequence_value_info("empty", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("seq_len", TensorProto.INT64, []),
            helper.make_tensor_value_info("at", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("concat", TensorProto.FLOAT, [4]),
            helper.make_tensor_sequence_value_info("split_seq", TensorProto.FLOAT, None),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == [
        "SequenceEmpty", "SequenceConstruct", "SequenceLength", "SequenceAt",
        "SequenceInsert", "SequenceErase", "ConcatFromSequence", "SplitToSequence"
    ]


# 验证 `test_onnx17_optional_ops` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_onnx17_optional_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    tensor = Tensor(2, dtype="float32", data=np.array([1.0, 2.0], dtype=np.float32))
    optional = Optional(["x"], ["opt"], dtype="float32").forward(tensor)["tensor"]
    np.testing.assert_array_equal(OptionalGetElement(["opt"], ["y"], dtype="float32").forward(optional)["tensor"].data, tensor.data)
    has = OptionalHasElement(["opt"], ["has"]).forward(optional)["tensor"]
    assert has.size == ()
    np.testing.assert_array_equal(has.data, np.array(True, dtype=np.bool_))
    empty = Optional([], ["opt"], dtype="float32").forward()["tensor"]
    empty_has = OptionalHasElement(["opt"], ["has"]).forward(empty)["tensor"]
    assert empty_has.size == ()
    np.testing.assert_array_equal(empty_has.data, np.array(False, dtype=np.bool_))
    with pytest.raises(ValueError, match="empty optional"):
        OptionalGetElement(["opt"], ["y"], dtype="float32").forward(empty)

    model_path = tmp_path / "onnx17_optional_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("Optional", ["x"], ["opt"]),
            helper.make_node("OptionalHasElement", ["opt"], ["has"]),
            helper.make_node("OptionalGetElement", ["opt"], ["y"]),
        ],
        "onnx17_optional_ops",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [
            helper.make_tensor_value_info("has", TensorProto.BOOL, []),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [2]),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == ["Optional", "OptionalHasElement", "OptionalGetElement"]


# 验证 `test_onnx17_control_flow_ops` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_onnx17_control_flow_ops(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    one_const = helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])
    zero_const = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    true_const = helper.make_tensor("true", TensorProto.BOOL, [], [True])

    then_graph = helper.make_graph(
        [helper.make_node("Constant", [], ["branch_y"], value=one_const)],
        "then_branch",
        [],
        [helper.make_tensor_value_info("branch_y", TensorProto.FLOAT, [])],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Constant", [], ["branch_y"], value=zero_const)],
        "else_branch",
        [],
        [helper.make_tensor_value_info("branch_y", TensorProto.FLOAT, [])],
    )
    if_out = If(["cond"], ["y"], then_branch=then_graph, else_branch=else_graph).forward(
        Tensor(dtype="bool", data=np.array(True, dtype=np.bool_))
    )["tensor"]
    np.testing.assert_array_equal(if_out.data, np.array(1.0, dtype=np.float32))

    loop_body = helper.make_graph(
        [
            helper.make_node("Constant", [], ["cond_out"], value=true_const),
            helper.make_node("Constant", [], ["one"], value=one_const),
            helper.make_node("Add", ["v_in", "one"], ["v_out"]),
            helper.make_node("Identity", ["v_out"], ["scan_out"]),
        ],
        "loop_body",
        [
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_in", TensorProto.FLOAT, []),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_out", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, []),
        ],
    )
    loop_final, loop_scan = Loop(["m", "cond", "v"], ["v_final", "scan"], body=loop_body).forward(
        Tensor(dtype="int64", data=np.array(3, dtype=np.int64)),
        Tensor(dtype="bool", data=np.array(True, dtype=np.bool_)),
        Tensor(dtype="float32", data=np.array(0.0, dtype=np.float32)),
    )["tensor"]
    np.testing.assert_array_equal(loop_final.data, np.array(3.0, dtype=np.float32))
    np.testing.assert_array_equal(loop_scan.data, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    zero_loop_final, zero_loop_scan = Loop(["m", "cond", "v"], ["v_final", "scan"], body=loop_body).forward(
        Tensor(dtype="int64", data=np.array(0, dtype=np.int64)),
        Tensor(dtype="bool", data=np.array(True, dtype=np.bool_)),
        Tensor(dtype="float32", data=np.array(5.0, dtype=np.float32)),
    )["tensor"]
    np.testing.assert_array_equal(zero_loop_final.data, np.array(5.0, dtype=np.float32))
    assert zero_loop_scan.data.shape == (0,)

    scan_body = helper.make_graph(
        [
            helper.make_node("Add", ["state_in", "x_in"], ["state_out"]),
            helper.make_node("Identity", ["state_out"], ["scan_y"]),
        ],
        "scan_body",
        [
            helper.make_tensor_value_info("state_in", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("x_in", TensorProto.FLOAT, []),
        ],
        [
            helper.make_tensor_value_info("state_out", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("scan_y", TensorProto.FLOAT, []),
        ],
    )
    scan_final, scan_y = Scan(["state", "x"], ["state_final", "scan_y"], body=scan_body, num_scan_inputs=1).forward(
        Tensor(dtype="float32", data=np.array(0.0, dtype=np.float32)),
        Tensor(3, dtype="float32", data=np.array([1.0, 2.0, 3.0], dtype=np.float32)),
    )["tensor"]
    np.testing.assert_array_equal(scan_final.data, np.array(6.0, dtype=np.float32))
    np.testing.assert_array_equal(scan_y.data, np.array([1.0, 3.0, 6.0], dtype=np.float32))

    seq_body = helper.make_graph(
        [
            helper.make_node("Constant", [], ["one"], value=one_const),
            helper.make_node("Add", ["item", "one"], ["mapped"]),
        ],
        "sequence_map_body",
        [helper.make_tensor_value_info("item", TensorProto.FLOAT, [])],
        [helper.make_tensor_value_info("mapped", TensorProto.FLOAT, [])],
    )
    mapped = SequenceMap(["seq"], ["out_seq"], body=seq_body).forward(
        [
            Tensor(dtype="float32", data=np.array(1.0, dtype=np.float32)),
            Tensor(dtype="float32", data=np.array(2.0, dtype=np.float32)),
        ]
    )["tensor"]
    assert len(mapped) == 2
    np.testing.assert_array_equal(mapped[0].data, np.array(2.0, dtype=np.float32))
    np.testing.assert_array_equal(mapped[1].data, np.array(3.0, dtype=np.float32))

    model_path = tmp_path / "onnx17_control_flow_ops.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("If", ["cond"], ["if_y"], then_branch=then_graph, else_branch=else_graph),
            helper.make_node("Loop", ["m", "loop_cond", "loop_v"], ["loop_final", "loop_scan"], body=loop_body),
            helper.make_node("Scan", ["scan_state", "scan_x"], ["scan_final", "scan_out"], body=scan_body, num_scan_inputs=1),
            helper.make_node("SequenceMap", ["seq"], ["mapped_seq"], body=seq_body),
        ],
        "onnx17_control_flow_ops",
        [
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("m", TensorProto.INT64, []),
            helper.make_tensor_value_info("loop_cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("loop_v", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("scan_state", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("scan_x", TensorProto.FLOAT, [3]),
            helper.make_tensor_sequence_value_info("seq", TensorProto.FLOAT, None),
        ],
        [
            helper.make_tensor_value_info("if_y", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("loop_final", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("loop_scan", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("scan_final", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [3]),
            helper.make_tensor_sequence_value_info("mapped_seq", TensorProto.FLOAT, None),
        ],
    )
    onnx.save(helper.make_model(graph), model_path)

    ops = ONNXImport(str(model_path), strict=True)

    assert [op.__class__.__name__ for op in ops] == ["If", "Loop", "Scan", "SequenceMap"]


# 验证 `test_control_flow_subgraphs_capture_outer_scope` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_control_flow_subgraphs_capture_outer_scope(monkeypatch):
    _disable_c_backend(monkeypatch)

    class PassThroughOp:
        # 初始化 `PassThroughOp` 的构造参数，保存后续运行、形状推断或验证所需的状态。
        def __init__(self, inputs, outputs):
            self.inputs = inputs
            self.outputs = outputs
            self.name = None

        # 执行 `PassThroughOp` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
        def forward(self, x):
            return {"tensor": x, "parameters": None}

        # 执行 `PassThroughOp` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
        def forward_(self, x):
            return {"tensor": Tensor_(*x.size, dtype=x.dtype), "parameters": None}

    one_const = helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])
    neg_one_const = helper.make_tensor("neg_one", TensorProto.FLOAT, [], [-1.0])

    then_graph = helper.make_graph(
        [helper.make_node("Add", ["outer_x", "one"], ["branch_y"])],
        "then_capture",
        [],
        [helper.make_tensor_value_info("branch_y", TensorProto.FLOAT, [])],
        initializer=[one_const],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Add", ["outer_x", "neg_one"], ["branch_y"])],
        "else_capture",
        [],
        [helper.make_tensor_value_info("branch_y", TensorProto.FLOAT, [])],
        initializer=[neg_one_const],
    )
    if_graph = Graph(
        [
            PassThroughOp(["cond"], ["cond2"]),
            If(["cond2"], ["y"], then_branch=then_graph, else_branch=else_graph),
        ],
        input_name=["cond", "outer_x"],
        output_name=["y"],
    )
    if_out = if_graph.forward(
        Tensor(dtype="bool", data=np.array(True, dtype=np.bool_)),
        Tensor(dtype="float32", data=np.array(2.0, dtype=np.float32)),
    )
    np.testing.assert_array_equal(if_out.data, np.array(3.0, dtype=np.float32))

    loop_body = helper.make_graph(
        [
            helper.make_node("Identity", ["cond_in"], ["cond_out"]),
            helper.make_node("Add", ["v_in", "bias"], ["v_out"]),
        ],
        "loop_capture",
        [
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_in", TensorProto.FLOAT, []),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_out", TensorProto.FLOAT, []),
        ],
    )
    loop_graph = Graph(
        [
            PassThroughOp(["cond"], ["cond2"]),
            Loop(["trip_count", "cond2", "v"], ["v_final"], body=loop_body),
        ],
        input_name=["trip_count", "cond", "v", "bias"],
        output_name=["v_final"],
    )
    loop_out = loop_graph.forward(
        Tensor(dtype="int64", data=np.array(3, dtype=np.int64)),
        Tensor(dtype="bool", data=np.array(True, dtype=np.bool_)),
        Tensor(dtype="float32", data=np.array(0.0, dtype=np.float32)),
        Tensor(dtype="float32", data=np.array(2.0, dtype=np.float32)),
    )
    np.testing.assert_array_equal(loop_out.data, np.array(6.0, dtype=np.float32))

    scan_body = helper.make_graph(
        [
            helper.make_node("Add", ["state_in", "x_in"], ["tmp"]),
            helper.make_node("Add", ["tmp", "bias"], ["state_out"]),
            helper.make_node("Identity", ["state_out"], ["scan_y"]),
        ],
        "scan_capture",
        [
            helper.make_tensor_value_info("state_in", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("x_in", TensorProto.FLOAT, []),
        ],
        [
            helper.make_tensor_value_info("state_out", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("scan_y", TensorProto.FLOAT, []),
        ],
    )
    scan_graph = Graph(
        [
            PassThroughOp(["state"], ["state2"]),
            Scan(["state2", "x"], ["state_final", "scan_y"], body=scan_body, num_scan_inputs=1),
        ],
        input_name=["state", "x", "bias"],
        output_name=["state_final", "scan_y"],
    )
    scan_final, scan_y = scan_graph.forward(
        Tensor(dtype="float32", data=np.array(0.0, dtype=np.float32)),
        Tensor(2, dtype="float32", data=np.array([1.0, 2.0], dtype=np.float32)),
        Tensor(dtype="float32", data=np.array(10.0, dtype=np.float32)),
    )
    np.testing.assert_array_equal(scan_final.data, np.array(23.0, dtype=np.float32))
    np.testing.assert_array_equal(scan_y.data, np.array([11.0, 23.0], dtype=np.float32))

    seq_body = helper.make_graph(
        [helper.make_node("Add", ["item", "bias"], ["mapped"])],
        "sequence_map_capture",
        [helper.make_tensor_value_info("item", TensorProto.FLOAT, [])],
        [helper.make_tensor_value_info("mapped", TensorProto.FLOAT, [])],
    )
    seq_graph = Graph(
        [
            PassThroughOp(["seq"], ["seq2"]),
            SequenceMap(["seq2"], ["mapped_seq"], body=seq_body),
        ],
        input_name=["seq", "bias"],
        output_name=["mapped_seq"],
    )
    mapped = seq_graph.forward(
        [
            Tensor(dtype="float32", data=np.array(1.0, dtype=np.float32)),
            Tensor(dtype="float32", data=np.array(2.0, dtype=np.float32)),
        ],
        Tensor(dtype="float32", data=np.array(5.0, dtype=np.float32)),
    )
    assert len(mapped) == 2
    np.testing.assert_array_equal(mapped[0].data, np.array(6.0, dtype=np.float32))
    np.testing.assert_array_equal(mapped[1].data, np.array(7.0, dtype=np.float32))


# 验证 `test_onehot_and_compress_shape_inference` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_onehot_and_compress_shape_inference(monkeypatch):
    _disable_c_backend(monkeypatch)

    depth = Tensor(1, dtype="int64", data=np.array([4], dtype=np.int64))
    values = Tensor(2, dtype="float32", data=np.array([0.0, 1.0], dtype=np.float32))
    onehot = OneHot(["indices", "depth", "values"], ["out"], axis=-1, dtype="float32").forward_(Tensor_(2, 3, dtype="int64"), depth, values)["tensor"]
    assert onehot.size == (2, 3, 4)

    condition = Tensor(5, dtype="bool", data=np.array([True, False, True, False, True]))
    compress_axis = Compress(["x", "cond"], ["out"], axis=1, dtype="float32").forward_(Tensor_(2, 5, dtype="float32"), condition)["tensor"]
    assert compress_axis.size == (2, 3)

    data = Tensor(2, 3, dtype="float32", data=np.arange(6, dtype=np.float32).reshape(2, 3))
    flat_cond = Tensor(6, dtype="bool", data=np.array([True, False, True, False, False, True]))
    flat = Compress(["x", "cond"], ["out"], axis=None, dtype="float32").forward(data, flat_cond)["tensor"]
    np.testing.assert_array_equal(flat.data, np.array([0.0, 2.0, 5.0], dtype=np.float32))
    assert flat.size == (3,)


# 验证 `test_reduce_and_nonzero_constant_shape_inference` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_reduce_and_nonzero_constant_shape_inference(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)

    axes = Tensor(1, dtype="int64", data=np.array([1], dtype=np.int64))
    reduced = ReduceSum(["x", "axes"], ["out"], axes=None, keepdims=0, dtype="float32").forward_(
        Tensor_(2, 3, 4, dtype="float32"), axes
    )["tensor"]
    assert reduced.size == (2, 4)

    data_arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    data = Tensor(2, 3, dtype="float32", data=data_arr)
    empty_axes = Tensor(0, dtype="int64", data=np.array([], dtype=np.int64))
    reduce_all = ReduceSum(["x", "axes"], ["out"], axes=None, keepdims=0, dtype="float32").forward(
        data, empty_axes
    )["tensor"]
    np.testing.assert_array_equal(reduce_all.data, np.sum(data_arr))
    assert reduce_all.size == ()

    no_op = ReduceSum(
        ["x", "axes"], ["out"], axes=None, keepdims=0, noop_with_empty_axes=1, dtype="float32"
    ).forward(data, empty_axes)["tensor"]
    np.testing.assert_array_equal(no_op.data, data_arr)
    assert no_op.size == (2, 3)

    axes_initializer = helper.make_tensor("axes_empty", TensorProto.INT64, [0], [])
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y0_info = helper.make_tensor_value_info("y0", TensorProto.FLOAT, [])
    y1_info = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [2, 3])
    graph = helper.make_graph(
        [
            helper.make_node("ReduceSum", ["x", "axes_empty"], ["y0"], keepdims=0),
            helper.make_node("ReduceSum", ["x", "axes_empty"], ["y1"], keepdims=0, noop_with_empty_axes=1),
        ],
        "reduce_sum_empty_axes",
        [x_info],
        [y0_info, y1_info],
        [axes_initializer],
    )
    model_path = tmp_path / "reduce_sum_empty_axes.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), model_path)
    imported = ONNXImport(str(model_path), strict=True)
    reduce_ops = [op for op in imported if op.__class__.__name__ == "ReduceSum"]
    assert reduce_ops[0].noop_with_empty_axes == 0
    assert reduce_ops[1].noop_with_empty_axes == 1

    data = Tensor(2, 3, dtype="float32", data=np.array([[1, 0, 2], [0, 0, 3]], dtype=np.float32))
    nonzero = NonZero(["x"], ["out"]).forward_(data)["tensor"]
    assert nonzero.size == (2, 3)
