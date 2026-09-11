import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from nn import Graph, Tensor, Tensor_
from nn.ONNXImport import ONNXImport
from nn.Operators import Resize, Slice


def _tensor(array, dtype):
    array = np.asarray(array)
    return Tensor(*array.shape, dtype=dtype, data=array)


def test_resize_opset17_omitted_attributes_match_reference(tmp_path):
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 2])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 3])
    sizes = numpy_helper.from_array(np.array([1, 1, 1, 3], dtype=np.int64), "sizes")
    graph_proto = helper.make_graph(
        [helper.make_node("Resize", ["x", "", "", "sizes"], ["y"], mode="nearest")],
        "resize_defaults",
        [x_info],
        [y_info],
        initializer=[sizes],
    )
    model = helper.make_model(
        graph_proto, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    onnx.checker.check_model(model, full_check=True)
    model_path = tmp_path / "resize_defaults.onnx"
    onnx.save(model, model_path)

    x = np.array([[[[1.0, 2.0]]]], dtype=np.float32)
    expected = ReferenceEvaluator(model).run(None, {"x": x})[0]
    ops = ONNXImport(str(model_path), strict=True)
    imported = next(op for op in ops if isinstance(op, Resize))
    actual = Graph(ops, ["x"], ["y"]).forward(_tensor(x, "float32"))

    assert imported.coord_mode_str == "half_pixel"
    assert imported.nearest_mode_str == "round_prefer_floor"
    np.testing.assert_array_equal(actual.data, expected)


def test_resize_direct_defaults_match_opset17_reference():
    x = np.array([[[[1.0, 2.0]]]], dtype=np.float32)
    sizes = np.array([1, 1, 1, 4], dtype=np.int64)
    op = Resize(["x", "", "", "sizes"], ["y"], dtype="float32")
    actual = op.forward(_tensor(x, "float32"), None, None, _tensor(sizes, "int64"))["tensor"]

    node = helper.make_node("Resize", ["x", "", "", "sizes"], ["y"], mode="nearest")
    graph_proto = helper.make_graph(
        [node],
        "resize_direct_defaults",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
            helper.make_tensor_value_info("sizes", TensorProto.INT64, [4]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 4])],
    )
    model = helper.make_model(graph_proto, opset_imports=[helper.make_opsetid("", 17)])
    expected = ReferenceEvaluator(model).run(None, {"x": x, "sizes": sizes})[0]
    np.testing.assert_array_equal(actual.data, expected)


@pytest.mark.parametrize(
    ("starts", "ends", "step"),
    [
        (np.iinfo(np.int64).min, np.iinfo(np.int64).max, 1),
        (np.iinfo(np.int64).max, np.iinfo(np.int64).min, -1),
    ],
)
def test_slice_empty_dimension_matches_reference_and_shape_inference(tmp_path, starts, ends, step):
    data = np.empty((0,), dtype=np.float32)
    starts_tensor = _tensor(np.array([starts], dtype=np.int64), "int64")
    ends_tensor = _tensor(np.array([ends], dtype=np.int64), "int64")
    axes_tensor = _tensor(np.array([0], dtype=np.int64), "int64")
    steps_tensor = _tensor(np.array([step], dtype=np.int64), "int64")
    op = Slice(["x", "starts", "ends", "axes", "steps"], ["y"], dtype="float32")

    actual = op.forward(
        _tensor(data, "float32"), starts_tensor, ends_tensor, axes_tensor, steps_tensor
    )["tensor"]
    fallback_op = Slice(["x", "starts", "ends", "axes", "steps"], ["y"], dtype="float32")
    fallback_op.lib = None
    fallback = fallback_op.forward(
        _tensor(data, "float32"), starts_tensor, ends_tensor, axes_tensor, steps_tensor
    )["tensor"]
    inferred = op.forward_(
        Tensor_(0, dtype="float32"), starts_tensor, ends_tensor, axes_tensor, steps_tensor
    )["tensor"]

    node = helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"])
    graph_proto = helper.make_graph(
        [node],
        "slice_empty",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [0])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [0])],
        initializer=[
            numpy_helper.from_array(starts_tensor.data, "starts"),
            numpy_helper.from_array(ends_tensor.data, "ends"),
            numpy_helper.from_array(axes_tensor.data, "axes"),
            numpy_helper.from_array(steps_tensor.data, "steps"),
        ],
    )
    model = helper.make_model(graph_proto, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model, full_check=True)
    expected = ReferenceEvaluator(model).run(None, {"x": data})[0]
    model_path = tmp_path / f"slice_empty_{step}.onnx"
    onnx.save(model, model_path)
    imported = Graph(ONNXImport(str(model_path), strict=True), ["x"], ["y"]).forward(
        _tensor(data, "float32")
    )

    assert actual.size == expected.shape == (0,)
    assert fallback.size == expected.shape
    assert imported.size == expected.shape
    assert inferred.size == (0,)
    np.testing.assert_array_equal(actual.data, expected)
    np.testing.assert_array_equal(fallback.data, expected)
    np.testing.assert_array_equal(imported.data, expected)


@pytest.mark.parametrize(
    ("starts", "ends", "step", "expected"),
    [(1, 5, 2, [1.0, 3.0]), (5, np.iinfo(np.int64).min, -2, [5.0, 3.0, 1.0])],
)
def test_slice_nonempty_positive_and_negative_controls(starts, ends, step, expected):
    data = np.arange(6, dtype=np.float32)
    inputs = [
        _tensor(data, "float32"),
        _tensor(np.array([starts], dtype=np.int64), "int64"),
        _tensor(np.array([ends], dtype=np.int64), "int64"),
        _tensor(np.array([0], dtype=np.int64), "int64"),
        _tensor(np.array([step], dtype=np.int64), "int64"),
    ]
    actual = Slice(["x", "starts", "ends", "axes", "steps"], ["y"], dtype="float32").forward(*inputs)["tensor"]
    expected_array = np.array(expected, dtype=np.float32)
    np.testing.assert_array_equal(actual.data, expected_array)
    assert actual.size == expected_array.shape


def test_slice_extreme_negative_start_keeps_schema_clamping_across_backends():
    data = np.arange(3, dtype=np.float32)
    starts = _tensor(np.array([np.iinfo(np.int64).min], dtype=np.int64), "int64")
    ends = _tensor(np.array([np.iinfo(np.int64).min], dtype=np.int64), "int64")
    axes = _tensor(np.array([0], dtype=np.int64), "int64")
    steps = _tensor(np.array([-1], dtype=np.int64), "int64")
    args = (_tensor(data, "float32"), starts, ends, axes, steps)

    c_op = Slice(["x", "starts", "ends", "axes", "steps"], ["y"], dtype="float32")
    fallback_op = Slice(["x", "starts", "ends", "axes", "steps"], ["y"], dtype="float32")
    fallback_op.lib = None
    actual_c = c_op.forward(*args)["tensor"]
    actual_fallback = fallback_op.forward(*args)["tensor"]
    inferred = c_op.forward_(Tensor_(3, dtype="float32"), starts, ends, axes, steps)["tensor"]
    expected = np.array([0.0], dtype=np.float32)

    np.testing.assert_array_equal(actual_c.data, expected)
    np.testing.assert_array_equal(actual_fallback.data, expected)
    assert actual_c.size == actual_fallback.size == inferred.size == expected.shape
