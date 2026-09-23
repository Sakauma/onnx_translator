"""Checker-valid regressions for control-flow container runtime and metadata."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T, helper, numpy_helper

import nn
from nn import Graph, Tensor, Tensor_
from nn.ONNXImport import ONNXImport
from conftest import _disable_c_backend


def _vi(name, shape, dtype=T.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def _tensor(value, dtype="float32"):
    array = np.asarray(value)
    return Tensor(*array.shape, dtype=dtype, data=array)


def _import_graph(tmp_path, name, graph):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model, full_check=True)
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, path)
    return Graph(
        ONNXImport(str(path), strict=True),
        [value.name for value in graph.input],
        [value.name for value in graph.output],
    )


@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("present", [False, True])
def test_loop_optional_state_preserves_presence_and_other_carried_value(
    monkeypatch, tmp_path, count, present
):
    _disable_c_backend(monkeypatch)
    element_type = helper.make_tensor_type_proto(T.FLOAT, [2])
    optional_type = helper.make_optional_type_proto(element_type)
    body = helper.make_graph(
        [
            helper.make_node("Identity", ["cond_in"], ["cond_out"]),
            helper.make_node("Identity", ["optional_in"], ["optional_out"]),
            helper.make_node("Identity", ["other_in"], ["other_out"]),
        ],
        "optional_body",
        [_vi("iter", [], T.INT64), _vi("cond_in", [], T.BOOL),
         helper.make_value_info("optional_in", optional_type), _vi("other_in", [2])],
        [_vi("cond_out", [], T.BOOL),
         helper.make_value_info("optional_out", optional_type), _vi("other_out", [2])],
    )
    optional_node = (
        helper.make_node("Optional", ["x"], ["initial"])
        if present else helper.make_node("Optional", [], ["initial"], type=element_type)
    )
    graph = helper.make_graph(
        [
            optional_node,
            helper.make_node(
                "Loop", ["m", "cond", "initial", "other"],
                ["final_optional", "final_other"], body=body,
            ),
            helper.make_node("OptionalHasElement", ["final_optional"], ["has"]),
        ],
        "loop_optional_carried",
        [_vi("m", [], T.INT64), _vi("cond", [], T.BOOL),
         _vi("x", [2]), _vi("other", [2])],
        [_vi("has", [], T.BOOL), _vi("final_other", [2])],
    )
    runtime = _import_graph(tmp_path, f"loop_optional_{count}_{present}", graph)
    x = np.array([3, 4], np.float32)
    other = np.array([8, 9], np.float32)
    has, final_other = runtime.forward(
        _tensor(count, "int64"), _tensor(True, "bool"),
        _tensor(x), _tensor(other),
    )
    assert bool(has.data) is present
    np.testing.assert_array_equal(final_other.data, other)
    has_meta, other_meta = runtime.forward_(
        Tensor_(dtype="int64"), Tensor_(dtype="bool"),
        Tensor_(2), Tensor_(2),
    )
    assert isinstance(has_meta, Tensor_) and has_meta.size == () and has_meta.dtype == "bool"
    assert isinstance(other_meta, Tensor_) and other_meta.size == (2,)


@pytest.mark.parametrize("empty", [False, True])
def test_sequence_map_preserves_length_and_declared_element_kind(monkeypatch, tmp_path, empty):
    _disable_c_backend(monkeypatch)
    body = helper.make_graph(
        [helper.make_node("Identity", ["item"], ["mapped_item"])],
        "map_body", [_vi("item", ["width"])], [_vi("mapped_item", ["width"])],
    )
    nodes = []
    inputs = []
    if empty:
        nodes.append(helper.make_node("SequenceEmpty", [], ["sequence"], dtype=T.FLOAT))
    else:
        inputs.append(_vi("x", [2]))
        nodes.append(helper.make_node("SequenceConstruct", ["x"], ["sequence"]))
    nodes.extend([
        helper.make_node("SequenceMap", ["sequence"], ["mapped"], body=body),
        helper.make_node("SequenceLength", ["mapped"], ["length"]),
    ])
    if not empty:
        nodes.append(helper.make_node("SequenceAt", ["mapped", "index"], ["item"]))
    outputs = [_vi("length", [], T.INT64)]
    if not empty:
        outputs.append(_vi("item", [2]))
    initializers = (
        [] if empty else [numpy_helper.from_array(np.asarray(0, np.int64), name="index")]
    )
    graph = helper.make_graph(
        nodes, "map_length", inputs, outputs, initializer=initializers,
    )
    runtime = _import_graph(tmp_path, f"map_length_{empty}", graph)
    if empty:
        empty_op = next(op for op in runtime.ops.values() if isinstance(op, nn.Operators.SequenceEmpty))
        empty_meta = empty_op.forward_()["tensor"]
        assert isinstance(empty_meta, nn.Sequence_)
        assert empty_meta.length == 0 and empty_meta.element.dtype == "float32"
        assert int(runtime.forward().data) == 0
        assert int(runtime.forward_().data) == 0
        mapped = next(op for op in runtime.ops.values() if isinstance(op, nn.Operators.SequenceMap))
        result = mapped.forward_([])["tensor"]
        assert isinstance(result, nn.Sequence_)
        assert result.length == 0 and result.element.dtype == "float32"
    else:
        actual_length, actual_item = runtime.forward(_tensor(np.array([3, 4], np.float32)))
        metadata_length, metadata_item = runtime.forward_(Tensor_(2))
        assert int(actual_length.data) == int(metadata_length.data) == 1
        np.testing.assert_array_equal(actual_item.data, [3, 4])
        assert metadata_item.size == (2,)


def test_sequence_map_unknown_length_keeps_symbolic_element_shape(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)
    body = helper.make_graph(
        [helper.make_node("Identity", ["item"], ["mapped_item"])],
        "map_body", [_vi("item", ["width"])], [_vi("mapped_item", ["width"])],
    )
    graph = helper.make_graph(
        [helper.make_node("SequenceMap", ["sequence"], ["mapped"], body=body)],
        "map_unknown", [helper.make_tensor_sequence_value_info("sequence", T.FLOAT, [None])],
        [helper.make_tensor_sequence_value_info("mapped", T.FLOAT, [None])],
    )
    runtime = _import_graph(tmp_path, "map_unknown", graph)
    mapped = runtime.forward_(nn.Sequence_(Tensor_(3)))
    assert isinstance(mapped, nn.Sequence_)
    assert mapped.length is None and mapped.element.size == (3,)


@pytest.mark.parametrize(
    "axis,new_axis,expected_runtime,expected_meta",
    [(-1, 0, (2, 3), (2, None)), (-1, 1, (2, 3, 1), (2, 3, None))],
)
def test_if_sequence_concat_handles_unknown_length_and_negative_axis(
    monkeypatch, tmp_path, axis, new_axis, expected_runtime, expected_meta
):
    _disable_c_backend(monkeypatch)
    sequence_info = helper.make_tensor_sequence_value_info("branch_seq", T.FLOAT, [2, 3])
    def branch(name):
        return helper.make_graph(
            [helper.make_node("SequenceConstruct", ["x"], ["branch_seq"])],
            name, [], [sequence_info],
        )
    graph = helper.make_graph(
        [
            helper.make_node("If", ["cond"], ["sequence"],
                             then_branch=branch("then"), else_branch=branch("else")),
            helper.make_node("ConcatFromSequence", ["sequence"], ["y"],
                             axis=axis, new_axis=new_axis),
        ],
        "if_concat", [_vi("cond", [], T.BOOL), _vi("x", [2, 3])],
        [_vi("y", list(expected_runtime))],
    )
    runtime = _import_graph(tmp_path, f"if_concat_{new_axis}", graph)
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(runtime.forward(_tensor(True, "bool"), _tensor(x)).data, x.reshape(expected_runtime))
    assert runtime.forward_(Tensor_(dtype="bool"), Tensor_(2, 3)).size == expected_meta


@pytest.mark.parametrize("condition", [False, True])
def test_if_optional_has_element_and_conservative_get_element(monkeypatch, tmp_path, condition):
    _disable_c_backend(monkeypatch)
    element_type = helper.make_tensor_type_proto(T.FLOAT, [2])
    optional_type = helper.make_optional_type_proto(element_type)
    output_info = helper.make_value_info("branch_optional", optional_type)
    present = helper.make_graph(
        [helper.make_node("Optional", ["x"], ["branch_optional"])],
        "present", [], [output_info],
    )
    absent = helper.make_graph(
        [helper.make_node("Optional", [], ["branch_optional"], type=element_type)],
        "absent", [], [output_info],
    )
    graph = helper.make_graph(
        [
            helper.make_node("If", ["cond"], ["optional"],
                             then_branch=present, else_branch=absent),
            helper.make_node("OptionalHasElement", ["optional"], ["has"]),
        ],
        "if_optional", [_vi("cond", [], T.BOOL), _vi("x", [2])],
        [_vi("has", [], T.BOOL)],
    )
    runtime = _import_graph(tmp_path, f"if_optional_{condition}", graph)
    x = _tensor(np.array([3, 4], np.float32))
    assert bool(runtime.forward(_tensor(condition, "bool"), x).data) is condition
    result = runtime.forward_(Tensor_(dtype="bool"), Tensor_(2))
    assert isinstance(result, Tensor_) and result.size == () and result.dtype == "bool"
    if_operator = next(op for op in runtime.ops.values() if isinstance(op, nn.Operators.If))
    optional_meta = if_operator.forward_(Tensor_(dtype="bool"))["tensor"]
    assert isinstance(optional_meta, nn.Optional_)
    assert optional_meta.present is None and optional_meta.element.size == (2,)
    element_meta = nn.Operators.OptionalGetElement(["optional"], ["y"]).forward_(optional_meta)["tensor"]
    assert element_meta.size == (2,)


def test_empty_optional_sequence_keeps_declared_element_kind(monkeypatch, tmp_path):
    _disable_c_backend(monkeypatch)
    sequence_type = helper.make_sequence_type_proto(helper.make_tensor_type_proto(T.FLOAT, [2]))
    optional_type = helper.make_optional_type_proto(sequence_type)
    graph = helper.make_graph(
        [helper.make_node("Optional", [], ["optional"], type=sequence_type)],
        "empty_optional_sequence", [], [helper.make_value_info("optional", optional_type)],
    )
    runtime = _import_graph(tmp_path, "empty_optional_sequence", graph)
    assert runtime.forward() is None
    metadata = runtime.forward_()
    assert isinstance(metadata, nn.Optional_) and metadata.present is False
    assert isinstance(metadata.element, nn.Sequence_)
    assert metadata.element.element.size == (2,)
