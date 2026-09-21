# /**
#   ******************************************************************************
#   * @file        test_graph_initializer_overrides.py
#   * @author      Egor Izmaylov
#   * @brief       覆盖 graph input 与 initializer 同名时的默认值和运行时覆盖语义。
#   * @details     2026.09.21  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from nn import Graph, Tensor, Tensor_
from nn.ModelInitParas import ONNXParasGen
from nn.Operators import Constant
from nn.importer import ONNXImport


def _tensor(values):
    data = np.asarray(values, dtype=np.float32)
    return Tensor(*data.shape, dtype="float32", data=data)


def _save_override_model(tmp_path, input_order=("x", "w", "z")):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [1])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    default_w = helper.make_tensor("w", TensorProto.FLOAT, [1], [2.0])
    input_infos = {"x": x, "w": w, "z": z}
    inputs = [input_infos[name] for name in input_order]
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "w"], ["xw"]),
            helper.make_node("Add", ["xw", "z"], ["y"]),
        ],
        "overridable_initializer",
        inputs,
        [y],
        [default_w],
    )
    model = helper.make_model(
        graph, ir_version=8, opset_imports=[helper.make_opsetid("", 17)]
    )
    onnx.checker.check_model(model, full_check=True)
    path = tmp_path / "overridable_initializer.onnx"
    onnx.save(model, path)
    return path


def test_initializer_default_and_override_calls_are_repeatable(monkeypatch, tmp_path):
    path = _save_override_model(tmp_path)
    names, generated = ONNXParasGen(str(path))

    assert names == ["x", "w", "z"]
    assert len(generated) == len(names)
    np.testing.assert_array_equal(generated[1].data, np.array([2.0], dtype=np.float32))

    ops = ONNXImport(str(path), strict=True)
    default_op = ops[0]
    assert default_op.is_initializer is True
    assert default_op.is_overridable_initializer is True
    graph = Graph(ops, names, ["y"])

    generated_result = graph.forward(*generated)
    np.testing.assert_array_equal(
        generated_result.data,
        generated[0].data + generated[1].data + generated[2].data,
    )

    x = _tensor([3.0])
    z = _tensor([10.0])
    override = _tensor([5.0])

    # Omitting w binds positional inputs to the required input order x, z.
    np.testing.assert_array_equal(graph.forward(x, z).data, np.array([15.0]))
    # A complete positional call retains the ONNX declaration order x, w, z.
    np.testing.assert_array_equal(
        graph.forward(x, override, z).data, np.array([18.0])
    )
    np.testing.assert_array_equal(
        graph.forward(x, z, w=override).data, np.array([18.0])
    )
    np.testing.assert_array_equal(
        graph.forward(x=x, z=z, w=override).data, np.array([18.0])
    )
    assert graph.forward_(Tensor_(1), Tensor_(1)).size == (1,)
    assert graph.forward_(Tensor_(1), Tensor_(1), w=Tensor_(1)).size == (1,)
    # The override belongs only to its call; the serialized default remains intact.
    np.testing.assert_array_equal(graph.forward(x, z).data, np.array([15.0]))


def test_initializer_override_input_binding_validation(monkeypatch, tmp_path):
    path = _save_override_model(tmp_path)
    names, _ = ONNXParasGen(str(path))
    graph = Graph(ONNXImport(str(path), strict=True), names, ["y"])
    x, z, override = _tensor([1.0]), _tensor([4.0]), _tensor([7.0])

    with pytest.raises(ValueError, match="Unknown graph input"):
        graph.forward(x, z, missing=override)
    with pytest.raises(ValueError, match="bound more than once"):
        graph.forward(x, z, override, w=override)
    with pytest.raises(ValueError, match="Missing required graph input"):
        graph.forward(x)


@pytest.mark.parametrize(
    "input_order",
    [("w", "x", "z"), ("x", "w", "z"), ("x", "z", "w")],
)
def test_required_inputs_keep_their_order_when_default_position_varies(
    monkeypatch, tmp_path, input_order
):
    path = _save_override_model(tmp_path, input_order=input_order)
    names, _ = ONNXParasGen(str(path))
    graph = Graph(ONNXImport(str(path), strict=True), names, ["y"])
    values = {"x": _tensor([3.0]), "w": _tensor([5.0]), "z": _tensor([10.0])}

    np.testing.assert_array_equal(
        graph.forward(values["x"], values["z"]).data, np.array([15.0])
    )
    np.testing.assert_array_equal(
        graph.forward(*(values[name] for name in input_order)).data,
        np.array([18.0]),
    )


def test_plain_initializer_remains_internal(monkeypatch, tmp_path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    w = helper.make_tensor("w", TensorProto.FLOAT, [1], [2.0])
    graph_proto = helper.make_graph(
        [helper.make_node("Add", ["x", "w"], ["y"])],
        "internal_initializer",
        [x],
        [y],
        [w],
    )
    path = tmp_path / "internal_initializer.onnx"
    onnx.save(
        helper.make_model(
            graph_proto, ir_version=8,
            opset_imports=[helper.make_opsetid("", 17)],
        ),
        path,
    )

    names, tensors = ONNXParasGen(str(path))
    assert names == ["x"]
    assert len(tensors) == 1
    ops = ONNXImport(str(path), strict=True)
    assert ops[0].is_overridable_initializer is False
    runtime = Graph(ops, names, ["y"])
    np.testing.assert_array_equal(runtime.forward(_tensor([3.0])).data, np.array([5.0]))


def test_only_marked_initializer_may_share_an_input_edge():
    with pytest.raises(ValueError, match="output edge name w repeat"):
        Graph([Constant([], ["w"], value=np.array([2.0], dtype=np.float32))], ["w"])


def test_default_marker_cannot_bypass_duplicate_edge_validation():
    impostor = Constant([], ["w"], value=np.array([2.0], dtype=np.float32))
    impostor.is_overridable_initializer = True
    with pytest.raises(ValueError, match="output edge name w repeat"):
        Graph([impostor], ["w"])

    class PretendInitializer:
        def __init__(self):
            self.inputs = []
            self.outputs = ["w"]
            self.name = None
            self.is_initializer = True
            self.is_overridable_initializer = True

    with pytest.raises(ValueError, match="output edge name w repeat"):
        Graph([PretendInitializer()], ["w"])

    first = Constant([], ["w"], value=np.array([2.0], dtype=np.float32))
    second = Constant([], ["w"], value=np.array([3.0], dtype=np.float32))
    for op in (first, second):
        op.is_initializer = True
        op.is_overridable_initializer = True
    with pytest.raises(ValueError, match="output edge name w repeat"):
        Graph([first, second], ["w"])
