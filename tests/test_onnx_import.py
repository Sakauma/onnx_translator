import onnx
import pytest
from onnx import TensorProto, helper

from nn.ONNXImport import GenericNode, ONNXImport


def _write_unsupported_model(path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("UnsupportedForTest", ["x"], ["y"], name="bad_node")
    graph = helper.make_graph([node], "unsupported_graph", [x], [y])
    model = helper.make_model(graph)
    onnx.save(model, path)


def test_onnx_import_strict_raises_on_unsupported_node(tmp_path):
    model_path = tmp_path / "unsupported.onnx"
    _write_unsupported_model(model_path)

    with pytest.raises(RuntimeError, match="UnsupportedForTest"):
        ONNXImport(str(model_path), strict=True)


def test_onnx_import_non_strict_records_generic_error(tmp_path):
    model_path = tmp_path / "unsupported.onnx"
    _write_unsupported_model(model_path)

    ops = ONNXImport(str(model_path), strict=False)

    assert len(ops) == 1
    assert isinstance(ops[0], GenericNode)
    assert ops[0].op_type == "UnsupportedForTest"
    assert "NotImplementedError" in ops[0].error
