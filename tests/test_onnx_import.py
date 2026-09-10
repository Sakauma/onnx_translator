# /**
#   ******************************************************************************
#   * @file        test_onnx_import.py
#   * @author      Egor Izmaylov
#   * @brief       验证 ONNX 导入器在严格模式和非严格模式下对不支持节点的处理行为。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import onnx
import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from nn.ONNXImport import GenericNode, ONNXImport
from nn import Tensor
from nn.Operators import ADD, Softmax


# 封装 `_write_unsupported_model` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
def _write_unsupported_model(path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("UnsupportedForTest", ["x"], ["y"], name="bad_node")
    graph = helper.make_graph([node], "unsupported_graph", [x], [y])
    model = helper.make_model(graph)
    onnx.save(model, path)


# 验证 `test_onnx_import_strict_raises_on_unsupported_node` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_onnx_import_strict_raises_on_unsupported_node(tmp_path):
    model_path = tmp_path / "unsupported.onnx"
    _write_unsupported_model(model_path)

    with pytest.raises(RuntimeError, match="UnsupportedForTest"):
        ONNXImport(str(model_path), strict=True)


# 验证 `test_onnx_import_non_strict_records_generic_error` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_onnx_import_non_strict_records_generic_error(tmp_path):
    model_path = tmp_path / "unsupported.onnx"
    _write_unsupported_model(model_path)

    ops = ONNXImport(str(model_path), strict=False)

    assert len(ops) == 1
    assert isinstance(ops[0], GenericNode)
    assert ops[0].op_type == "UnsupportedForTest"
    assert "NotImplementedError" in ops[0].error


def _write_add_model(path, *, node_domain="", model_domain="", opset=17):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])
    node = helper.make_node("Add", ["x", "y"], ["z"], domain=node_domain)
    graph = helper.make_graph([node], "add_graph", [x, y], [z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid(model_domain, opset)])
    onnx.save(model, path)


def test_import_dispatches_by_canonical_domain_and_never_falls_back(tmp_path):
    default_path = tmp_path / "default_add.onnx"
    alias_path = tmp_path / "alias_add.onnx"
    custom_path = tmp_path / "custom_add.onnx"
    _write_add_model(default_path)
    _write_add_model(alias_path, node_domain="ai.onnx", model_domain="ai.onnx")
    _write_add_model(custom_path, node_domain="com.example", model_domain="com.example", opset=1)

    assert isinstance(ONNXImport(str(default_path), strict=True)[0], ADD)
    assert isinstance(ONNXImport(str(alias_path), strict=True)[0], ADD)
    with pytest.raises(RuntimeError) as exc_info:
        ONNXImport(str(custom_path), strict=True)
    message = str(exc_info.value)
    assert "domain=com.example" in message
    assert "Add" in message
    assert "opset=1" in message

    generic = ONNXImport(str(custom_path), strict=False)[0]
    assert isinstance(generic, GenericNode)
    assert (generic.domain, generic.op_type, generic.opset) == ("com.example", "Add", 1)
    assert "unimported domain" in generic.error
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        generic.forward()


def test_import_accepts_same_schema_segment_and_rejects_older_revision(tmp_path):
    compatible_path = tmp_path / "add_v20.onnx"
    old_path = tmp_path / "add_v13.onnx"
    _write_add_model(compatible_path, opset=20)
    _write_add_model(old_path, opset=13)
    assert isinstance(ONNXImport(str(compatible_path), strict=True)[0], ADD)
    with pytest.raises(RuntimeError, match=r"Add.*opset=13"):
        ONNXImport(str(old_path), strict=True)


@pytest.mark.parametrize("opset", [0, 999])
def test_import_rejects_invalid_or_future_opset_in_strict_and_non_strict_modes(tmp_path, opset):
    model_path = tmp_path / f"add_v{opset}.onnx"
    _write_add_model(model_path, opset=opset)

    with pytest.raises(RuntimeError) as exc_info:
        ONNXImport(str(model_path), strict=True)
    assert f"opset={opset}" in str(exc_info.value)

    generic = ONNXImport(str(model_path), strict=False)[0]
    assert isinstance(generic, GenericNode)
    assert generic.opset == opset
    assert "opset" in generic.error


def _write_softmax_model(path, opset, axis_marker=None):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2, 3])
    attrs = {} if axis_marker is None else {"axis": axis_marker}
    graph = helper.make_graph([helper.make_node("Softmax", ["x"], ["y"], **attrs)], "softmax", [x], [y])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)]), path)


@pytest.mark.parametrize("axis_marker, expected_axis", [(None, 1), (-1, -1)])
def test_softmax_v11_uses_flatten_semantics_with_default_and_negative_axis(
    tmp_path, axis_marker, expected_axis
):
    model_path = tmp_path / f"softmax_v11_{axis_marker}.onnx"
    _write_softmax_model(model_path, 11, axis_marker)
    op = ONNXImport(str(model_path), strict=True)[0]
    assert isinstance(op, Softmax)
    assert (op.version, op.axis) == ("11", expected_axis)

    values = np.array(
        [[[1.0, 2.0, 3.0], [4.0, 0.0, -1.0]], [[-2.0, 1.0, 0.0], [3.0, 2.0, 1.0]]],
        dtype=np.float32,
    )
    axis = expected_axis % values.ndim
    flattened = values.reshape(np.prod(values.shape[:axis], dtype=int), -1)
    shifted = flattened - flattened.max(axis=1, keepdims=True)
    expected = (np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)).reshape(values.shape)
    actual = op.forward(Tensor(*values.shape, dtype="float32", data=values))["tensor"].data
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_softmax_v17_keeps_modern_last_axis_default(tmp_path):
    model_path = tmp_path / "softmax_v17.onnx"
    _write_softmax_model(model_path, 17)
    op = ONNXImport(str(model_path), strict=True)[0]
    assert (op.version, op.axis) == ("17", -1)


def _write_external_initializer_model(path):
    weight = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name="weight")
    output = helper.make_tensor_value_info("weight", TensorProto.FLOAT, [1])
    graph = helper.make_graph([], "external_initializer", [], [output], [weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )


def test_external_initializer_loads_relative_to_model_directory(tmp_path):
    model_path = tmp_path / "external.onnx"
    _write_external_initializer_model(model_path)
    imported = ONNXImport(str(model_path), strict=True)
    np.testing.assert_array_equal(imported[0].value, np.array([2.0], dtype=np.float32))


def test_missing_external_initializer_fails_or_becomes_diagnostic(tmp_path):
    model_path = tmp_path / "external.onnx"
    _write_external_initializer_model(model_path)
    (tmp_path / "weights.bin").unlink()

    with pytest.raises(RuntimeError, match=r"external\.onnx.*weights\.bin"):
        ONNXImport(str(model_path), strict=True)
    imported = ONNXImport(str(model_path), strict=False)
    assert len(imported) == 1
    assert isinstance(imported[0], GenericNode)
    assert imported[0].diagnostic_kind == "initializer"
    assert imported[0].outputs == ["weight"]
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        imported[0].forward()
