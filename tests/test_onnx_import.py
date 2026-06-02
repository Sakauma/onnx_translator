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
import pytest
from onnx import TensorProto, helper

from nn.ONNXImport import GenericNode, ONNXImport


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
