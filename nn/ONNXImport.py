# /**
#   ******************************************************************************
#   * @file        ONNXImport.py
#   * @author      Egor Izmaylov
#   * @brief       兼容旧的 `nn.ONNXImport` 导入路径，转发到拆分后的 ONNX 导入器实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from nn.importer import GenericNode, ONNXImport

__all__ = ["GenericNode", "ONNXImport"]
