"""文件功能：兼容旧的 `nn.ONNXImport` 导入路径，转发到拆分后的 ONNX 导入器实现。
作者：Egor Izmaylov
时间：2026-06-02
"""

from nn.importer import GenericNode, ONNXImport

__all__ = ["GenericNode", "ONNXImport"]
