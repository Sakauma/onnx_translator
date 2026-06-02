"""文件功能：聚合 ONNX 导入器的上下文、节点工厂和兼容入口。
作者：Egor Izmaylov
时间：2026-06-02
"""

from .core import ONNXImport
from .context import GenericNode, ImportContext
from .registry import OP_FACTORY_REGISTRY, register_factory

__all__ = ["ONNXImport", "GenericNode", "ImportContext", "OP_FACTORY_REGISTRY", "register_factory"]
