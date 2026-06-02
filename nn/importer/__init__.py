# /**
#   ******************************************************************************
#   * @file        __init__.py
#   * @author      Egor Izmaylov
#   * @brief       聚合 ONNX 导入器的上下文、节点工厂和兼容入口。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .core import ONNXImport
from .context import GenericNode, ImportContext
from .registry import OP_FACTORY_REGISTRY, register_factory

__all__ = ["ONNXImport", "GenericNode", "ImportContext", "OP_FACTORY_REGISTRY", "register_factory"]
