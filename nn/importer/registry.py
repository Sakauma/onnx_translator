# /**
#   ******************************************************************************
#   * @file        registry.py
#   * @author      Egor Izmaylov
#   * @brief       维护 ONNX 节点工厂注册表，便于后续按算子拆分导入逻辑。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

from collections.abc import Callable
from typing import Any

OP_FACTORY_REGISTRY: dict[str, Callable[..., Any]] = {}


# 注册指定 ONNX op_type 的节点工厂函数，返回原函数以支持装饰器式使用。
def register_factory(op_type: str, factory: Callable[..., Any] | None = None):
    def _decorator(func: Callable[..., Any]):
        OP_FACTORY_REGISTRY[op_type] = func
        return func

    if factory is not None:
        return _decorator(factory)
    return _decorator
