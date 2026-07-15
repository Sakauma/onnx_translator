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

# 注册表在工厂模块导入时填充；core.py 的显式导入不可删除，否则装饰器不会执行。
# 键沿用工厂声明的 ONNX op_type 拼写，分派端同时尝试原始名称和全大写名称。
OP_FACTORY_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_factory(op_type: str, factory: Callable[..., Any] | None = None):
    """注册单节点工厂，并兼容装饰器和直接调用两种写法。

    工厂接收 ``(NodeProto, ImportContext)`` 并返回一个内部算子对象。相同键再次
    注册时后者覆盖前者，因此每个 op_type 应只有一个权威实现。
    """

    def _decorator(func: Callable[..., Any]):
        OP_FACTORY_REGISTRY[op_type] = func
        return func

    if factory is not None:
        return _decorator(factory)
    return _decorator
