# /**
#   ******************************************************************************
#   * @file        context.py
#   * @author      Egor Izmaylov
#   * @brief       定义 ONNX 导入流程共享的上下文对象和 GenericNode 占位节点。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nn import Tensor_


@dataclass
class ImportContext:
    """一次模型导入期间由所有节点工厂共享的状态。

    ``dtype_map`` 以 ONNX 张量名为键，只保存类型推断结果，不持有张量数据；
    ``generic_nodes`` 则由非严格模式下的失败节点追加诊断记录。上下文的生命周期
    仅覆盖一次 :func:`ONNXImport` 调用，不能跨模型复用。
    """

    dtype_map: dict[str, int]
    strict: bool = False
    generic_nodes: list[Any] = field(default_factory=list)

    def get_dtype(self, name, default):
        """按张量名返回 ONNX ``TensorProto`` 类型编号，未知名称使用调用方默认值。"""
        return self.dtype_map.get(name, default)


class GenericNode:
    """非严格导入模式下保留图拓扑和失败信息的占位节点。

    该对象只保证输入、输出名称仍可参与后续图连接；它不实现算子语义。
    需要可执行图的调用方应使用严格模式，或在运行前检查导入器报告的
    ``generic_nodes``。
    """

    def __init__(self, op_type, inputs, outputs, name=None, attributes=None, error=None):
        self.op_type = op_type
        self.inputs = list(inputs) if inputs else []
        self.outputs = list(outputs) if outputs else []
        self.name = name if name else f"{op_type}_{outputs[0] if outputs else 'unknown'}"
        self.attributes = attributes if attributes else {}
        self.error = error

    def forward(self, *args):
        # 保持图运行器的返回协议，但用 None 明确表示没有产生可用数值。
        return {"tensor": [None] * len(self.outputs), "parameters": None}

    def forward_(self, *args):
        # 占位形状不代表 ONNX 推断结果，仅让非严格模式能够继续遍历图结构。
        out_tensors = []
        for _ in self.outputs:
            out_tensors.append(Tensor_(1, dtype="float32"))
        res = out_tensors[0] if len(out_tensors) == 1 else out_tensors
        return {"tensor": res, "parameters": None, "graph": None}

    @property
    def parameters(self):
        """返回适合图标签展示的紧凑诊断文本，不暴露大型属性载荷。"""
        info = []
        if self.error:
            info.append(f"error={self.error}")
        for k, v in self.attributes.items():
            val_str = str(v)
            if len(val_str) > 20: val_str = val_str[:17] + "..."
            info.append(f"{k}={val_str}")
        return {"info": "\\n".join(info)}
