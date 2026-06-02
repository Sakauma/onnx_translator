"""文件功能：定义 ONNX 导入流程共享的上下文对象和 GenericNode 占位节点。
作者：Egor Izmaylov
时间：2026-06-02
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nn import Tensor_


@dataclass
class ImportContext:
    dtype_map: dict[str, int]
    strict: bool = False
    generic_nodes: list[Any] = field(default_factory=list)

    # 根据张量名称查询 ONNX dtype，缺省时返回调用方提供的默认类型。
    def get_dtype(self, name, default):
        return self.dtype_map.get(name, default)


class GenericNode:
    """
    通用占位节点：用于承载尚未实现、解析失败或自定义的算子。
    """
    # 初始化 `GenericNode` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, op_type, inputs, outputs, name=None, attributes=None, error=None):
        self.op_type = op_type
        self.inputs = list(inputs) if inputs else []
        self.outputs = list(outputs) if outputs else []
        self.name = name if name else f"{op_type}_{outputs[0] if outputs else 'unknown'}"
        self.attributes = attributes if attributes else {}
        self.error = error

    # 执行 `GenericNode` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, *args):
        # 运行时占位
        return {"tensor": [None] * len(self.outputs), "parameters": None}

    # 执行 `GenericNode` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, *args):
        # 图推断占位
        out_tensors = []
        for _ in self.outputs:
            out_tensors.append(Tensor_(1, dtype="float32"))
        res = out_tensors[0] if len(out_tensors) == 1 else out_tensors
        return {"tensor": res, "parameters": None, "graph": None}

    # 汇总 `GenericNode` 的导入参数和诊断信息，供图可视化或非严格导入模式读取。
    @property
    def parameters(self):
        info = []
        if self.error:
            info.append(f"error={self.error}")
        for k, v in self.attributes.items():
            val_str = str(v)
            if len(val_str) > 20: val_str = val_str[:17] + "..."
            info.append(f"{k}={val_str}")
        return {"info": "\\n".join(info)}
