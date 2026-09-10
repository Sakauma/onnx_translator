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
    opset_versions: dict[str, int] = field(default_factory=dict)
    model_path: str | None = None

    def get_dtype(self, name, default):
        """按张量名返回 ONNX ``TensorProto`` 类型编号，未知名称使用调用方默认值。"""
        return self.dtype_map.get(name, default)

    def get_opset(self, domain):
        """返回节点 domain 在当前模型声明的 effective opset。"""
        from .registry import canonical_domain

        normalized = canonical_domain(domain)
        if normalized not in self.opset_versions:
            shown = normalized or "ai.onnx"
            raise ValueError(f"model does not declare an opset for domain {shown!r}")
        return self.opset_versions[normalized]


class GenericNode:
    """非严格导入模式下保留图拓扑和失败信息的占位节点。

    该对象只保证输入、输出名称仍可参与后续图连接；它不实现算子语义。
    需要可执行图的调用方应使用严格模式，或在运行前检查导入器报告的
    ``generic_nodes``。
    """

    def __init__(self, op_type, inputs, outputs, name=None, attributes=None, error=None,
                 domain="", opset=None, diagnostic_kind="node"):
        self.op_type = op_type
        self.inputs = list(inputs) if inputs else []
        self.outputs = list(outputs) if outputs else []
        self.name = name if name else f"{op_type}_{outputs[0] if outputs else 'unknown'}"
        self.attributes = attributes if attributes else {}
        self.error = error
        self.domain = domain or "ai.onnx"
        self.opset = opset
        self.diagnostic_kind = diagnostic_kind
        self.executable = False

    def forward(self, *args):
        raise RuntimeError(
            f"GenericNode is diagnostic-only and cannot execute: "
            f"{self.domain}:{self.op_type} opset={self.opset}: {self.error}"
        )

    def forward_(self, *args):
        raise RuntimeError(
            f"GenericNode has no valid shape semantics: "
            f"{self.domain}:{self.op_type} opset={self.opset}: {self.error}"
        )

    @property
    def parameters(self):
        """返回适合图标签展示的紧凑诊断文本，不暴露大型属性载荷。"""
        info = []
        if self.error:
            info.append(f"error={self.error}")
        info.append(f"domain={self.domain}")
        info.append(f"opset={self.opset}")
        for k, v in self.attributes.items():
            val_str = str(v)
            if len(val_str) > 20: val_str = val_str[:17] + "..."
            info.append(f"{k}={val_str}")
        return {"info": "\\n".join(info)}
