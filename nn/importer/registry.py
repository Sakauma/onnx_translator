from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import onnx


DEFAULT_DOMAIN = ""


def canonical_domain(domain: str | None) -> str:
    """将 ONNX 默认域的两种合法拼写归一成空字符串。"""
    return DEFAULT_DOMAIN if domain in (None, "", "ai.onnx") else domain


# 保留公开注册表的 callable 值，避免破坏已有的直接使用者；版本元数据单独存放。
OP_FACTORY_REGISTRY: dict[tuple[str, str], Callable[..., Any]] = {}
OP_FACTORY_VERSION_SUPPORT: dict[tuple[str, str], frozenset[int]] = {}


def _key(domain: str | None, op_type: str) -> tuple[str, str]:
    return canonical_domain(domain), op_type


def register_factory(
    op_type: str,
    factory: Callable[..., Any] | None = None,
    *,
    domain: str = DEFAULT_DOMAIN,
    versions: Iterable[int] = (17,),
):
    """注册工厂；旧的 ``@register_factory("Add")`` 形式仍然有效。

    未显式声明的旧工厂只承诺其实现长期采用的 opset 17。额外版本必须通过
    ``declare_factory_versions`` 增量登记，避免把 schema 存在误当成实现兼容。
    """
    key = _key(domain, op_type)
    supported = frozenset(int(version) for version in versions)
    if not supported:
        raise ValueError(f"Factory {key} must declare at least one supported opset")

    def _decorator(func: Callable[..., Any]):
        OP_FACTORY_REGISTRY[key] = func
        OP_FACTORY_VERSION_SUPPORT[key] = supported
        return func

    if factory is not None:
        return _decorator(factory)
    return _decorator


def declare_factory_versions(
    op_type: str,
    versions: Iterable[int],
    *,
    domain: str = DEFAULT_DOMAIN,
) -> None:
    """为已经注册且经过验证的组合增加精确 opset 版本。"""
    key = _key(domain, op_type)
    if key not in OP_FACTORY_REGISTRY:
        raise KeyError(f"Cannot declare versions for unregistered factory {key}")
    OP_FACTORY_VERSION_SUPPORT[key] = frozenset(
        set(OP_FACTORY_VERSION_SUPPORT[key]) | {int(version) for version in versions}
    )


def lookup_factory(domain: str | None, op_type: str, opset: int):
    """按声明 anchor 所对应的 schema revision 匹配 effective opset。"""
    normalized_domain = canonical_domain(domain)
    keys = [_key(normalized_domain, op_type)]
    # 历史工厂中基础算子使用全大写注册名；该兼容仅发生在同一 domain 内。
    if op_type.upper() != op_type:
        keys.append(_key(normalized_domain, op_type.upper()))

    for key in keys:
        factory = OP_FACTORY_REGISTRY.get(key)
        if factory is None:
            continue
        anchors = OP_FACTORY_VERSION_SUPPORT[key]
        schema_domain = key[0]
        schema_op_type = op_type
        try:
            effective_schema = onnx.defs.get_schema(schema_op_type, opset, schema_domain)
        except Exception as error:
            return None, f"no ONNX schema at opset {opset}: {error}"

        supported_revisions = set()
        for anchor in anchors:
            try:
                supported_revisions.add(
                    onnx.defs.get_schema(schema_op_type, anchor, schema_domain).since_version
                )
            except Exception:
                # An anchor earlier than an operator's introduction grants no capability.
                continue
        if effective_schema.since_version not in supported_revisions:
            shown_anchors = ", ".join(str(version) for version in sorted(anchors))
            shown_revisions = ", ".join(str(version) for version in sorted(supported_revisions)) or "none"
            return None, (
                f"unsupported opset {opset} (effective schema revision "
                f"{effective_schema.since_version}); declared anchor opset(s): {shown_anchors}; "
                f"implemented schema revision(s): {shown_revisions}"
            )
        return factory, None

    known_domains = sorted({key[0] or "ai.onnx" for key in OP_FACTORY_REGISTRY})
    shown_domain = normalized_domain or "ai.onnx"
    if shown_domain not in known_domains:
        return None, f"unimported domain {shown_domain!r}"
    return None, f"operator {op_type!r} is not registered in domain {shown_domain!r}"
