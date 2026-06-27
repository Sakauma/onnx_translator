# /**
#   ******************************************************************************
#   * @file        onnx_semantic_matrix.py
#   * @author      Egor Izmaylov
#   * @brief       Builds an ONNX official operator semantic coverage matrix.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_ops import (
    DEEP_SEMANTIC_PYTEST_COVERAGE,
    PYTHON_ORCHESTRATION_RUNTIME,
    REFERENCE_PARITY_PYTEST_COVERAGE,
    OperatorInfo,
    audit,
    normalize_name,
    parse_import_supported_raw_ops,
    parse_import_supported_ops,
)


DEFAULT_JSON = ROOT / "docs" / "onnx_semantic_matrix.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "onnx_semantic_matrix.md"

OFFICIAL_SEMANTIC_ALIASES = {
    normalize_name("Scatter"): {
        "kind": "deprecated_alias",
        "canonical_operator_classes": ("ScatterElements",),
    },
    normalize_name("Upsample"): {
        "kind": "deprecated_alias",
        "canonical_operator_classes": ("Resize",),
    },
}


def parse_latest_official_schema_details() -> tuple[dict[str, dict[str, object]], str | None]:
    try:
        from onnx import defs
    except Exception as exc:  # pragma: no cover - exercised when onnx is missing locally.
        return {}, f"无法导入 onnx.defs: {exc}"

    latest = {}
    for schema in defs.get_all_schemas_with_history():
        if schema.domain != "":
            continue
        if schema.name not in latest or schema.since_version > latest[schema.name].since_version:
            latest[schema.name] = schema
    return {
        normalize_name(name): {
            "op_type": name,
            "since_version": schema.since_version,
            "deprecated": bool(schema.deprecated),
        }
        for name, schema in latest.items()
    }, None


def semantic_evidence(info: OperatorInfo) -> list[str]:
    evidence = []
    if info.class_name in REFERENCE_PARITY_PYTEST_COVERAGE:
        evidence.append("onnx_reference_pytest")
    if info.class_name in DEEP_SEMANTIC_PYTEST_COVERAGE:
        evidence.append("deep_semantic_pytest")
    if info.numerical_planned:
        evidence.append("numerical_plan")
    if info.cuda_verified:
        evidence.append("cuda_verifier")
    if info.c_runtime_functions:
        evidence.append("c_runtime")
    if info.class_name in PYTHON_ORCHESTRATION_RUNTIME:
        evidence.append("python_orchestration")
    return evidence


def row_status(import_supported: bool, matches: list[OperatorInfo], evidence: set[str]) -> str:
    if not import_supported:
        return "missing_import"
    if not matches:
        return "missing_operator_class"
    strong_evidence = {
        "onnx_reference_pytest",
        "deep_semantic_pytest",
        "numerical_plan",
        "cuda_verifier",
        "python_orchestration",
    }
    if evidence & strong_evidence:
        return "verified"
    if "c_runtime" in evidence:
        return "runtime_only"
    return "missing_semantic_evidence"


def _semantic_alias_for(normalized_name: str, infos_by_class: dict[str, OperatorInfo]) -> dict[str, object] | None:
    alias = OFFICIAL_SEMANTIC_ALIASES.get(normalized_name)
    if not alias:
        return None
    canonical_classes = [
        class_name for class_name in alias["canonical_operator_classes"] if class_name in infos_by_class
    ]
    if not canonical_classes:
        return None
    return {
        "kind": alias["kind"],
        "canonical_operator_classes": canonical_classes,
    }


def build_matrix() -> tuple[dict[str, object], list[str]]:
    infos, metadata = audit()
    official_latest, official_error = parse_latest_official_schema_details()
    import_supported = parse_import_supported_ops()
    import_supported_raw = {normalize_name(op) for op in parse_import_supported_raw_ops()}

    infos_by_name: dict[str, list[OperatorInfo]] = defaultdict(list)
    infos_by_class = {}
    for info in infos:
        infos_by_name[normalize_name(info.class_name)].append(info)
        infos_by_class[info.class_name] = info

    rows = []
    failures = []
    for normalized_name, details in sorted(official_latest.items(), key=lambda item: item[1]["op_type"]):
        direct_matches = sorted(infos_by_name.get(normalized_name, []), key=lambda item: item.class_name)
        semantic_alias = _semantic_alias_for(normalized_name, infos_by_class)
        alias_matches = []
        if semantic_alias:
            alias_matches = [infos_by_class[class_name] for class_name in semantic_alias["canonical_operator_classes"]]
        matches_by_class = {info.class_name: info for info in direct_matches + alias_matches}
        matches = sorted(matches_by_class.values(), key=lambda item: item.class_name)
        evidence = sorted({item for info in matches for item in semantic_evidence(info)})
        raw_import_supported = normalized_name in import_supported_raw
        status = row_status(raw_import_supported, matches, set(evidence))
        row = {
            "op_type": details["op_type"],
            "normalized_name": normalized_name,
            "since_version": details["since_version"],
            "deprecated": details["deprecated"],
            "import_supported": raw_import_supported,
            "direct_import_class_supported": normalized_name in import_supported,
            "direct_operator_classes": [info.class_name for info in direct_matches],
            "operator_classes": [info.class_name for info in matches],
            "semantic_alias": semantic_alias,
            "runtime_kinds": sorted({info.c_runtime_kind for info in matches}),
            "statuses": sorted({info.status for info in matches}),
            "semantic_evidence": evidence,
            "status": status,
        }
        rows.append(row)
        if status != "verified":
            failures.append(f"{official_name}: {status}")

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_onnx_latest_count": len(official_latest),
        "official_onnx_latest_error": official_error,
        "operator_class_count": metadata["operator_class_count"],
        "verified_count": sum(1 for row in rows if row["status"] == "verified"),
        "runtime_only_count": sum(1 for row in rows if row["status"] == "runtime_only"),
        "missing_or_weak_count": sum(1 for row in rows if row["status"] != "verified"),
        "deprecated_count": sum(1 for row in rows if row["deprecated"]),
        "deprecated_alias_count": sum(1 for row in rows if row["semantic_alias"]),
        "rows": rows,
    }
    if official_error:
        failures.append(official_error)
    return payload, failures


def render_markdown(payload: dict[str, object], failures: list[str]) -> str:
    lines = [
        "# ONNX Official Semantic Matrix",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Official latest default-domain operators: `{payload['official_onnx_latest_count']}`",
        f"- Verified operators: `{payload['verified_count']}`",
        f"- Runtime-only or weak evidence: `{payload['missing_or_weak_count']}`",
        f"- Deprecated official operators: `{payload['deprecated_count']}`",
        f"- Deprecated aliases covered by canonical classes: `{payload['deprecated_alias_count']}`",
        "",
        "| Op | Since | Deprecated | Import | Classes | Alias | Evidence | Status |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        classes = ", ".join(f"`{item}`" for item in row["operator_classes"]) or "-"
        alias = "-"
        if row["semantic_alias"]:
            alias = (
                f"{row['semantic_alias']['kind']} -> "
                + ", ".join(f"`{item}`" for item in row["semantic_alias"]["canonical_operator_classes"])
            )
        evidence = ", ".join(f"`{item}`" for item in row["semantic_evidence"]) or "-"
        lines.append(
            f"| `{row['op_type']}` | {row['since_version']} | {row['deprecated']} | "
            f"{row['import_supported']} | {classes} | {alias} | {evidence} | `{row['status']}` |"
        )
    if failures:
        lines.extend(["", "## Strict Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ONNX official operator semantic coverage matrix.")
    parser.add_argument("--json", default=str(DEFAULT_JSON), help="Path to write the matrix JSON.")
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN), help="Path to write the matrix Markdown report.")
    parser.add_argument("--check", action="store_true", help="Fail if any latest official op lacks semantic evidence.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload, failures = build_matrix()

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote ONNX semantic matrix JSON: {json_path}")

    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload, failures), encoding="utf-8")
    print(f"Wrote ONNX semantic matrix report: {markdown_path}")

    if args.check and failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
