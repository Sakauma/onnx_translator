# /**
#   ******************************************************************************
#   * @file        release_dashboard.py
#   * @author      Egor Izmaylov
#   * @brief       Builds a concise release evidence dashboard from preflight output and CI configuration.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFLIGHT = ROOT / "result" / "release_preflight.json"
DEFAULT_MARKDOWN = ROOT / "result" / "release_dashboard.md"
DEFAULT_JSON = ROOT / "result" / "release_dashboard.json"
TREND_MANIFEST = ROOT / "docs" / "release_trend_manifest.json"
TREND_HISTORY = ROOT / "docs" / "release_trend_history.json"


GATES = [
    {
        "id": "sanitizer",
        "title": "ASan/UBSan Sanitizer",
        "steps": ["sanitize"],
        "artifacts": [],
        "workflow_checks": [(".github/workflows/ci.yml", "Run ASan/UBSan C backend gate")],
    },
    {
        "id": "manylinux_smoke",
        "title": "Manylinux Smoke Wheel",
        "steps": ["manylinux-wheels", "manylinux-wheelhouse-check"],
        "artifacts": ["wheelhouse"],
        "workflow_checks": [(".github/workflows/wheels.yml", "manylinux-smoke")],
    },
    {
        "id": "manylinux_full",
        "title": "Manylinux Full Matrix",
        "steps": ["manylinux-wheels-full", "manylinux-wheelhouse-check-full"],
        "artifacts": ["wheelhouse"],
        "workflow_checks": [(".github/workflows/wheels.yml", "manylinux-full")],
    },
    {
        "id": "performance_smoke",
        "title": "Performance Smoke/Baseline",
        "steps": ["benchmark-smoke-report", "benchmark-baseline-check"],
        "artifacts": ["result/benchmark_smoke.json", "result/benchmark_baseline_check.json"],
        "workflow_checks": [(".github/workflows/ci.yml", "Run benchmark smoke gate")],
    },
    {
        "id": "performance_fixed_runner",
        "title": "Fixed Runner Performance",
        "steps": ["benchmark-fixed-runner-check"],
        "artifacts": ["docs/performance_fixed_runner_baseline.json", "result/benchmark_fixed_runner_check.json"],
        "workflow_checks": [(".github/workflows/performance.yml", "fixed-runner-baseline")],
    },
    {
        "id": "cuda_smoke",
        "title": "CUDA Smoke",
        "steps": ["verify-cuda-smoke"],
        "artifacts": [],
        "workflow_checks": [(".github/workflows/cuda.yml", "cuda-smoke")],
    },
    {
        "id": "cuda_full",
        "title": "Full CUDA Numerical",
        "steps": ["verify-cuda-full"],
        "artifacts": [],
        "workflow_checks": [
            (".github/workflows/cuda.yml", "cuda-full"),
            (".github/workflows/cuda.yml", "make PYTHON=python verify-cuda-full"),
        ],
    },
]


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _step_statuses(preflight: dict[str, object] | None) -> dict[str, str]:
    return {name: str(record.get("status", "unknown")) for name, record in _step_records(preflight).items()}


def _step_records(preflight: dict[str, object] | None) -> dict[str, dict[str, object]]:
    if not preflight:
        return {}
    records = {}
    for step in preflight.get("steps", []):
        if isinstance(step, dict) and isinstance(step.get("name"), str):
            records[str(step["name"])] = {
                "name": str(step["name"]),
                "command": list(step.get("command", [])),
                "status": str(step.get("status", "unknown")),
                "returncode": step.get("returncode"),
                "duration_seconds": step.get("duration_seconds"),
                "started_at": step.get("started_at"),
            }
    return records


def _artifact(path_text: str) -> dict[str, object]:
    path = ROOT / path_text
    if path.is_dir():
        count = sum(1 for _ in path.iterdir())
        return {"path": path_text, "present": True, "kind": "directory", "entries": count}
    if path.exists():
        return {"path": path_text, "present": True, "kind": "file", "bytes": path.stat().st_size}
    return {"path": path_text, "present": False, "kind": "missing"}


def _workflow_check(path_text: str, token: str) -> dict[str, object]:
    path = ROOT / path_text
    present = path.exists() and token in path.read_text(encoding="utf-8")
    return {"path": path_text, "token": token, "present": present}


def _unique_texts(values: list[object]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _history_by_window(history: dict[str, object] | None) -> dict[str, dict[str, object]]:
    if not history:
        return {}
    return {
        str(window.get("id")): window
        for window in history.get("windows", [])
        if isinstance(window, dict) and window.get("id")
    }


def _latest_successful_run(window: dict[str, object]) -> dict[str, object] | None:
    runs = window.get("successful_runs", [])
    if isinstance(runs, list) and runs and isinstance(runs[0], dict):
        return runs[0]
    return None


def _trend_windows(
    manifest: dict[str, object] | None,
    history: dict[str, object] | None,
) -> list[dict[str, object]]:
    if not manifest:
        return []
    history_windows = _history_by_window(history)
    windows = []
    for window in manifest.get("trend_windows", []):
        if not isinstance(window, dict):
            continue
        window_id = str(window.get("id", ""))
        history_window = history_windows.get(window_id, {})
        latest_run = _latest_successful_run(history_window)
        workflow = str(window.get("source_workflow", ""))
        tokens = list(window.get("workflow_tokens", []))
        artifact_pattern = str(window.get("artifact_pattern", ""))
        retention_days = window.get("retention_days")
        if artifact_pattern:
            tokens.append(artifact_pattern)
        if retention_days is not None:
            tokens.append(f"retention-days: {retention_days}")
        windows.append(
            {
                "id": window_id,
                "title": str(window.get("title", window.get("id", ""))),
                "cadence": str(window.get("cadence", "")),
                "source_workflow": workflow,
                "artifact_pattern": artifact_pattern,
                "retention_days": retention_days,
                "required_payloads": [str(item) for item in window.get("required_payloads", [])],
                "workflow_checks": [_workflow_check(workflow, token) for token in _unique_texts(tokens) if workflow],
                "history_status": str(history_window.get("status", "missing")),
                "history_successful_run_count": int(history_window.get("successful_run_count", 0)),
                "history_minimum_required": int(
                    history_window.get(
                        "minimum_required",
                        manifest.get("minimum_history_for_top_tier_release", 0),
                    )
                ),
                "history_latest_run_url": latest_run.get("run_url") if latest_run else None,
                "history_next_action": history_window.get("next_action"),
            }
        )
    return windows


def _status_for_gate(gate: dict[str, object], step_statuses: dict[str, str], workflow_checks: list[dict[str, object]]) -> str:
    steps = list(gate["steps"])
    if not steps:
        return "configured" if all(check["present"] for check in workflow_checks) else "missing"

    statuses = [step_statuses.get(step) for step in steps]
    if all(status == "passed" for status in statuses):
        return "passed"
    if any(status == "failed" for status in statuses):
        return "failed"
    if all(status == "skipped" for status in statuses):
        return "planned"
    if any(status is not None for status in statuses):
        return "partial"
    return "configured" if all(check["present"] for check in workflow_checks) else "missing"


def build_dashboard(preflight_path: Path = DEFAULT_PREFLIGHT) -> dict[str, object]:
    preflight = _read_json(preflight_path)
    trend_manifest = _read_json(TREND_MANIFEST)
    trend_history = _read_json(TREND_HISTORY)
    records = _step_records(preflight)
    statuses = _step_statuses(preflight)
    gates = []
    for gate in GATES:
        workflow_checks = [_workflow_check(path, token) for path, token in gate["workflow_checks"]]
        artifacts = [_artifact(path) for path in gate["artifacts"]]
        gate_steps = {step: statuses.get(step, "not-run") for step in gate["steps"]}
        gate_step_records = [
            records.get(
                step,
                {
                    "name": step,
                    "command": [],
                    "status": "not-run",
                    "returncode": None,
                    "duration_seconds": None,
                    "started_at": None,
                },
            )
            for step in gate["steps"]
        ]
        gates.append(
            {
                "id": gate["id"],
                "title": gate["title"],
                "status": _status_for_gate(gate, statuses, workflow_checks),
                "steps": gate_steps,
                "step_records": gate_step_records,
                "artifacts": artifacts,
                "workflow_checks": workflow_checks,
            }
        )

    preflight_flags = {}
    if preflight:
        for key in [
            "include_cuda_smoke",
            "include_cuda_full",
            "include_manylinux",
            "include_manylinux_full",
            "include_fixed_runner_perf",
        ]:
            preflight_flags[key] = bool(preflight.get(key, False))

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preflight_path": str(preflight_path),
        "preflight_present": preflight is not None,
        "preflight_status": preflight.get("status") if preflight else "missing",
        "preflight_dry_run": preflight.get("dry_run") if preflight else None,
        "preflight_flags": preflight_flags,
        "gates": gates,
        "trend_manifest_path": str(TREND_MANIFEST),
        "trend_manifest_present": trend_manifest is not None,
        "trend_history_path": str(TREND_HISTORY),
        "trend_history_present": trend_history is not None,
        "trend_history_all_windows_top_tier_ready": (
            trend_history.get("all_windows_top_tier_ready") if trend_history else None
        ),
        "trend_minimum_history_for_top_tier_release": (
            trend_manifest.get("minimum_history_for_top_tier_release") if trend_manifest else None
        ),
        "trend_windows": _trend_windows(trend_manifest, trend_history),
    }


def _format_command(command: list[object]) -> str:
    if not command:
        return "-"
    return " ".join(str(part) for part in command)


def _format_steps(step_records: list[dict[str, object]]) -> str:
    if not step_records:
        return "workflow-only"
    parts = []
    for record in step_records:
        details = []
        if record.get("returncode") is not None:
            details.append(f"rc={record['returncode']}")
        if record.get("duration_seconds") is not None:
            details.append(f"{record['duration_seconds']}s")
        suffix = f" ({', '.join(details)})" if details else ""
        parts.append(
            "`{name}`: {status}{suffix}<br>`{command}`".format(
                name=record["name"],
                status=record["status"],
                suffix=suffix,
                command=_format_command(record.get("command", [])),
            )
        )
    return "<br>".join(parts)


def _format_artifacts(artifacts: list[dict[str, object]]) -> str:
    if not artifacts:
        return "-"
    parts = []
    for artifact in artifacts:
        if not artifact["present"]:
            suffix = "missing"
        elif artifact["kind"] == "directory":
            suffix = f"present ({artifact['entries']} entries)"
        else:
            suffix = f"present ({artifact['bytes']} bytes)"
        parts.append(f"`{artifact['path']}`: {suffix}")
    return "<br>".join(parts)


def _format_workflows(checks: list[dict[str, object]]) -> str:
    parts = []
    for check in checks:
        suffix = "configured" if check["present"] else "missing"
        parts.append(f"`{check['path']}`: {suffix}<br>`{check['token']}`")
    return "<br>".join(parts)


def _format_payloads(payloads: list[str]) -> str:
    if not payloads:
        return "-"
    return "<br>".join(f"`{payload}`" for payload in payloads)


def _format_history(window: dict[str, object]) -> str:
    count = window.get("history_successful_run_count", 0)
    minimum = window.get("history_minimum_required", 0)
    status = window.get("history_status", "missing")
    parts = [f"`{status}`", f"`{count}/{minimum}`"]
    latest_url = window.get("history_latest_run_url")
    if latest_url:
        parts.append(f"[latest]({latest_url})")
    next_action = window.get("history_next_action")
    if next_action and status != "satisfied":
        parts.append(str(next_action))
    return "<br>".join(parts)


def _format_flags(flags: dict[str, object]) -> str:
    if not flags:
        return "-"
    return ", ".join(f"`{key}`={value}" for key, value in sorted(flags.items()))


def render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Release Evidence Dashboard",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Preflight report: `{payload['preflight_path']}`",
        f"- Preflight status: `{payload['preflight_status']}`",
        f"- Dry run: `{payload['preflight_dry_run']}`",
        f"- Optional gate flags: {_format_flags(payload.get('preflight_flags', {}))}",
        "",
        "| Gate | Status | Local Evidence | Artifacts | CI / Workflow Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for gate in payload["gates"]:
        lines.append(
            "| {title} | `{status}` | {steps} | {artifacts} | {workflows} |".format(
                title=gate["title"],
                status=gate["status"],
                steps=_format_steps(gate["step_records"]),
                artifacts=_format_artifacts(gate["artifacts"]),
                workflows=_format_workflows(gate["workflow_checks"]),
            )
        )
    lines.extend(
        [
            "",
            "## Trend Evidence",
            "",
            f"- Trend manifest: `{payload['trend_manifest_path']}`",
            f"- Trend history: `{payload['trend_history_path']}`",
            f"- Minimum history for top-tier release: `{payload['trend_minimum_history_for_top_tier_release']}`",
            f"- All windows top-tier ready: `{payload['trend_history_all_windows_top_tier_ready']}`",
            "",
            "| Window | Cadence | Artifact | Retention | Payloads | Workflow Evidence | Historical Evidence |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    if not payload["trend_windows"]:
        lines.append("| - | - | - | - | - | missing | missing |")
    for window in payload["trend_windows"]:
        lines.append(
            "| {title} | {cadence} | `{artifact}` | {retention} days | {payloads} | {workflows} | {history} |".format(
                title=window["title"],
                cadence=window["cadence"] or "-",
                artifact=window["artifact_pattern"] or "-",
                retention=window["retention_days"] if window["retention_days"] is not None else "-",
                payloads=_format_payloads(window["required_payloads"]),
                workflows=_format_workflows(window["workflow_checks"]),
                history=_format_history(window),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_dashboard(payload: dict[str, object], markdown_path: Path, json_path: Path | None = None) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote release dashboard: {markdown_path}")
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote release dashboard JSON: {json_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a concise release evidence dashboard.")
    parser.add_argument("--preflight", default=str(DEFAULT_PREFLIGHT), help="Path to release_preflight.json.")
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN), help="Path to write the Markdown dashboard.")
    parser.add_argument("--json", default=str(DEFAULT_JSON), help="Path to write the JSON dashboard.")
    parser.add_argument("--no-json", action="store_true", help="Only write the Markdown dashboard.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_dashboard(Path(args.preflight))
    write_dashboard(payload, Path(args.markdown), None if args.no_json else Path(args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
