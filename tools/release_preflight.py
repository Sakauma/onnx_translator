# /**
#   ******************************************************************************
#   * @file        release_preflight.py
#   * @author      Egor Izmaylov
#   * @brief       Runs the aggregate release preflight gate and writes a JSON report.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "result" / "release_preflight.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.release_dashboard import build_dashboard, write_dashboard


@dataclass(frozen=True)
class PreflightStep:
    name: str
    command: list[str]


def build_steps(
    python_executable: str,
    include_cuda_smoke: bool = False,
    include_cuda_full: bool = False,
    include_manylinux: bool = False,
    include_manylinux_full: bool = False,
    include_fixed_runner_perf: bool = False,
) -> list[PreflightStep]:
    make_python = f"PYTHON={python_executable}"
    steps = [
        PreflightStep("onnx-semantic-matrix", ["make", make_python, "onnx-semantic-matrix"]),
        PreflightStep("release-check", ["make", make_python, "release-check"]),
        PreflightStep("unit-tests", [python_executable, "-m", "pytest", "-q", "tests"]),
        PreflightStep("model-smoke", ["make", make_python, "model-smoke"]),
        PreflightStep("benchmark-smoke-report", ["make", make_python, "benchmark-smoke-report"]),
        PreflightStep("benchmark-baseline-check", ["make", make_python, "benchmark-baseline-check"]),
        PreflightStep("package-smoke", ["make", make_python, "package-smoke"]),
        PreflightStep("release-artifacts", ["make", make_python, "release-artifacts"]),
        PreflightStep("sanitize", ["make", make_python, "sanitize"]),
    ]
    if include_manylinux:
        steps.extend(
            [
                PreflightStep("manylinux-wheels", ["make", make_python, "manylinux-wheels"]),
                PreflightStep("manylinux-wheelhouse-check", ["make", make_python, "manylinux-wheelhouse-check"]),
            ]
        )
    if include_manylinux_full:
        steps.extend(
            [
                PreflightStep("manylinux-wheels-full", ["make", make_python, "manylinux-wheels-full"]),
                PreflightStep(
                    "manylinux-wheelhouse-check-full",
                    ["make", make_python, "manylinux-wheelhouse-check-full"],
                ),
            ]
        )
    if include_fixed_runner_perf:
        steps.append(PreflightStep("benchmark-fixed-runner-check", ["make", make_python, "benchmark-fixed-runner-check"]))
    if include_cuda_smoke:
        steps.append(PreflightStep("verify-cuda-smoke", ["make", make_python, "verify-cuda-smoke"]))
    if include_cuda_full:
        steps.append(PreflightStep("verify-cuda-full", ["make", make_python, "verify-cuda-full"]))
    return steps


def run_step(step: PreflightStep, dry_run: bool = False) -> dict[str, object]:
    print(f"\n==> {step.name}: {' '.join(step.command)}", flush=True)
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    if dry_run:
        return {
            "name": step.name,
            "command": step.command,
            "status": "skipped",
            "returncode": 0,
            "started_at": started_at.isoformat(),
            "duration_seconds": 0.0,
        }

    result = subprocess.run(step.command, cwd=ROOT, check=False)
    duration = time.perf_counter() - start
    return {
        "name": step.name,
        "command": step.command,
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "started_at": started_at.isoformat(),
        "duration_seconds": round(duration, 3),
    }


def write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote release preflight report: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the aggregate release preflight gate.")
    parser.add_argument("--json", default=str(DEFAULT_REPORT), help="Path for the machine-readable preflight report.")
    parser.add_argument("--include-cuda-smoke", action="store_true", help="Also run the CUDA smoke verification gate.")
    parser.add_argument("--include-cuda-full", action="store_true", help="Also run the full CUDA numerical verification gate.")
    parser.add_argument("--include-manylinux", action="store_true", help="Also build and inspect manylinux wheels with cibuildwheel.")
    parser.add_argument(
        "--include-manylinux-full",
        action="store_true",
        help="Also build and inspect the full cp310/cp311/cp312 manylinux wheel matrix.",
    )
    parser.add_argument(
        "--include-fixed-runner-perf",
        action="store_true",
        help="Also run the fixed-runner performance baseline gate.",
    )
    parser.add_argument("--keep-going", action="store_true", help="Continue running later gates after a failure.")
    parser.add_argument("--dry-run", action="store_true", help="Print and report the planned gates without executing them.")
    parser.add_argument("--no-dashboard", action="store_true", help="Do not write the release evidence dashboard.")
    parser.add_argument("--dashboard", help="Path for the Markdown release evidence dashboard.")
    parser.add_argument("--dashboard-json", help="Path for the JSON release evidence dashboard.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    steps = build_steps(
        sys.executable,
        include_cuda_smoke=args.include_cuda_smoke,
        include_cuda_full=args.include_cuda_full,
        include_manylinux=args.include_manylinux,
        include_manylinux_full=args.include_manylinux_full,
        include_fixed_runner_perf=args.include_fixed_runner_perf,
    )
    results = []
    failed = False
    for step in steps:
        result = run_step(step, dry_run=args.dry_run)
        results.append(result)
        if result["returncode"] != 0:
            failed = True
            if not args.keep_going:
                break

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "dry_run": args.dry_run,
        "include_cuda_smoke": args.include_cuda_smoke,
        "include_cuda_full": args.include_cuda_full,
        "include_manylinux": args.include_manylinux,
        "include_manylinux_full": args.include_manylinux_full,
        "include_fixed_runner_perf": args.include_fixed_runner_perf,
        "keep_going": args.keep_going,
        "status": "failed" if failed else "passed",
        "steps": results,
    }
    report_path = Path(args.json)
    write_report(report_path, payload)
    if not args.no_dashboard:
        dashboard_path = Path(args.dashboard) if args.dashboard else report_path.with_name("release_dashboard.md")
        dashboard_json_path = Path(args.dashboard_json) if args.dashboard_json else report_path.with_name("release_dashboard.json")
        write_dashboard(build_dashboard(report_path), dashboard_path, dashboard_json_path)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
