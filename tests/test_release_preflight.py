# /**
#   ******************************************************************************
#   * @file        test_release_preflight.py
#   * @author      Egor Izmaylov
#   * @brief       Covers the aggregate release preflight gate planner and report writer.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import json

from tools.release_preflight import build_steps, main


def test_build_steps_includes_release_runtime_and_safety_gates():
    steps = build_steps("/env/bin/python")
    names = [step.name for step in steps]

    assert names == [
        "onnx-semantic-matrix",
        "release-check",
        "unit-tests",
        "model-smoke",
        "benchmark-smoke-report",
        "benchmark-baseline-check",
        "package-smoke",
        "release-artifacts",
        "sanitize",
    ]
    assert steps[0].command == ["make", "PYTHON=/env/bin/python", "onnx-semantic-matrix"]
    assert steps[1].command == ["make", "PYTHON=/env/bin/python", "release-check"]
    assert all("verify-cuda-smoke" not in step.command for step in steps)


def test_build_steps_can_include_cuda_smoke_gate():
    steps = build_steps("/env/bin/python", include_cuda_smoke=True)

    assert steps[-1].name == "verify-cuda-smoke"
    assert steps[-1].command == ["make", "PYTHON=/env/bin/python", "verify-cuda-smoke"]


def test_build_steps_can_include_full_cuda_gate():
    steps = build_steps("/env/bin/python", include_cuda_full=True)

    assert steps[-1].name == "verify-cuda-full"
    assert steps[-1].command == ["make", "PYTHON=/env/bin/python", "verify-cuda-full"]


def test_build_steps_can_include_manylinux_wheel_gate():
    steps = build_steps("/env/bin/python", include_manylinux=True)
    names = [step.name for step in steps]

    assert "manylinux-wheels" in names
    assert "manylinux-wheelhouse-check" in names
    assert steps[-2].command == ["make", "PYTHON=/env/bin/python", "manylinux-wheels"]
    assert steps[-1].command == ["make", "PYTHON=/env/bin/python", "manylinux-wheelhouse-check"]


def test_build_steps_can_include_full_manylinux_wheel_matrix_gate():
    steps = build_steps("/env/bin/python", include_manylinux_full=True)

    assert steps[-2].name == "manylinux-wheels-full"
    assert steps[-2].command == ["make", "PYTHON=/env/bin/python", "manylinux-wheels-full"]
    assert steps[-1].name == "manylinux-wheelhouse-check-full"
    assert steps[-1].command == ["make", "PYTHON=/env/bin/python", "manylinux-wheelhouse-check-full"]


def test_build_steps_can_include_fixed_runner_performance_gate():
    steps = build_steps("/env/bin/python", include_fixed_runner_perf=True)

    assert steps[-1].name == "benchmark-fixed-runner-check"
    assert steps[-1].command == ["make", "PYTHON=/env/bin/python", "benchmark-fixed-runner-check"]


def test_release_preflight_dry_run_writes_machine_readable_report(tmp_path):
    report = tmp_path / "preflight.json"

    assert main(["--dry-run", "--json", str(report)]) == 0

    dashboard = tmp_path / "release_dashboard.md"
    dashboard_json = tmp_path / "release_dashboard.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["dry_run"] is True
    assert payload["status"] == "passed"
    assert payload["steps"][0]["name"] == "onnx-semantic-matrix"
    assert all(step["status"] == "skipped" for step in payload["steps"])
    assert dashboard.exists()
    assert dashboard_json.exists()
    assert "Release Evidence Dashboard" in dashboard.read_text(encoding="utf-8")


def test_release_preflight_dry_run_reports_full_manylinux_flag(tmp_path):
    report = tmp_path / "preflight_manylinux_full.json"

    assert main(["--dry-run", "--include-manylinux-full", "--json", str(report)]) == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["include_manylinux_full"] is True
    assert payload["steps"][-2]["name"] == "manylinux-wheels-full"
    assert payload["steps"][-1]["name"] == "manylinux-wheelhouse-check-full"


def test_release_preflight_dry_run_reports_full_cuda_flag(tmp_path):
    report = tmp_path / "preflight_cuda_full.json"

    assert main(["--dry-run", "--include-cuda-full", "--json", str(report)]) == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["include_cuda_full"] is True
    assert payload["steps"][-1]["name"] == "verify-cuda-full"
    assert payload["steps"][-1]["command"] == ["make", f"PYTHON={payload['python']}", "verify-cuda-full"]


def test_release_preflight_dry_run_reports_fixed_runner_perf_flag(tmp_path):
    report = tmp_path / "preflight_fixed_perf.json"

    assert main(["--dry-run", "--include-fixed-runner-perf", "--json", str(report)]) == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["include_fixed_runner_perf"] is True
    assert payload["steps"][-1]["name"] == "benchmark-fixed-runner-check"
    assert payload["steps"][-1]["command"] == ["make", f"PYTHON={payload['python']}", "benchmark-fixed-runner-check"]


def test_release_preflight_can_skip_dashboard(tmp_path):
    report = tmp_path / "preflight.json"

    assert main(["--dry-run", "--json", str(report), "--no-dashboard"]) == 0

    assert report.exists()
    assert not (tmp_path / "release_dashboard.md").exists()
    assert not (tmp_path / "release_dashboard.json").exists()
