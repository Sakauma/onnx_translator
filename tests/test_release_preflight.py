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


def test_release_preflight_dry_run_writes_machine_readable_report(tmp_path):
    report = tmp_path / "preflight.json"

    assert main(["--dry-run", "--json", str(report)]) == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["dry_run"] is True
    assert payload["status"] == "passed"
    assert payload["steps"][0]["name"] == "onnx-semantic-matrix"
    assert all(step["status"] == "skipped" for step in payload["steps"])


def test_release_preflight_dry_run_reports_full_manylinux_flag(tmp_path):
    report = tmp_path / "preflight_manylinux_full.json"

    assert main(["--dry-run", "--include-manylinux-full", "--json", str(report)]) == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["include_manylinux_full"] is True
    assert payload["steps"][-2]["name"] == "manylinux-wheels-full"
    assert payload["steps"][-1]["name"] == "manylinux-wheelhouse-check-full"
