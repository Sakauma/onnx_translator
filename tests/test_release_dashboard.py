# /**
#   ******************************************************************************
#   * @file        test_release_dashboard.py
#   * @author      Egor Izmaylov
#   * @brief       Covers release evidence dashboard aggregation.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import json

from tools.release_dashboard import build_dashboard, render_markdown


def test_release_dashboard_summarizes_preflight_and_workflow_evidence(tmp_path):
    preflight = tmp_path / "release_preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "status": "passed",
                "dry_run": False,
                "steps": [
                    {"name": "sanitize", "status": "passed"},
                    {"name": "manylinux-wheels", "status": "skipped"},
                    {"name": "manylinux-wheelhouse-check", "status": "skipped"},
                    {"name": "benchmark-smoke-report", "status": "passed"},
                    {"name": "benchmark-baseline-check", "status": "passed"},
                    {
                        "name": "verify-cuda-full",
                        "command": ["make", "PYTHON=/env/bin/python", "verify-cuda-full"],
                        "status": "skipped",
                        "returncode": 0,
                        "duration_seconds": 0.0,
                    },
                    {
                        "name": "benchmark-fixed-runner-check",
                        "command": ["make", "PYTHON=/env/bin/python", "benchmark-fixed-runner-check"],
                        "status": "skipped",
                        "returncode": 0,
                        "duration_seconds": 0.0,
                    },
                ],
                "include_cuda_full": True,
                "include_fixed_runner_perf": True,
            }
        ),
        encoding="utf-8",
    )

    payload = build_dashboard(preflight)
    gates = {gate["id"]: gate for gate in payload["gates"]}
    markdown = render_markdown(payload)

    assert payload["preflight_status"] == "passed"
    assert gates["sanitizer"]["status"] == "passed"
    assert gates["manylinux_smoke"]["status"] == "planned"
    assert gates["performance_smoke"]["status"] == "passed"
    assert gates["performance_fixed_runner"]["status"] == "planned"
    assert gates["cuda_full"]["status"] == "planned"
    assert gates["cuda_full"]["step_records"][0]["command"] == ["make", "PYTHON=/env/bin/python", "verify-cuda-full"]
    assert payload["preflight_flags"]["include_cuda_full"] is True
    assert payload["preflight_flags"]["include_fixed_runner_perf"] is True
    assert payload["trend_manifest_present"] is True
    assert payload["trend_minimum_history_for_top_tier_release"] == 3
    assert {window["id"] for window in payload["trend_windows"]} >= {
        "cuda_full",
        "fixed_runner_performance",
        "manylinux_full",
        "release_evidence",
    }
    assert "Full CUDA Numerical" in markdown
    assert "Trend Evidence" in markdown
    assert "Fixed runner performance trend" in markdown
    assert "make PYTHON=/env/bin/python verify-cuda-full" in markdown
    assert "make PYTHON=/env/bin/python benchmark-fixed-runner-check" in markdown
    assert "Optional gate flags" in markdown
