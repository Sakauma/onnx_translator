# /**
#   ******************************************************************************
#   * @file        test_release_trend_history.py
#   * @author      Egor Izmaylov
#   * @brief       Covers archived release trend history validation.
#   * @details     2026.06.28  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import json

from tools.release_trend_history import (
    DEFAULT_HISTORY,
    DEFAULT_MANIFEST,
    refresh_trend_history,
    validate_trend_history,
    validate_trend_history_payload,
)


class FakeActionsClient:
    repository = "Sakauma/onnx_translator"

    def list_workflows(self):
        return [
            {"id": 1, "path": ".github/workflows/ci.yml"},
            {"id": 2, "path": ".github/workflows/wheels.yml"},
        ]

    def list_workflow_runs(self, workflow_id, max_runs):
        runs = {
            1: [
                _run(101, "success"),
                _run(102, "success"),
                _run(103, "failure"),
            ],
            2: [
                _run(201, "success"),
                _run(202, "success"),
            ],
        }
        return runs[workflow_id][:max_runs]

    def list_run_artifacts(self, run_id):
        artifacts = {
            101: [_artifact(1001, "release-evidence-101")],
            102: [_artifact(1002, "release-evidence-102")],
            201: [
                _artifact(2001, "manylinux-cp310-manylinux_x86_64-201"),
                _artifact(2002, "manylinux-cp311-manylinux_x86_64-201"),
                _artifact(2003, "manylinux-cp312-manylinux_x86_64-201"),
            ],
            202: [
                _artifact(2021, "manylinux-cp312-manylinux_x86_64-202"),
            ],
        }
        return artifacts.get(run_id, [])


def _run(run_id, conclusion):
    return {
        "id": run_id,
        "html_url": f"https://github.com/Sakauma/onnx_translator/actions/runs/{run_id}",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "head_sha": f"{run_id:040d}"[-40:],
        "created_at": "2026-06-28T05:00:00Z",
        "status": "completed",
        "conclusion": conclusion,
    }


def _artifact(artifact_id, name):
    return {
        "id": artifact_id,
        "name": name,
        "url": f"https://api.github.com/repos/Sakauma/onnx_translator/actions/artifacts/{artifact_id}",
        "digest": f"sha256:{artifact_id:064d}"[-71:],
        "expires_at": "2026-09-26T05:00:00Z",
        "expired": False,
    }


def test_release_trend_history_default_snapshot_is_valid():
    summary, failures = validate_trend_history()

    assert failures == []
    windows = {window["id"]: window for window in summary["windows"]}
    assert windows["release_evidence"]["successful_run_count"] == 3
    assert windows["release_evidence"]["status"] == "satisfied"
    assert windows["cuda_full"]["status"] == "insufficient_history"
    assert summary["all_windows_top_tier_ready"] is False


def test_release_trend_history_rejects_inconsistent_counts(tmp_path):
    history = json.loads(DEFAULT_HISTORY.read_text(encoding="utf-8"))
    history["windows"][0]["successful_run_count"] = 99
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps(history), encoding="utf-8")

    _, failures = validate_trend_history(history_path, DEFAULT_MANIFEST)

    assert "trend window release_evidence successful_run_count must match successful_runs length" in failures


def test_release_trend_history_refresh_collects_matching_artifacts():
    manifest = {
        "minimum_history_for_top_tier_release": 2,
        "trend_windows": [
            {
                "id": "release_evidence",
                "source_workflow": ".github/workflows/ci.yml",
                "artifact_pattern": "release-evidence-${{ github.run_id }}",
                "workflow_tokens": [],
            },
            {
                "id": "manylinux_full",
                "source_workflow": ".github/workflows/wheels.yml",
                "artifact_pattern": "manylinux-${{ matrix.cibw-build }}-${{ github.run_id }}",
                "workflow_tokens": [
                    "cp310-manylinux_x86_64",
                    "cp311-manylinux_x86_64",
                    "cp312-manylinux_x86_64",
                ],
            },
        ],
    }

    history = refresh_trend_history(
        manifest,
        {"repository": "Sakauma/onnx_translator"},
        FakeActionsClient(),
        captured_at="2026-06-28T05:00:00Z",
    )
    summary, failures = validate_trend_history_payload(history, manifest)

    windows = {window["id"]: window for window in history["windows"]}
    assert failures == []
    assert windows["release_evidence"]["status"] == "satisfied"
    assert windows["release_evidence"]["successful_run_count"] == 2
    assert windows["manylinux_full"]["status"] == "insufficient_history"
    assert windows["manylinux_full"]["successful_run_count"] == 1
    assert summary["all_windows_top_tier_ready"] is False
