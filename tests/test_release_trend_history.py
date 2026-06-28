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

from tools.release_trend_history import DEFAULT_HISTORY, DEFAULT_MANIFEST, validate_trend_history


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
