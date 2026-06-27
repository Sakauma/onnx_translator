# /**
#   ******************************************************************************
#   * @file        test_benchmark_runtime.py
#   * @author      Egor Izmaylov
#   * @brief       覆盖性能基线读写和阈值解析，防止性能门禁退化。
#   * @details     2026.06.27  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import argparse
import json
from pathlib import Path

import pytest

from tools.benchmark_runtime import (
    BENCHMARKS,
    SMOKE_MIN_THROUGHPUT,
    _build_payload,
    _effective_thresholds,
    _load_baseline,
    _load_baseline_payload,
    _parse_min_throughput,
)


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_baseline_loader_accepts_payload_and_legacy_list(tmp_path):
    result = {
        "name": "add",
        "repeat": 1,
        "warmup": 0,
        "output_elements": 16,
        "median_ms": 1.0,
        "min_ms": 1.0,
        "max_ms": 1.0,
        "elements_per_second": 16000.0,
    }
    args = argparse.Namespace(repeat=1, warmup=0, seed=123)
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_build_payload([result], args, {"add": 1000.0})), encoding="utf-8")
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps([result]), encoding="utf-8")

    assert _load_baseline(payload_path)["add"]["elements_per_second"] == 16000.0
    assert _load_baseline(legacy_path)["add"]["median_ms"] == 1.0
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["thresholds"] == {"add": 1000.0}
    assert payload["smoke"] is False


def test_min_throughput_parser_rejects_unknown_or_malformed_ops():
    assert _parse_min_throughput(["add=1000"]) == {"add": 1000.0}
    with pytest.raises(ValueError):
        _parse_min_throughput(["add"])
    with pytest.raises(ValueError):
        _parse_min_throughput(["unknown=1"])


def test_smoke_thresholds_cover_default_benchmarks_and_allow_overrides():
    thresholds = _effective_thresholds(True, ["add=42"])

    assert set(BENCHMARKS) <= set(thresholds)
    assert thresholds["add"] == 42.0
    assert _effective_thresholds(False, []) == {}


def test_benchmark_payload_marks_smoke_runs():
    args = argparse.Namespace(repeat=3, warmup=1, seed=123, smoke=True)
    result = {
        "name": "add",
        "repeat": 3,
        "warmup": 1,
        "output_elements": 16,
        "median_ms": 1.0,
        "min_ms": 0.9,
        "max_ms": 1.1,
        "elements_per_second": 16000.0,
    }

    payload = _build_payload([result], args, {"add": 1000.0})

    assert payload["smoke"] is True
    assert payload["thresholds"] == {"add": 1000.0}
    assert payload["runtime_library"]


def test_benchmark_payload_records_fixed_runner_metadata():
    args = argparse.Namespace(repeat=20, warmup=5, seed=123, smoke=False, runner_id="perf-a", baseline_kind="fixed_runner")
    result = {
        "name": "add",
        "repeat": 20,
        "warmup": 5,
        "output_elements": 16,
        "median_ms": 1.0,
        "min_ms": 0.9,
        "max_ms": 1.1,
        "elements_per_second": 16000.0,
    }

    payload = _build_payload([result], args)

    assert payload["baseline_kind"] == "fixed_runner"
    assert payload["runner_id"] == "perf-a"
    assert payload["machine"]["architecture"]


def test_benchmark_baseline_payload_loader_preserves_runner_metadata(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "baseline_kind": "fixed_runner",
                "runner_id": "perf-a",
                "benchmarks": [
                    {
                        "name": "add",
                        "repeat": 20,
                        "warmup": 5,
                        "output_elements": 16,
                        "median_ms": 1.0,
                        "min_ms": 1.0,
                        "max_ms": 1.0,
                        "elements_per_second": 16000.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = _load_baseline_payload(baseline_path)

    assert payload["baseline_kind"] == "fixed_runner"
    assert payload["runner_id"] == "perf-a"
    assert _load_baseline(baseline_path)["add"]["elements_per_second"] == 16000.0


def test_versioned_performance_baseline_covers_default_benchmarks():
    baseline = _load_baseline(ROOT / "docs" / "performance_baseline.json")

    assert set(baseline) == set(BENCHMARKS)
    for name, case in BENCHMARKS.items():
        assert baseline[name]["output_elements"] == case.elements
        assert baseline[name]["elements_per_second"] > 0.0
        assert baseline[name]["elements_per_second"] <= SMOKE_MIN_THROUGHPUT[name]
