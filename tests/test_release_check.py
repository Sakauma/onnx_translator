# /**
#   ******************************************************************************
#   * @file        test_release_check.py
#   * @author      Egor Izmaylov
#   * @brief       Covers release readiness static configuration checks.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from tools.release_check import (
    C_BACKEND_MAX_SHARD_LINES,
    REQUIRED_FILES,
    REQUIRED_MAKE_TARGETS,
    REQUIRED_SCRIPTS,
    _check_c_backend_shard_budgets,
    _check_cibuildwheel_config,
    _check_heavy_gate_artifact_retention,
    _check_release_evidence_checklist,
    _check_release_evidence_workflow,
    _check_release_trend_history,
    _check_release_trend_manifest,
)


def _pyproject_with_cibuildwheel():
    return {
        "tool": {
            "cibuildwheel": {
                "build": "cp310-manylinux_x86_64 cp311-manylinux_x86_64 cp312-manylinux_x86_64",
                "skip": "*musllinux*",
                "linux": {
                    "archs": ["x86_64"],
                    "manylinux-x86_64-image": "manylinux2014",
                },
                "before-build": (
                    "rm -rf build dist onnx_translator.egg-info result onnx_model cache wheelhouse "
                    "tensor_ops.so tensor_ops_asan.so nn/tensor_ops.so"
                ),
            }
        }
    }


def test_cibuildwheel_release_check_accepts_manylinux_runtime_matrix():
    assert _check_cibuildwheel_config(_pyproject_with_cibuildwheel(), "cibuildwheel>=2.20") == []


def test_cibuildwheel_release_check_requires_full_python_matrix_and_external_smoke():
    pyproject = _pyproject_with_cibuildwheel()
    pyproject["tool"]["cibuildwheel"]["build"] = "cp312-manylinux_x86_64"
    pyproject["tool"]["cibuildwheel"]["test-command"] = "python -c \"pass\""
    pyproject["tool"]["cibuildwheel"]["before-build"] = "rm -rf tensor_ops.so"

    failures = _check_cibuildwheel_config(pyproject, "")

    assert "cibuildwheel.build is missing: cp310-manylinux_x86_64, cp311-manylinux_x86_64" in failures
    assert "cibuildwheel.test-command must stay unset; use wheelhouse-smoke after build instead" in failures
    assert "cibuildwheel.before-build must clean build" in failures
    assert "cibuildwheel.before-build must clean onnx_translator.egg-info" in failures
    assert "cibuildwheel.before-build must clean tensor_ops_asan.so" in failures
    assert "requirements-dev.txt must include cibuildwheel" in failures


def test_release_check_requires_full_manylinux_make_targets():
    assert "manylinux-wheels-full:" in REQUIRED_MAKE_TARGETS
    assert "manylinux-wheelhouse-check-full:" in REQUIRED_MAKE_TARGETS


def test_release_check_requires_fixed_runner_performance_target():
    assert "benchmark-fixed-runner-check:" in REQUIRED_MAKE_TARGETS


def test_release_check_requires_full_cuda_target():
    assert "verify-cuda-full:" in REQUIRED_MAKE_TARGETS


def test_release_check_requires_onnx_semantic_matrix_gate():
    assert "onnx-translator-onnx-semantic-matrix" in REQUIRED_SCRIPTS
    assert "onnx-semantic-matrix:" in REQUIRED_MAKE_TARGETS
    assert "tools/onnx_semantic_matrix.py" in REQUIRED_FILES
    assert "docs/onnx_semantic_matrix.json" in REQUIRED_FILES
    assert "docs/onnx_semantic_matrix.md" in REQUIRED_FILES


def test_release_check_requires_release_dashboard_gate():
    assert "onnx-translator-release-dashboard" in REQUIRED_SCRIPTS
    assert "release-dashboard:" in REQUIRED_MAKE_TARGETS
    assert "tools/release_dashboard.py" in REQUIRED_FILES


def test_release_check_requires_release_evidence_checklist():
    assert "docs/release_evidence_checklist.md" in REQUIRED_FILES
    assert _check_release_evidence_checklist() == []


def test_release_check_requires_ci_release_evidence_dashboard_artifact():
    assert _check_release_evidence_workflow() == []


def test_release_check_requires_release_trend_manifest():
    assert "docs/release_trend_manifest.json" in REQUIRED_FILES
    assert _check_release_trend_manifest() == []


def test_release_check_requires_release_trend_history():
    assert "onnx-translator-release-trend-history" in REQUIRED_SCRIPTS
    assert "release-trend-history:" in REQUIRED_MAKE_TARGETS
    assert "release-trend-history-refresh:" in REQUIRED_MAKE_TARGETS
    assert "tools/release_trend_history.py" in REQUIRED_FILES
    assert "docs/release_trend_history.json" in REQUIRED_FILES
    assert _check_release_trend_history() == []


def test_release_check_requires_heavy_gate_artifact_retention():
    assert _check_heavy_gate_artifact_retention() == []


def test_release_check_requires_split_audit_data_module():
    assert "tools/audit_operator_data.py" in REQUIRED_FILES


def test_release_check_requires_split_dtype_header():
    assert "tensor_ops/tensor_ops_dtype.h" in REQUIRED_FILES


def test_release_check_requires_c_backend_shard_budget():
    assert C_BACKEND_MAX_SHARD_LINES == 600
    for path in [
        "tensor_ops/tensor_ops_activation_extra.c",
        "tensor_ops/tensor_ops_compare_logic.c",
        "tensor_ops/tensor_ops_detection_sampling.c",
        "tensor_ops/tensor_ops_global_pool.c",
        "tensor_ops/tensor_ops_layout_sequence.c",
        "tensor_ops/tensor_ops_loss.c",
        "tensor_ops/tensor_ops_pool_roi.c",
        "tensor_ops/tensor_ops_recurrent.c",
        "tensor_ops/tensor_ops_shape_grid.c",
        "tensor_ops/tensor_ops_softmax_family.c",
    ]:
        assert path in REQUIRED_FILES
    assert _check_c_backend_shard_budgets() == []
