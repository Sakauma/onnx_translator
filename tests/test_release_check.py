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

from tools.release_check import REQUIRED_FILES, REQUIRED_MAKE_TARGETS, REQUIRED_SCRIPTS, _check_cibuildwheel_config


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


def test_release_check_requires_onnx_semantic_matrix_gate():
    assert "onnx-translator-onnx-semantic-matrix" in REQUIRED_SCRIPTS
    assert "onnx-semantic-matrix:" in REQUIRED_MAKE_TARGETS
    assert "tools/onnx_semantic_matrix.py" in REQUIRED_FILES
    assert "docs/onnx_semantic_matrix.json" in REQUIRED_FILES
    assert "docs/onnx_semantic_matrix.md" in REQUIRED_FILES
