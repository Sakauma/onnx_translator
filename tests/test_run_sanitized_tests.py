# /**
#   ******************************************************************************
#   * @file        test_run_sanitized_tests.py
#   * @author      Egor Izmaylov
#   * @brief       Covers sanitizer gate orchestration.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import sys

from tools import run_sanitized_tests


def test_parse_args_keeps_pytest_args_and_skip_model_suite():
    args = run_sanitized_tests.parse_args(["--skip-model-suite", "--", "tests/test_operator_c_backend.py"])

    assert args.skip_model_suite is True
    assert args.pytest_args == ["tests/test_operator_c_backend.py"]


def test_sanitizer_default_runs_pytest_and_model_suite(monkeypatch, tmp_path):
    library = tmp_path / "tensor_ops_asan.so"
    library.write_bytes(b"asan")
    calls = []

    def fake_run(command, env):
        calls.append((command, env))
        return 0

    monkeypatch.setenv("ONNX_TRANSLATOR_SANITIZER_REEXEC", "1")
    monkeypatch.setattr(run_sanitized_tests, "_run", fake_run)

    assert run_sanitized_tests.main(["--library", str(library), "--", "tests/test_operator_c_backend.py"]) == 0
    assert calls[0][0] == [sys.executable, "-m", "pytest", "-q", "tests/test_operator_c_backend.py"]
    assert calls[1][0] == [sys.executable, "tools/model_suite.py"]
    assert calls[0][1]["TENSOR_OPS_LIB"] == str(library.resolve())


def test_sanitizer_can_skip_model_suite(monkeypatch, tmp_path):
    library = tmp_path / "tensor_ops_asan.so"
    library.write_bytes(b"asan")
    calls = []

    monkeypatch.setenv("ONNX_TRANSLATOR_SANITIZER_REEXEC", "1")
    monkeypatch.setattr(run_sanitized_tests, "_run", lambda command, env: calls.append(command) or 0)

    assert run_sanitized_tests.main(["--library", str(library), "--skip-model-suite"]) == 0
    assert calls == [[sys.executable, "-m", "pytest", "-q", *run_sanitized_tests.DEFAULT_TESTS]]
