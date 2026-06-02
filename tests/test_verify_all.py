import argparse
from pathlib import Path

from tools.verify_all import build_steps, cleanup_artifacts


# Egor Izmaylov: Function `_args` centralizes the args helper logic for the pytest verification suite, so edge-case normalization stays in one implementation boundary.
def _args(**overrides):
    values = {
        "skip_cuda": False,
        "skip_numerical": False,
        "iterations": 20,
        "op": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


# Egor Izmaylov: Function `test_skip_cuda_plan_omits_cuda_and_numerical_steps` locks down the test skip cuda plan omits cuda and numerical steps behavior in the pytest verification suite, covering regressions that could break ONNX import, runtime, or verification.
def test_skip_cuda_plan_omits_cuda_and_numerical_steps():
    steps = build_steps(_args(skip_cuda=True), Path("/repo"))
    names = [step.name for step in steps]

    assert "compile CUDA verifiers" not in names
    assert "run numerical correctness checks" not in names
    assert "--require-cuda" not in steps[0].command


# Egor Izmaylov: Function `test_full_plan_includes_numerical_filters` locks down the test full plan includes numerical filters behavior in the pytest verification suite, covering regressions that could break ONNX import, runtime, or verification.
def test_full_plan_includes_numerical_filters():
    steps = build_steps(_args(iterations=3, op=["add", "mul"]), Path("/repo"))
    numerical_step = next(step for step in steps if step.name == "run numerical correctness checks")

    assert numerical_step.command[-7:] == [
        "--iterations",
        "3",
        "--skip-plots",
        "--op",
        "add",
        "--op",
        "mul",
    ]


# Egor Izmaylov: Function `test_cleanup_artifacts_removes_known_generated_paths` locks down the test cleanup artifacts removes known generated paths behavior in the pytest verification suite, covering regressions that could break ONNX import, runtime, or verification.
def test_cleanup_artifacts_removes_known_generated_paths(tmp_path):
    (tmp_path / "cache").mkdir()
    (tmp_path / "onnx_model").mkdir()
    (tmp_path / "result").mkdir()
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "tensor_ops.so").write_bytes(b"compiled")

    cleanup_artifacts(tmp_path)

    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "onnx_model").exists()
    assert not (tmp_path / "result").exists()
    assert not (tmp_path / ".pytest_cache").exists()
    assert not (tmp_path / "pkg" / "__pycache__").exists()
    assert not (tmp_path / "tensor_ops.so").exists()
