# /**
#   ******************************************************************************
#   * @file        test_verify_all.py
#   * @author      Egor Izmaylov
#   * @brief       验证 tools.verify_all 的步骤编排、参数传递和清理逻辑。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import argparse
from pathlib import Path

from tools.verify_all import build_steps, cleanup_artifacts


# 封装 `_args` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
def _args(**overrides):
    values = {
        "skip_cuda": False,
        "skip_numerical": False,
        "skip_audit": False,
        "skip_model_suite": False,
        "keep_artifacts": False,
        "iterations": 20,
        "op": None,
        "force_cuda_compile": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


# 验证 `test_skip_cuda_plan_omits_cuda_and_numerical_steps` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_skip_cuda_plan_omits_cuda_and_numerical_steps():
    steps = build_steps(_args(skip_cuda=True), Path("/repo"))
    names = [step.name for step in steps]

    assert "compile CUDA verifiers" not in names
    assert "run numerical correctness checks" not in names
    assert "audit strict operator coverage" in names
    assert "--require-cuda" not in steps[0].command


# 验证 `test_full_plan_includes_numerical_filters` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_full_plan_includes_numerical_filters():
    steps = build_steps(_args(iterations=3, op=["add", "mul"]), Path("/repo"))
    numerical_step = next(step for step in steps if step.name == "run numerical correctness checks")
    compile_step = next(step for step in steps if step.name == "compile CUDA verifiers")
    audit_step = next(step for step in steps if step.name == "audit strict operator coverage")

    assert audit_step.command[-2:] == ["--strict", "--no-output"]
    assert compile_step.command[-4:] == ["--op", "add", "--op", "mul"]
    assert numerical_step.command[-7:] == [
        "--iterations",
        "3",
        "--skip-plots",
        "--op",
        "add",
        "--op",
        "mul",
    ]


def test_cuda_compile_step_can_force_rebuild():
    steps = build_steps(_args(force_cuda_compile=True, op=["add"]), Path("/repo"))
    compile_step = next(step for step in steps if step.name == "compile CUDA verifiers")

    assert compile_step.command[-3:] == ["--force", "--op", "add"]


# 验证 `test_skip_audit_plan_omits_strict_audit_step` 覆盖的回归场景，防止排障开关影响默认门禁。
def test_skip_audit_plan_omits_strict_audit_step():
    steps = build_steps(_args(skip_audit=True), Path("/repo"))
    names = [step.name for step in steps]

    assert "audit strict operator coverage" not in names


def test_default_plan_includes_representative_model_suite():
    steps = build_steps(_args(skip_cuda=True), Path("/repo"))
    model_step = next(step for step in steps if step.name == "run representative model suite")

    assert model_step.command[-1] == "tools/model_suite.py"


def test_model_suite_step_can_be_skipped_or_keep_artifacts():
    skipped = build_steps(_args(skip_cuda=True, skip_model_suite=True), Path("/repo"))
    kept = build_steps(_args(skip_cuda=True, keep_artifacts=True), Path("/repo"))

    assert "run representative model suite" not in [step.name for step in skipped]
    model_step = next(step for step in kept if step.name == "run representative model suite")
    assert model_step.command[-1] == "--keep-artifacts"


# 验证 `test_cleanup_artifacts_removes_known_generated_paths` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
def test_cleanup_artifacts_removes_known_generated_paths(tmp_path):
    (tmp_path / "cache").mkdir()
    (tmp_path / "onnx_model").mkdir()
    (tmp_path / "result").mkdir()
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "nn").mkdir()
    (tmp_path / "tensor_ops.so").write_bytes(b"compiled")
    (tmp_path / "tensor_ops_asan.so").write_bytes(b"asan")
    (tmp_path / "nn" / "tensor_ops.so").write_bytes(b"packaged")

    cleanup_artifacts(tmp_path)

    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "onnx_model").exists()
    assert not (tmp_path / "result").exists()
    assert not (tmp_path / ".pytest_cache").exists()
    assert not (tmp_path / "pkg" / "__pycache__").exists()
    assert not (tmp_path / "tensor_ops.so").exists()
    assert not (tmp_path / "tensor_ops_asan.so").exists()
    assert not (tmp_path / "nn" / "tensor_ops.so").exists()
