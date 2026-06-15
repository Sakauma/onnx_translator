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
        "iterations": 20,
        "op": None,
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
    audit_step = next(step for step in steps if step.name == "audit strict operator coverage")

    assert audit_step.command[-1] == "--strict"
    assert numerical_step.command[-7:] == [
        "--iterations",
        "3",
        "--skip-plots",
        "--op",
        "add",
        "--op",
        "mul",
    ]


# 验证 `test_skip_audit_plan_omits_strict_audit_step` 覆盖的回归场景，防止排障开关影响默认门禁。
def test_skip_audit_plan_omits_strict_audit_step():
    steps = build_steps(_args(skip_audit=True), Path("/repo"))
    names = [step.name for step in steps]

    assert "audit strict operator coverage" not in names


# 验证 `test_cleanup_artifacts_removes_known_generated_paths` 覆盖的回归场景，防止 ONNX 导入、图运行或算子实现被破坏。
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
