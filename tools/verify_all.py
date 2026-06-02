"""文件功能：编排构建、图验证、审计和数值验证步骤，作为工程级验证入口。
作者：Egor Izmaylov
时间：2026-06-02
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    env: dict[str, str] | None = None


# 实现 `repo_root` 步骤，规范化输入并返回下游期望的数据或元信息。
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# 封装 `_python` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
def _python() -> str:
    return sys.executable


# 实现 `build_steps` 步骤，规范化输入并返回下游期望的数据或元信息。
def build_steps(args: argparse.Namespace, root: Path) -> list[Step]:
    health_command = [_python(), "tools/health_check.py"]
    if not args.skip_cuda:
        health_command.append("--require-cuda")

    steps = [
        Step("check environment", health_command),
    ]

    if not args.skip_cuda:
        steps.append(Step("compile CUDA verifiers", [_python(), "tools/cli.py", "compile-cuda"]))

    make_env = {"PYTHON": _python()}
    steps.extend(
        [
            Step("clean C backend", ["make", "clean"], env=make_env),
            Step("build and static-check C/Python", ["make", "check"], env=make_env),
            Step("run unit tests", [_python(), "-m", "pytest", "-q"]),
            Step("generate graph-ops ONNX model", [_python(), "tools/cli.py", "create-graph-model"]),
            Step(
                "verify graph-ops model",
                [
                    _python(),
                    "tools/cli.py",
                    "verify-graph",
                    "--model",
                    "./onnx_model/model.onnx",
                    "--task-name",
                    "nps_graph_ops",
                ],
            ),
            Step("generate PyTorch export ONNX model", [_python(), "tools/cli.py", "create-model"]),
            Step(
                "verify PyTorch export model",
                [
                    _python(),
                    "tools/cli.py",
                    "verify-graph",
                    "--model",
                    "./onnx_model/model.onnx",
                    "--task-name",
                    "nps_verification",
                ],
            ),
        ]
    )

    if not args.skip_cuda and not args.skip_numerical:
        numerical_command = [
            _python(),
            "tools/cli.py",
            "numerical",
            "--iterations",
            str(args.iterations),
            "--skip-plots",
        ]
        for op_name in args.op or []:
            numerical_command.extend(["--op", op_name])
        steps.append(Step("run numerical correctness checks", numerical_command))

    return steps


# 封装 `_assert_under_root` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
def _assert_under_root(root: Path, target: Path) -> None:
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    if target_resolved != root_resolved and root_resolved not in target_resolved.parents:
        raise RuntimeError(f"Refusing to remove path outside repository: {target_resolved}")


# 实现 `cleanup_artifacts` 步骤，规范化输入并返回下游期望的数据或元信息。
def cleanup_artifacts(root: Path) -> None:
    targets = [
        root / ".pytest_cache",
        root / "cache",
        root / "onnx_model",
        root / "result",
        root / "tensor_ops.so",
    ]
    targets.extend(root.rglob("__pycache__"))

    for target in targets:
        if not target.exists():
            continue
        _assert_under_root(root, target)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()


# 实现 `run_steps` 步骤，规范化输入并返回下游期望的数据或元信息。
def run_steps(steps: list[Step], root: Path) -> None:
    for idx, step in enumerate(steps, start=1):
        print(f"\n[{idx}/{len(steps)}] {step.name}", flush=True)
        print("$ " + " ".join(step.command), flush=True)
        env = os.environ.copy()
        if step.env:
            env.update(step.env)
        subprocess.run(step.command, cwd=root, env=env, check=True)


# 实现 `parse_args` 步骤，规范化输入并返回下游期望的数据或元信息。
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the project engineering verification gate.")
    parser.add_argument(
        "--skip-cuda",
        action="store_true",
        help="Skip CUDA verifier compilation and CUDA-backed numerical correctness checks.",
    )
    parser.add_argument(
        "--skip-numerical",
        action="store_true",
        help="Compile CUDA verifiers but skip numerical correctness checks.",
    )
    parser.add_argument("--iterations", type=int, default=20, help="Iterations per numerical test plan.")
    parser.add_argument("--op", action="append", help="Limit numerical checks to a named op. Can be repeated.")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep generated build and verification artifacts.")
    parser.add_argument("--no-clean-before", action="store_true", help="Do not remove generated artifacts before running.")
    return parser.parse_args(argv)


# 作为 `tools/verify_all.py` 的命令行入口，解析参数、调度检查流程并返回进程退出码。
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()

    if not args.no_clean_before:
        cleanup_artifacts(root)

    steps = build_steps(args, root)
    try:
        run_steps(steps, root)
    except subprocess.CalledProcessError as exc:
        print(f"\nERROR: step failed with exit code {exc.returncode}: {' '.join(exc.cmd)}")
        print("Generated artifacts were kept for debugging.")
        return exc.returncode

    if args.keep_artifacts:
        print("\nVerification passed; generated artifacts were kept.")
    else:
        cleanup_artifacts(root)
        print("\nVerification passed; generated artifacts were cleaned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
