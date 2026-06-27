# /**
#   ******************************************************************************
#   * @file        run_sanitized_tests.py
#   * @author      Egor Izmaylov
#   * @brief       Re-executes pytest with ASan/UBSan preloaded for the C backend shared library.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTS = [
    "tests/test_operator_c_backend.py",
    "tests/test_graph_runtime.py",
]


def _gcc_runtime(name: str) -> str | None:
    result = subprocess.run(
        ["gcc", f"-print-file-name={name}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    path = result.stdout.strip()
    if path and path != name and Path(path).exists():
        return path
    return None


def _build_preload(existing: str | None) -> str:
    runtimes = [path for path in [_gcc_runtime("libasan.so"), _gcc_runtime("libubsan.so")] if path]
    if existing:
        runtimes.append(existing)
    return ":".join(runtimes)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pytest under sanitizer runtimes.")
    parser.add_argument(
        "--library",
        default=str(ROOT / "tensor_ops_asan.so"),
        help="Sanitized tensor_ops shared library path.",
    )
    parser.add_argument(
        "--skip-model-suite",
        action="store_true",
        help="Skip representative ONNX model smoke checks under sanitizer.",
    )
    args, pytest_args = parser.parse_known_args(argv)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    args.pytest_args = pytest_args
    return args


def _run(command: list[str], env: dict[str, str]) -> int:
    print("$ " + " ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT, env=env)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    library = Path(args.library).resolve()
    if not library.exists():
        print(f"ERROR: sanitized C backend library not found: {library}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["TENSOR_OPS_LIB"] = str(library)
    env.setdefault("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=1:allocator_may_return_null=1")
    env.setdefault("UBSAN_OPTIONS", "print_stacktrace=1:halt_on_error=1")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    if env.get("ONNX_TRANSLATOR_SANITIZER_REEXEC") != "1":
        env["ONNX_TRANSLATOR_SANITIZER_REEXEC"] = "1"
        preload = _build_preload(env.get("LD_PRELOAD"))
        if preload:
            env["LD_PRELOAD"] = preload
        script_args = ["--library", str(library)]
        if args.skip_model_suite:
            script_args.append("--skip-model-suite")
        print(f"Re-executing with LD_PRELOAD={env.get('LD_PRELOAD', '<unset>')}", flush=True)
        os.execvpe(
            sys.executable,
            [sys.executable, str(Path(__file__).resolve()), *script_args, *args.pytest_args],
            env,
        )

    pytest_args = args.pytest_args or DEFAULT_TESTS
    returncode = _run([sys.executable, "-m", "pytest", "-q", *pytest_args], env)
    if returncode != 0:
        print(f"sanitized pytest exited with code {returncode}", flush=True)
        return returncode

    if not args.skip_model_suite:
        returncode = _run([sys.executable, "tools/model_suite.py"], env)
        if returncode != 0:
            print(f"sanitized model suite exited with code {returncode}", flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
