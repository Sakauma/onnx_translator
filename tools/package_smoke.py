# /**
#   ******************************************************************************
#   * @file        package_smoke.py
#   * @author      Egor Izmaylov
#   * @brief       Builds a wheel and verifies the installed package loads its packaged C runtime.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _find_single_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel in {dist_dir}, found {len(wheels)}")
    return wheels[0]


def _assert_runtime_wheel(wheel: Path) -> None:
    platform_tag = wheel.name.removesuffix(".whl").rsplit("-", 1)[-1]
    if platform_tag == "any":
        raise RuntimeError(f"runtime wheel must be platform-specific because it packages tensor_ops.so: {wheel.name}")

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        runtime_paths = [name for name in names if name == "nn/tensor_ops.so" or name.endswith("/nn/tensor_ops.so")]
        if not runtime_paths:
            raise RuntimeError(f"runtime wheel is missing packaged C backend: {wheel.name}")
        purelib_runtime_paths = [name for name in runtime_paths if ".data/purelib/" in name]
        if purelib_runtime_paths:
            raise RuntimeError(
                "runtime shared libraries must be installed from platlib, not purelib: "
                + ", ".join(sorted(purelib_runtime_paths))
            )
        wheel_metadata_name = next((name for name in names if name.endswith(".dist-info/WHEEL")), None)
        if wheel_metadata_name is None:
            raise RuntimeError(f"runtime wheel is missing WHEEL metadata: {wheel.name}")
        wheel_metadata = archive.read(wheel_metadata_name).decode("utf-8")
    if "Root-Is-Purelib: false" not in wheel_metadata:
        raise RuntimeError(f"runtime wheel must declare Root-Is-Purelib: false: {wheel.name}")


def _build_wheel(dist_dir: Path) -> None:
    if importlib.util.find_spec("build") is not None:
        _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)], cwd=ROOT)
        return
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(dist_dir),
        ],
        cwd=ROOT,
    )


def _smoke_script() -> str:
    return r"""
import os
import sys
from pathlib import Path

import numpy as np

import nn
from nn import Tensor
from nn.Operators import ADD

install_dir = Path(os.environ["ONNX_TRANSLATOR_PACKAGE_INSTALL_DIR"]).resolve()
runtime_path = Path(nn.TENSOR_OPS_LIB_PATH).resolve()
if install_dir not in runtime_path.parents:
    raise SystemExit(f"runtime library was not loaded from installed package: {runtime_path}")

repo_root = Path(os.environ["ONNX_TRANSLATOR_REPO_ROOT"]).resolve()
module_path = Path(nn.__file__).resolve()
if repo_root in module_path.parents:
    raise SystemExit(f"import unexpectedly resolved to repository source: {module_path}")

left = Tensor(2, 2, dtype="float32", data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
right = Tensor(2, 2, dtype="float32", data=np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))
actual = ADD(["left", "right"], ["out"], dtype="float32").forward(left, right)["tensor"].data
np.testing.assert_array_equal(actual, np.array([[11.0, 22.0], [33.0, 44.0]], dtype=np.float32))
print(f"package smoke passed; runtime={runtime_path}")
"""


def run_package_smoke(keep_artifacts: bool = False) -> None:
    temp_context = tempfile.TemporaryDirectory(prefix="onnx_translator_package_")
    temp_dir = Path(temp_context.name)
    try:
        dist_dir = temp_dir / "dist"
        install_dir = temp_dir / "install"
        smoke_cwd = temp_dir / "cwd"
        smoke_cwd.mkdir()

        _build_wheel(dist_dir)
        wheel = _find_single_wheel(dist_dir)
        _assert_runtime_wheel(wheel)
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                "--target",
                str(install_dir),
                str(wheel),
            ],
            cwd=ROOT,
        )

        env = os.environ.copy()
        env.pop("TENSOR_OPS_LIB", None)
        env["PYTHONPATH"] = str(install_dir)
        env["ONNX_TRANSLATOR_PACKAGE_INSTALL_DIR"] = str(install_dir)
        env["ONNX_TRANSLATOR_REPO_ROOT"] = str(ROOT)
        _run([sys.executable, "-c", _smoke_script()], cwd=smoke_cwd, env=env)
        if keep_artifacts:
            output_dir = ROOT / "result" / "package_smoke"
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.copytree(temp_dir, output_dir)
            print(f"Kept package smoke artifacts: {output_dir}")
    finally:
        temp_context.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a wheel and verify installed package runtime loading.")
    parser.add_argument("--keep-artifacts", action="store_true", help="Copy temporary build/install artifacts to result/package_smoke.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_package_smoke(keep_artifacts=args.keep_artifacts)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        return exc.returncode
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
