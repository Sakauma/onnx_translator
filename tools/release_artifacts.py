# /**
#   ******************************************************************************
#   * @file        release_artifacts.py
#   * @author      Egor Izmaylov
#   * @brief       Builds and verifies release source and wheel artifacts.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.package_smoke import _assert_runtime_wheel, _find_single_wheel, _run, _smoke_script

REQUIRED_SDIST_PATHS = {
    "Makefile",
    "MANIFEST.in",
    "pyproject.toml",
    "setup.py",
    "tensor_ops/tensor_ops.h",
    "tensor_ops/tensor_ops_internal.h",
    "tensor_ops/tensor_ops_dtype.h",
    "tensor_ops/internal/trig.h",
    "docs/abi_manifest.json",
    "docs/performance_baseline.json",
    "docs/release_evidence_checklist.md",
    "tools/audit_operator_data.py",
    "tools/model_suite.py",
    "tools/numerical/runner_config.py",
    "tools/package_smoke.py",
    "tools/wheelhouse_smoke.py",
    "cuda/verify_add.cu",
    "docs/release.md",
}


def _require_module(module_name: str) -> None:
    import importlib.util

    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(f"required release build module is missing: {module_name}; install requirements-dev.txt")


def _find_single_sdist(dist_dir: Path) -> Path:
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(sdists) != 1:
        raise RuntimeError(f"expected exactly one sdist in {dist_dir}, found {len(sdists)}")
    return sdists[0]


def _strip_sdist_root(member_name: str) -> str:
    parts = Path(member_name).parts
    if len(parts) <= 1:
        return ""
    return str(Path(*parts[1:])).replace("\\", "/")


def inspect_sdist(sdist_path: Path) -> set[str]:
    with tarfile.open(sdist_path, "r:gz") as archive:
        names = {_strip_sdist_root(member.name) for member in archive.getmembers() if member.isfile()}
    missing = sorted(REQUIRED_SDIST_PATHS - names)
    if missing:
        raise RuntimeError("sdist is missing required release files: " + ", ".join(missing))
    if not any(name.startswith("tensor_ops/") and name.endswith(".c") for name in names):
        raise RuntimeError("sdist is missing tensor_ops C sources")
    if not any(name.startswith("cuda/") and name.endswith(".cu") for name in names):
        raise RuntimeError("sdist is missing CUDA verifier sources")
    return names


def _build_artifacts(dist_dir: Path) -> None:
    _require_module("build")
    _run([sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(dist_dir)], cwd=ROOT)


def _check_artifacts(dist_dir: Path) -> None:
    _require_module("twine")
    artifacts = sorted(str(path) for path in dist_dir.glob("*"))
    if not artifacts:
        raise RuntimeError(f"no release artifacts found in {dist_dir}")
    _run([sys.executable, "-m", "twine", "check", *artifacts], cwd=ROOT)


def _build_wheel_from_sdist(sdist_path: Path, wheel_dir: Path) -> Path:
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(sdist_path),
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ],
        cwd=ROOT,
    )
    return _find_single_wheel(wheel_dir)


def _install_and_run_smoke(wheel: Path, install_dir: Path, smoke_cwd: Path) -> None:
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
    smoke_cwd.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-c", _smoke_script()], cwd=smoke_cwd, env=env)


def run_release_artifact_smoke(keep_artifacts: bool = False) -> None:
    temp_context = tempfile.TemporaryDirectory(prefix="onnx_translator_release_")
    temp_dir = Path(temp_context.name)
    try:
        dist_dir = temp_dir / "dist"
        sdist_wheel_dir = temp_dir / "sdist_wheel"
        install_dir = temp_dir / "install"
        smoke_cwd = temp_dir / "cwd"

        _build_artifacts(dist_dir)
        _assert_runtime_wheel(_find_single_wheel(dist_dir))
        _check_artifacts(dist_dir)
        sdist = _find_single_sdist(dist_dir)
        inspect_sdist(sdist)
        wheel_from_sdist = _build_wheel_from_sdist(sdist, sdist_wheel_dir)
        _assert_runtime_wheel(wheel_from_sdist)
        _install_and_run_smoke(wheel_from_sdist, install_dir, smoke_cwd)

        if keep_artifacts:
            output_dir = ROOT / "result" / "release_artifacts"
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.copytree(temp_dir, output_dir)
            print(f"Kept release artifacts: {output_dir}")
    finally:
        temp_context.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and verify release source/wheel artifacts.")
    parser.add_argument("--keep-artifacts", action="store_true", help="Copy release artifacts to result/release_artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_release_artifact_smoke(keep_artifacts=args.keep_artifacts)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        return exc.returncode
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
