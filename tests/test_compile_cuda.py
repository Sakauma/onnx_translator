# /**
#   ******************************************************************************
#   * @file        test_compile_cuda.py
#   * @author      Egor Izmaylov
#   * @brief       Covers CUDA verifier compile script filtering and cache behavior.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compile_cuda_filters_ops_and_skips_fresh_cache(tmp_path):
    cuda_dir = tmp_path / "cuda"
    cache_dir = tmp_path / "cache"
    cuda_dir.mkdir()
    (cuda_dir / "verify_add.cu").write_text("// add\n", encoding="utf-8")
    (cuda_dir / "verify_mul.cu").write_text("// mul\n", encoding="utf-8")
    (cuda_dir / "common.cuh").write_text("// common\n", encoding="utf-8")
    nvcc = tmp_path / "fake_nvcc.sh"
    nvcc.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "--version" ]; then
  echo "fake nvcc 1"
  exit 0
fi
out=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then
    out="$2"
    shift 2
    continue
  fi
  shift
done
echo compiled > "$out"
chmod +x "$out"
""",
        encoding="utf-8",
    )
    nvcc.chmod(0o755)
    env = os.environ.copy()
    env.update({"CUDA_DIR": str(cuda_dir), "CACHE_DIR": str(cache_dir), "NVCC": str(nvcc)})
    script = ROOT / "tools" / "commands" / "compile_cuda.sh"

    first = subprocess.run(["bash", str(script), "--op", "add"], cwd=ROOT, env=env, text=True, capture_output=True)
    second = subprocess.run(["bash", str(script), "--op", "add"], cwd=ROOT, env=env, text=True, capture_output=True)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert (cache_dir / "verify_add").exists()
    assert not (cache_dir / "verify_mul").exists()
    assert "compiled=1 skipped=0" in first.stdout
    assert "compiled=0 skipped=1" in second.stdout


def test_compile_cuda_rebuilds_when_nvcc_identity_changes(tmp_path):
    cuda_dir = tmp_path / "cuda"
    cache_dir = tmp_path / "cache"
    cuda_dir.mkdir()
    (cuda_dir / "verify_add.cu").write_text("// add\n", encoding="utf-8")

    def write_fake_nvcc(path, marker):
        path.write_text(
            f"""#!/usr/bin/env bash
set -euo pipefail
if [ "${{1:-}}" = "--version" ]; then
  echo "fake nvcc {marker}"
  exit 0
fi
out=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; shift 2; continue; fi
  shift
done
printf '%s\n' '{marker}' > "$out"
chmod +x "$out"
""",
            encoding="utf-8",
        )
        path.chmod(0o755)

    nvcc_a = tmp_path / "nvcc_a"
    nvcc_b = tmp_path / "nvcc_b"
    write_fake_nvcc(nvcc_a, "A")
    write_fake_nvcc(nvcc_b, "B")
    script = ROOT / "tools" / "commands" / "compile_cuda.sh"
    env = os.environ.copy()
    env.update({"CUDA_DIR": str(cuda_dir), "CACHE_DIR": str(cache_dir), "NVCC": str(nvcc_a)})

    first = subprocess.run(["bash", str(script), "--op", "add"], cwd=ROOT, env=env, text=True, capture_output=True)
    env["NVCC"] = str(nvcc_b)
    second = subprocess.run(["bash", str(script), "--op", "add"], cwd=ROOT, env=env, text=True, capture_output=True)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "compiled=1 skipped=0" in first.stdout
    assert "compiled=1 skipped=0" in second.stdout
    assert (cache_dir / "verify_add").read_text(encoding="utf-8") == "B\n"
    assert not list(cache_dir.glob("verify_*.build-id"))


def test_compile_cuda_rebuilds_when_compiler_binary_changes_in_place(tmp_path):
    cuda_dir = tmp_path / "cuda"
    cache_dir = tmp_path / "cache"
    cuda_dir.mkdir()
    (cuda_dir / "verify_add.cu").write_text("// add\n", encoding="utf-8")
    nvcc = tmp_path / "nvcc"

    def write_compiler(marker):
        nvcc.write_text(
            f"""#!/usr/bin/env bash
set -euo pipefail
if [ "${{1:-}}" = "--version" ]; then echo "stable version"; exit 0; fi
out=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; shift 2; continue; fi
  shift
done
printf '%s\\n' '{marker}' > "$out"
chmod +x "$out"
""",
            encoding="utf-8",
        )
        nvcc.chmod(0o755)

    script = ROOT / "tools" / "commands" / "compile_cuda.sh"
    env = os.environ.copy()
    env.update({"CUDA_DIR": str(cuda_dir), "CACHE_DIR": str(cache_dir), "NVCC": str(nvcc)})
    write_compiler("A")
    first = subprocess.run(["bash", str(script), "--op", "add"], cwd=ROOT, env=env, text=True, capture_output=True)
    write_compiler("B")
    second = subprocess.run(["bash", str(script), "--op", "add"], cwd=ROOT, env=env, text=True, capture_output=True)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "compiled=1 skipped=0" in second.stdout
    assert (cache_dir / "verify_add").read_text(encoding="utf-8") == "B\n"


def test_failed_recompile_invalidates_identity_and_next_run_retries(tmp_path):
    cuda_dir = tmp_path / "cuda"
    cache_dir = tmp_path / "cache"
    cuda_dir.mkdir()
    (cuda_dir / "verify_add.cu").write_text("// add\n", encoding="utf-8")
    nvcc = tmp_path / "nvcc"
    nvcc.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "--version" ]; then echo "fake nvcc stable"; exit 0; fi
out=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; shift 2; continue; fi
  shift
done
printf '%s\n' "${MARKER:-OK}" > "$out"
chmod +x "$out"
if [ "${FAIL_COMPILE:-0}" = "1" ]; then exit 9; fi
""",
        encoding="utf-8",
    )
    nvcc.chmod(0o755)
    script = ROOT / "tools" / "commands" / "compile_cuda.sh"
    env = os.environ.copy()
    env.update({"CUDA_DIR": str(cuda_dir), "CACHE_DIR": str(cache_dir), "NVCC": str(nvcc), "MARKER": "GOOD"})

    first = subprocess.run(["bash", str(script), "--op", "add"], cwd=ROOT, env=env, text=True, capture_output=True)
    env.update({"MARKER": "BROKEN", "FAIL_COMPILE": "1"})
    failed = subprocess.run(["bash", str(script), "--op", "add", "--force"], cwd=ROOT, env=env, text=True, capture_output=True)
    assert (cache_dir / "verify_add").read_text(encoding="utf-8") == "BROKEN\n"
    env.update({"MARKER": "RECOVERED", "FAIL_COMPILE": "0"})
    retry = subprocess.run(["bash", str(script), "--op", "add"], cwd=ROOT, env=env, text=True, capture_output=True)

    assert first.returncode == 0, first.stderr
    assert failed.returncode == 1
    assert retry.returncode == 0, retry.stderr
    assert "compiled=1 skipped=0" in retry.stdout
    assert (cache_dir / "verify_add").read_text(encoding="utf-8") == "RECOVERED\n"
