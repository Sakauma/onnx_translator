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
