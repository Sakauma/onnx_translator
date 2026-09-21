# /**
#   ******************************************************************************
#   * @file        test_cli_external_cwd.py
#   * @author      Egor Izmaylov
#   * @brief       Covers repository-relative CLI defaults outside the checkout.
#   * @details     2026.09.21  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _read_cuda_default(cwd, env):
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tools.numerical.cuda import CUDA_VERIFY_DIR; print(CUDA_VERIFY_DIR)",
        ],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def test_numerical_default_cache_is_repository_relative_outside_checkout(tmp_path):
    env = os.environ.copy()
    env.pop("CUDA_VERIFY_DIR", None)
    env["PYTHONPATH"] = str(ROOT)

    assert Path(_read_cuda_default(tmp_path, env)) == ROOT / "cache"


def test_numerical_explicit_cache_environment_remains_unchanged(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["CUDA_VERIFY_DIR"] = "caller-selected-cache"

    assert _read_cuda_default(tmp_path, env) == "caller-selected-cache"
