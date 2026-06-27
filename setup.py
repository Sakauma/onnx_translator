# /**
#   ******************************************************************************
#   * @file        setup.py
#   * @author      Egor Izmaylov
#   * @brief       Builds and packages the C runtime shared library with the Python package.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parent


class BuildPyWithRuntime(build_py):
    """Build tensor_ops.so and place it inside the nn package for wheel installs."""

    def run(self):
        subprocess.check_call(["make", "tensor_ops.so"], cwd=ROOT)
        super().run()
        package_dir = Path(self.build_lib) / "nn"
        package_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "tensor_ops.so", package_dir / "tensor_ops.so")


class BinaryDistribution(Distribution):
    """Force wheel install layout to platlib because nn/tensor_ops.so is native code."""

    def has_ext_modules(self):
        return True


class BDistWheelWithRuntime(bdist_wheel):
    """Mark wheels as platform-specific because they include tensor_ops.so."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False


setup(
    cmdclass={"build_py": BuildPyWithRuntime, "bdist_wheel": BDistWheelWithRuntime},
    distclass=BinaryDistribution,
)
