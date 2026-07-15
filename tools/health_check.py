# /**
#   ******************************************************************************
#   * @file        health_check.py
#   * @author      Egor Izmaylov
#   * @brief       检查 Python 模块、编译工具和 CUDA 工具链是否满足工程验证要求。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import argparse
import glob
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PYTHON_MODULES = [
    "numpy",
    "onnx",
    "torch",
    "matplotlib",
    "graphviz",
    "ml_dtypes",
    "pytest",
]

REQUIRED_TOOLS = ["gcc", "make", "dot"]
CUDA_TOOLS = ["nvcc"]


# 实现 `repo_root` 步骤，规范化输入并返回下游期望的数据或元信息。
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# 输出当前验证环境的关键信息，方便交接时快速定位路径、工具链和二进制产物状态。
def print_environment_summary():
    root = repo_root()
    tensor_ops_path = root / "tensor_ops.so"
    nvcc_env = os.environ.get("NVCC", "<unset>")
    detected_nvcc = resolve_tool("nvcc") or "<missing>"
    print(f"Repository root: {root}")
    print(f"Platform: {platform.platform()}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"tensor_ops.so: {'present' if tensor_ops_path.exists() else 'missing'}")
    print(f"NVCC env: {nvcc_env}")
    print(f"Detected nvcc: {detected_nvcc}")


# 实现 `tool_version` 步骤，规范化输入并返回下游期望的数据或元信息。
def tool_version(command):
    path = resolve_tool(command)
    if not path:
        return None, None
    version_args = [path, "-V"] if command == "dot" else [path, "--version"]
    try:
        result = subprocess.run(
            version_args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        version = result.stdout.splitlines()[0] if result.stdout else ""
    except OSError:
        version = ""
    return path, version


# 实现 `resolve_tool` 步骤，规范化输入并返回下游期望的数据或元信息。
def resolve_tool(command):
    if command == "nvcc" and os.environ.get("NVCC"):
        nvcc_env = os.environ["NVCC"]
        if os.path.exists(nvcc_env):
            return nvcc_env

    path = shutil.which(command)
    if path:
        return path

    if command == "nvcc":
        for candidate in ["/usr/local/cuda/bin/nvcc", *glob.glob("/usr/local/cuda-*/bin/nvcc")]:
            if os.path.exists(candidate):
                return candidate

    return None


# 实现 `check_modules` 步骤，规范化输入并返回下游期望的数据或元信息。
def check_modules():
    missing = []
    for module_name in PYTHON_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"OK module {module_name}: {version}")
        except Exception as exc:
            print(f"MISSING module {module_name}: {exc}")
            missing.append(module_name)
    return missing


# 实现 `check_tools` 步骤，规范化输入并返回下游期望的数据或元信息。
def check_tools(tools):
    missing = []
    for command in tools:
        path, version = tool_version(command)
        if not path:
            print(f"MISSING tool {command}")
            missing.append(command)
        else:
            print(f"OK tool {command}: {path} ({version})")
    return missing


# 作为 `tools/health_check.py` 的命令行入口，解析参数、调度检查流程并返回进程退出码。
def main():
    parser = argparse.ArgumentParser(description="Check local project runtime dependencies.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail when nvcc is unavailable.")
    args = parser.parse_args()

    print_environment_summary()
    missing_modules = check_modules()
    missing_tools = check_tools(REQUIRED_TOOLS)
    missing_cuda = check_tools(CUDA_TOOLS)

    failures = missing_modules + missing_tools
    if args.require_cuda:
        failures.extend(missing_cuda)

    if failures:
        print(f"ERROR: missing dependency/dependencies: {', '.join(failures)}")
        return 1

    if missing_cuda:
        print("WARNING: nvcc is unavailable; CUDA numerical verification cannot run in this environment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
