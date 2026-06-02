"""文件功能：检查 Python 模块、编译工具和 CUDA 工具链是否满足工程验证要求。
作者：Egor Izmaylov
时间：2026-06-02
"""

import argparse
import glob
import importlib
import os
import shutil
import subprocess
import sys


PYTHON_MODULES = [
    "numpy",
    "onnx",
    "onnxscript",
    "torch",
    "matplotlib",
    "graphviz",
    "pytest",
]

REQUIRED_TOOLS = ["gcc", "make", "dot"]
CUDA_TOOLS = ["nvcc"]


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

    print(f"Python: {sys.executable}")
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
