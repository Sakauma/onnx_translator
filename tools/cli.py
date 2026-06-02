# /**
#   ******************************************************************************
#   * @file        cli.py
#   * @author      Egor Izmaylov
#   * @brief       提供统一工程命令入口，集中调度模型生成、图验证、CUDA 编译和数值验证。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.commands.create_graph_ops_model import export_graph_ops_model
from tools.commands.create_model import export_model
from tools.commands.graph_logic import main as graph_logic_main
from tools.commands.verify_graph import main as verify_graph_main
from tools.numerical.cli import main as numerical_main


# 返回仓库根目录，保证从任意当前目录调用时都能找到相对路径资源。
def repo_root() -> Path:
    return REPO_ROOT


# 调用 CUDA verifier 编译脚本，保留原有 shell 实现中的 nvcc 自动发现逻辑。
def compile_cuda_main(argv: list[str]) -> int:
    script = repo_root() / "tools" / "commands" / "compile_cuda.sh"
    return subprocess.call(["bash", str(script), *argv], cwd=repo_root())


# 解析统一 CLI 参数，并将剩余参数透传给对应子命令。
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX Translator 工程工具统一入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("create-model", help="生成 PyTorch 导出的综合 ONNX 模型。")
    graph_model = subparsers.add_parser("create-graph-model", help="生成图结构/shape 类 ONNX 测试模型。")
    graph_model.add_argument("--output", default="./onnx_model/model.onnx", help="输出 ONNX 模型路径。")

    graph_logic = subparsers.add_parser("graph-logic", help="验证 ONNX 图导入和结构推断。")
    graph_logic.add_argument("--model", default="./onnx_model/model.onnx", help="ONNX 模型路径。")
    graph_logic.add_argument("--task-name", default="graph_logic_test", help="结果目录任务名。")

    passthrough = {
        "verify-graph": "验证 ONNX 导入、图构建、形状推断和可视化。",
        "numerical": "运行 C 后端与 CUDA 参考程序的数值正确性验证。",
        "compile-cuda": "编译 cuda/verify_*.cu 到 cache/。",
    }
    for name, help_text in passthrough.items():
        subparsers.add_parser(name, help=help_text)

    args, passthrough_args = parser.parse_known_args(argv)
    if args.command in passthrough:
        args.args = passthrough_args
    elif passthrough_args:
        parser.error("unrecognized arguments: " + " ".join(passthrough_args))
    return args


# 根据子命令调度对应实现，保持旧脚本功能集中到一个入口。
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "create-model":
        export_model()
        return 0
    if args.command == "create-graph-model":
        export_graph_ops_model(args.output)
        return 0
    if args.command == "graph-logic":
        return graph_logic_main(args.model, args.task_name)
    if args.command == "verify-graph":
        return verify_graph_main(args.args)
    if args.command == "numerical":
        return numerical_main(args.args)
    if args.command == "compile-cuda":
        return compile_cuda_main(args.args)
    raise RuntimeError(f"未知命令: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
