# 图入口门禁结果

## 执行状态

- 源码与测试冻结提交：`8a6f46985ad16265c5abc4283984f62ba98be2e1`
- WSL：`ubuntu2004`
- Python/PATH：与 `CPU_GATE_PLAN.md` 的固定环境一致。
- `graph-logic`：RC 0，stderr 为空。
- `verify-graph --no-clean`：RC 0，stderr 为空。
- 未运行 `verify_all`、clean、构建或全量 pytest；这两个入口不读取或写入 CUDA cache，可与 fresh nvcc 编译并行。

## 模型来源

两个入口均显式读取隔离 worktree 中执行前已存在的 `./onnx_model/model.onnx`。该目录由 `.gitignore:235` 的 `onnx_model/` 规则忽略，模型不属于冻结提交的 tracked 文件，因此不声称它由提交 `8a6f4698...` 提供。本轮没有运行模型生成命令，也没有改写该文件。

- 绝对路径：`D:\workspace\onnx_translator_bugfix_worktree\onnx_model\model.onnx`
- 大小：342876 bytes
- LastWriteTime：2026-09-10 16:07:41（Asia/Shanghai）
- SHA256：`3da592ce6de1f81f0c93b1c5139a0bc0497c16bdfa3f527d4105a4f6fca27348`
- Git blob hash（只读 `git hash-object`）：`91974ccdff25d3bfa8f937aec90838dfa7517181`
- 仓库生成入口候选：`tools/commands/create_graph_ops_model.py` 默认输出同一路径，并声明 producer `onnx_translator_graph_ops`；仅据路径不能证明现有 artifact 的具体生成运行。

## 结果明细

`graph-logic` 从模型解析 140 nodes，导入 148 个算子；生成输入 placeholder `(1, 4, 32, 32)`，`Graph.forward_` 完成，并生成 SVG。

`verify-graph` 在默认 strict 模式下同样导入 148 个算子，算子统计中没有 `GenericNode`；声明输出校验与 `Graph.forward_` 完成，并生成 SVG。命令使用独立任务名和 `--no-clean`，没有删除既有结果目录。

完整证据：

- `graph_logic.command.txt`、`graph_logic.stdout.txt`、`graph_logic.stderr.txt`、`graph_logic.rc.txt`
- `verify_graph.command.txt`、`verify_graph.stdout.txt`、`verify_graph.stderr.txt`、`verify_graph.rc.txt`
