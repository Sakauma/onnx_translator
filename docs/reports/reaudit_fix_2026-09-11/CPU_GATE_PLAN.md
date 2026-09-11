# 全量 pytest 与图入口门禁计划

## 冻结状态

- 源码与测试冻结提交：`8a6f46985ad16265c5abc4283984f62ba98be2e1`
- 执行目录：`/mnt/d/workspace/onnx_translator_bugfix_worktree`
- WSL 发行版：`ubuntu2004`
- Python：`/home/sakauma/data/miniconda3/envs/egor/bin/python`
- PATH：`/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin`
- 前置条件：负责 agent 完成 178 个 CUDA verifier 的 fresh 编译并报告成功后再执行。

## 计划命令

全量 pytest，包含仓库 tests 中已有的 CUDA protocol/GPU smoke；只运行一次，不另行重复定向测试：

```text
wsl.exe -d ubuntu2004 -- bash -lc 'cd /mnt/d/workspace/onnx_translator_bugfix_worktree && PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin /home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests'
```

图逻辑入口使用仓库已有小模型 `onnx_model/model.onnx`，显式模型与独立任务名：

```text
wsl.exe -d ubuntu2004 -- bash -lc 'cd /mnt/d/workspace/onnx_translator_bugfix_worktree && PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin /home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py graph-logic --model ./onnx_model/model.onnx --task-name reaudit_fix_graph_logic_20260911'
```

严格图验证入口同样显式使用该模型。`verify-graph` 默认会递归删除同名结果目录，因此必须传 `--no-clean`，并使用本轮独立任务名：

```text
wsl.exe -d ubuntu2004 -- bash -lc 'cd /mnt/d/workspace/onnx_translator_bugfix_worktree && PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin /home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py verify-graph --model ./onnx_model/model.onnx --task-name reaudit_fix_verify_graph_20260911 --no-clean'
```

`verify-graph` 保持默认 strict importer，且不传 `--allow-generic`，因此 GenericNode 会使门禁失败。现有 `onnx_model/model.onnx` 已在 worktree 中，不运行 `create-model` 或 `create-graph-model` 改写 fixture。

## 记录要求

每条命令分别保存 command、stdout、stderr 与真实 RC。pytest 报告的 passed/failed/skipped 数量按原样记录；每个 skip 必须从 pytest 输出、marker 或测试源码核对精确原因，不能只记总数。若 `-q` 的摘要没有给出 skip 原因，在不重复执行全量测试的前提下读取 pytest 终端摘要及对应测试源码/marker完成归因。

禁止运行 `verify_all.py`、`make clean` 或任何清理命令；不重建 `tensor_ops.so`，不重复编译 CUDA verifier。
