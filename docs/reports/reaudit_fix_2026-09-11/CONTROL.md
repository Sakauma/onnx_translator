# REAUD-003 至 REAUD-006 控制流修复记录

## 范围与基线

- 基线提交：`f253f634dd251f348f432e62e318faf25198fd30`
- 工作分支：`codex/postfix-bugfix-20260910`
- 修复范围：Loop 空 scan shape、Loop Sequence carried state、空 SequenceMap 多输出、Scan output axis。
- 未扩展 importer 支持范围；Optional 仅沿用既有 TypeProto 转换路径，本记录不声称复现或新增 Optional 支持。

## 设计与改动

### REAUD-003：Loop 零次迭代 scan shape

Loop 在 body 未执行时，从 body carried tensor input 的声明维与实际运行时 shape 建立符号绑定。重复符号若绑定到不同大小会明确失败；输出维只接受具体 `dim_value` 或已绑定 `dim_param`，保留真实的零维，未解析维抛出 `ValueError`。该路径不执行 body，也不再把未知维编造为 `1`。

### REAUD-004：Loop Sequence carried state

Loop 初始 carried values 使用递归 reference feed 转换，body 每次返回值保持原容器结构，最终根据对应 body output `TypeProto` 转回工程值。零次与非零次路径使用同一最终转换，因此合法 Sequence state 不会经 `np.asarray` 压成 tensor。已有 Optional 转换逻辑仍由同一个 TypeProto helper 处理，但本次没有扩大其声明范围。

### REAUD-005：空 SequenceMap 多输出

SequenceMap 在遍历前按 `len(body.output)` 创建输出 buckets。空输入会返回每个声明输出对应的空 sequence；非空输入按 body output `TypeProto` 转换，并校验实际返回 arity 与声明一致。

### REAUD-006：Scan output axis

Scan 的非空 runtime、空输出与 `forward_` 均使用最终输出 rank 规范化 `scan_output_axes`。`forward_` 同时规范化 input axis 后再读取 scan length。合法负边界 `-(body_rank + 1)`、正轴和 `-1` 均按 ONNX 位置解释；超出 `[-rank, rank - 1]` 的轴明确失败。Scan empty shape 的符号绑定只使用 `Tensor_` 元数据，不为 shape 分配真实数组。

## 持久回归

新增 `tests/test_reaudit_control_boundaries.py`，通过 full ONNX checker、`ONNXImport(..., strict=True)` 和真实 `Graph` 执行覆盖四个最小合法模型：

- Loop 符号 scan 的零次 `(0, 2)` 与一次 `(1, 2)`，以及 initial condition 为 false 的未解析维失败；
- Sequence carried state 的零次与一次路径；
- SequenceMap 两个 body outputs 的空与非空 sequence；
- Scan 正 output axis、`-1`、合法负边界，以及 runtime/shape-only 对非法正负轴的一致拒绝。

## 验证结果

统一 C 构建已由指定执行者完成，本控制流任务没有重建共享库。测试时使用的 `tensor_ops.so` SHA256 为 `602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66`，统一 `make` 记录为 RC 0。

在 WSL 发行版 `ubuntu2004` 中，使用固定 Python 与 PATH 运行：

```text
cd /mnt/d/workspace/onnx_translator_bugfix_worktree && PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin /home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_reaudit_control_boundaries.py tests/test_operator_sequence_control.py
```

结果：`23 passed in 1.53s`，pytest RC 0，0 skipped。stdout、空 stderr、RC 与外层 WSL 命令分别保存在 `control_targeted.stdout.txt`、`control_targeted.stderr.txt`、`control_targeted.rc.txt`、`control_targeted.command.txt`。

首次尝试使用不存在的发行版名称 `Ubuntu-20.04`，在进入 pytest 前由 WSL 返回 `WSL_E_DISTRO_NOT_FOUND`；随后按环境声明的精确名称 `ubuntu2004` 执行成功。未运行 `verify_all`、CUDA 或 numerical，因为本任务只改 Python 控制流实现且统一调度限定为新旧控制流定向回归；未执行 clean。
