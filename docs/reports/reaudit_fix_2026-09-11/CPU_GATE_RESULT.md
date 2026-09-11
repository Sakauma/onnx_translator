# 全量 pytest 门禁结果

## 运行身份

- 冻结源码/测试 HEAD：`8a6f46985ad16265c5abc4283984f62ba98be2e1`
- `tensor_ops.so` SHA256：`602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66`
- WSL：`ubuntu2004`
- Python：`/home/sakauma/data/miniconda3/envs/egor/bin/python`
- PATH：`/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin`
- 命令：`python -u -m pytest -q -ra tests`
- 开始：`2026-09-11T12:05:50.2328520+08:00`
- 结束：`2026-09-11T12:06:13.0069775+08:00`
- pytest 自报耗时：14.76s

## 结果

- `528 passed`
- `1 skipped`
- `0 failed`
- 真实 RC：0
- stderr：空

唯一 skip：`tests/test_operator_activation_semantics.py:115`，原因原文为 `Celu does not support float16 in ONNX17`。这是 ONNX17 的算子类型支持边界，不是缺少 C library、CUDA executable、GPU 或驱动导致的环境跳过。

## CUDA 编译库存、静态 protocol 与 GPU smoke

前置 fresh CUDA 证据见 `CUDA_BUILD.md`、`cuda_compile.stdout.txt`、`cuda_compile.stderr.txt`、`cuda_compile.rc.txt`：正式第二轮 RC 0，`compiled=178 skipped=0`，178 个源码对应 178 个 executable，missing/orphan 均为空。这是编译与库存证据，不表示 pytest 动态运行了全部 178 个 verifier。

全量 pytest 的 `-ra` 摘要只列出上述 Celu skip，因此所有 CUDA/GPU 条件测试均实际执行而未跳过，包括：

- `test_all_178_verifiers_reach_checked_runtime_and_io_helpers`，读取全部 178 份 verifier 源码并静态检查 launch/I/O error-check protocol；该测试不启动这 178 个 executable；
- real GPU DynamicQuantizeLinear ties-to-even fixture；
- real GPU ReduceLogSumExp 的 5 个 nonfinite/stability 参数 case；
- real GPU Unique 四输出精确值与 sidecar write failure case；
- real GPU 并发 Add 与 Unique 多输出临时目录清理 smoke；
- `CUDA_VISIBLE_DEVICES=""` 时 Add verifier 必须非零退出并给出诊断的 fail-closed case；
- runner 的 missing/truncated/extra main output、sidecar、stderr、并发隔离等 protocol 回归。

按测试源码，这些 real GPU case 会在 executable 缺失、无 CUDA device、驱动版本不足或初始化失败时调用 `pytest.skip`；最终摘要没有出现这些原因，证明本次没有走对应 skip 分支。

因此，本页只确认上述明确命名的 DQL、ReduceLogSumExp、Unique、Add 等 pytest case 进行了动态 GPU/进程执行；全量 178 verifier 的动态数值执行属于后续 numerical gate 的证据范围。

## 证据文件

- `full_pytest.command.txt`
- `full_pytest.stdout.txt`
- `full_pytest.stderr.txt`
- `full_pytest.rc.txt`
- `full_pytest.start.txt`
- `full_pytest.end.txt`

本轮没有重复图 CLI，没有运行 make、compile-cuda、numerical、`verify_all` 或 clean。

## 最终源码快照 attempt2

首次 numerical gate 暴露 QuantizeLinear CUDA reference 精度仍旧的问题后，负责 agent 修复并提交 3 个文件。最终 source snapshot 从 `8a6f46985ad16265c5abc4283984f62ba98be2e1` 前进到 `b8d90098176f0bec35c1ddea334781697f9881b7`；tracked worktree/index 在执行前均干净。

- `tensor_ops.so` 未变化，SHA256 仍为 `602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66`。
- 178 个 CUDA verifier 中仅 `cache/verify_quantize_linear` 相对首次 inventory 更新，最终 SHA256 为 `0c12b072e454d8e80aeefa3ac9af1ea3f7a9d03850a69e67bc0b849d2c7bf8be`；其他 177 个由 fresh inventory 复核为未变化。
- 图 CLI 与 C 源码/共享库未变化，按调度没有重复运行图 gate、make 或全 178 编译。

在最终 snapshot 上使用相同固定 WSL/Python/PATH，仅运行一次完整 pytest：

```text
python -u -m pytest -q -ra tests
```

attempt2 结果：`537 passed, 1 skipped in 23.16s`，0 failed，真实 RC 0，stderr 为空。开始时间 `2026-09-11T12:25:46.0198598+08:00`，结束时间 `2026-09-11T12:26:21.9602436+08:00`。

唯一 skip 仍为 `tests/test_operator_activation_semantics.py:115: Celu does not support float16 in ONNX17`。该原因属于 schema 支持边界；没有 C library、CUDA executable、GPU、驱动或初始化相关 skip，因此最终 snapshot 的 CUDA/GPU 条件测试均未跳过。

attempt2 完整证据独立保存在 `full_pytest_attempt2.command.txt`、`full_pytest_attempt2.stdout.txt`、`full_pytest_attempt2.stderr.txt`、`full_pytest_attempt2.rc.txt`、`full_pytest_attempt2.start.txt`、`full_pytest_attempt2.end.txt`，没有覆盖首次 528 passed / 1 skipped 记录。
