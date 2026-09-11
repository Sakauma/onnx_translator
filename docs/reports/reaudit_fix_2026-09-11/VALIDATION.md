# 修复验证记录

## 冻结身份

- 最终源码 HEAD：`b8d90098176f0bec35c1ddea334781697f9881b7`
- 主修复提交：`8a6f46985ad16265c5abc4283984f62ba98be2e1`
- CUDA oracle 跟进提交：`b8d90098176f0bec35c1ddea334781697f9881b7`
- `tensor_ops.so` SHA-256：`602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66`
- `verify_quantize_linear` SHA-256：`0c12b072e454d8e80aeefa3ac9af1ea3f7a9d03850a69e67bc0b849d2c7bf8be`

精确提交范围见 [FREEZE.md](FREEZE.md)，构建来源见 [NUMERIC_BUILD.md](NUMERIC_BUILD.md) 和 [CUDA_BUILD.md](CUDA_BUILD.md)。

## 最终门禁

- C 构建：RC 0。
- fresh `compile-cuda --force`：178/178 compiled、skipped 0、RC 0；CUDA oracle 跟进后单目标重编 1/1 成功。
- 最终全量 pytest：**537 passed，1 skipped，0 failed**，RC 0，stderr 为空，见 [CPU_GATE_RESULT.md](CPU_GATE_RESULT.md) 与 `full_pytest_attempt2.*`。
- 唯一 skip：既有 `Celu does not support float16 in ONNX17`，不是 C/CUDA/GPU 环境 skip。
- `graph-logic` 与 strict `verify-graph --no-clean`：任务分别为 `reaudit_fix_graph_logic_20260911`、`reaudit_fix_verify_graph_20260911`，均 RC 0、stderr 为空，见 [GRAPH_GATE_RESULT.md](GRAPH_GATE_RESULT.md)。
- 验证器定向集：83 passed、0 skipped、RC 0，见 [HARNESS.md](HARNESS.md)。
- QuantizeLinear CUDA 定向：9 passed、0 skipped、RC 0，见 [CUDA_QUANTIZATION_FIX.md](CUDA_QUANTIZATION_FIX.md) 与 [REVIEW_CUDA_QUANTIZATION.md](REVIEW_CUDA_QUANTIZATION.md)。

最终 pytest 的 GPU 条件测试没有 skip，并实际执行 DQL、5 个 ReduceLogSumExp 参数 case、Unique 精确多输出与 sidecar 失败、并发 Add+Unique，以及 QL half/bfloat16 precision cases。178 个 verifier 的完整动态数值覆盖由下述 numerical 提供。本轮未执行 `verify_all.py`，因此不声称其历史“10/10”标签。

## 第一次完整 numerical：保留失败

命令：

```text
/home/sakauma/data/miniconda3/envs/egor/bin/python -u tools/cli.py numerical --iterations 3 --skip-plots
```

- 时间：`2026-09-11T04:08:25Z` 至 `2026-09-11T04:17:56Z`
- source：`8a6f46985ad16265c5abc4283984f62ba98be2e1`
- 结果：**721/723 passed，2 failed，异常 0，RC 1，stderr 为空**

失败均为默认 `precision=0` 的 QuantizeLinear int8：FLOAT16 为 C `-24`、CUDA `-25`；BFLOAT16 为 C `116`、CUDA `117`。独立检查证明旧 CUDA params/kernel 只有 float/double 二值模式，把 scale dtype 所需的 half/bfloat16 division 提升成 float32。完整首轮证据保存在 `numerical_gate.{stdout,stderr,rc,meta}.txt` 与 `numerical_gate.cuda_inventory.txt`，没有覆盖或重分类为成功。

## 第二次完整 numerical：最终通过

CUDA oracle scoped 修复、独立审阅、单目标重编、真实 GPU 定向测试和最终 pytest 完成后，使用相同命令运行 attempt2：

- 时间：`2026-09-11T04:27:35Z` 至 `2026-09-11T04:39:44Z`
- source：`b8d90098176f0bec35c1ddea334781697f9881b7`
- compiled verifier inventory：178，逐文件 SHA-256 已保存
- plans started/passed：**723/723**
- 每计划：**3/3**
- 总迭代：**2169**
- failed plans / infrastructure exceptions：0 / 0
- stderr：空
- 真实 Python/wrapper RC：**0**

attempt2 证据保存在 `numerical_gate_attempt2.{stdout,stderr,rc,meta}.txt` 与 `numerical_gate_attempt2.cuda_inventory.txt`。日志有 723 条 `Testing`、723 条 `Pass (3/3)`，没有 `Fail`、plan exception 或最终 error summary。

## 验收结论

最终快照的构建、fresh CUDA、全量 pytest、图 CLI、定向回归和 723×3 numerical 均以真实零退出码完成。首轮 oracle 失败及修复链完整保留；最终结论没有依赖放宽容差、修改产品期望或把零结果/跳过计为通过。

## 主目录阶段一复验

主目录已通过 `git merge --ff-only` 从 `f253f634dd251f348f432e62e318faf25198fd30` 同步到最终源码 `b8d90098176f0bec35c1ddea334781697f9881b7`。主目录 C 构建 RC 0，`tensor_ops.so` SHA-256 与隔离区同为 `602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66`。

主目录单目标强制编译 QuantizeLinear verifier 为 `compiled=1 skipped=0`、RC 0。主 binary SHA-256 `27079beec490af52644f5092b933e615487eafe6bdedcc63dac65367ee69cfa7` 与隔离区不同，但大小相同、源码 SHA-256 同为 `6ccaff3ab0e48ce588d60b027e41f01c7b48d90ee0d5403af1d6979e5fb82c89`、ELF Build ID 同为 `0856c3069fbb815534bc98ce21ce9982c362b955`；唯一字节差异位于 `.strtab`。没有用反复重编或复制掩盖该元数据差异。

主目录随后运行 `test_reaudit_quantization_precision.py` 与 `test_reaudit_quantization_cuda.py`，结果为 **20 passed、0 skipped、0 failed、RC 0、stderr 为空**。这验证了主目录刚构建的真实 C library 与 GPU verifier。原始命令、时间、输出和 RC 均由 [MAIN_SYNC_RESULT.md](MAIN_SYNC_RESULT.md) 索引。

报告专用提交及其第二次主目录 fast-forward 在本文定稿时尚未执行，不计入上述阶段一结果。
