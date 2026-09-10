# 第二轮缺陷修复验收明细（2026-09-10）

状态：**全部目标修复与最终门禁通过**。

## 验收范围与源码状态

- 分支：`codex/postfix-bugfix-20260910`
- 动态验收目录：`/mnt/d/workspace/onnx_translator_bugfix_worktree`
- 最终生产源码与测试提交：`c5882ba16ace037e4ce6bab0a509a917db16df06`
- 最终门禁起始 HEAD：`f084e92bab08c8110d7a640360d0a6205c6d6302`
- `f084e92` 相对 `c5882ba` 只增加独立审查报告，没有改动生产源码或测试。
- 最终门禁前后 tracked worktree 均为空；无残留 pytest、verify_all、CLI、nvcc 或 verifier 进程。
- 权威问题清单：`../postfix_audit_2026-09-10/ACCEPTANCE.md`

所有构建、清理、CPU/CUDA 测试和数值验证均在隔离 worktree 中执行。主目录 `D:\workspace\onnx_translator` 未运行 `verify_all`，原有 `.so`、cache、模型、结果和未跟踪审计材料均未被清理或覆盖。

## 环境

环境探针 RC 0，见 [原始输出](environment.stdout_stderr.log) 和 [独立返回码](environment.rc)。

- WSL `ubuntu2004`，kernel `5.15.167.4-microsoft-standard-WSL2`
- Python `/home/sakauma/data/miniconda3/envs/egor/bin/python` 3.12.12
- NumPy 2.4.6；ONNX 1.21.0；PyTorch 2.12.0+cpu
- GCC 9.4.0；NVCC CUDA 12.4
- NVIDIA GeForce RTX 4060 Laptop GPU，driver 610.88，8188 MiB

固定 PATH：

```text
/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
```

## 定向回归与独立审查

| 范围 | 结果 | RC | 证据 |
|---|---:|---:|---|
| 数值语义：`test_operator_misc_semantics.py` + `test_operator_reduce_semantics.py` | 38 passed | 0 | [日志](numeric_semantics.stdout_stderr.log)、[RC](numeric_semantics.rc) |
| control/sequence/optional | 11 passed | 0 | [日志](control_targeted.stdout_stderr.log)、[RC](control_targeted.rc) |
| harness 非 GPU 终验 | 67 passed，10 deselected | 0 | [命令](harness_finish_targeted.command.txt)、[RC](harness_finish_targeted.rc.txt) |
| harness 实际 GPU 终验 | 1 passed | 0 | [命令](harness_finish_gpu.command.txt)、[RC](harness_finish_gpu.rc.txt) |
| fresh DQL 单计划 | 1/1 | 0 | [日志](dql_numerical.stdout_stderr.log)、[RC](dql_numerical.rc) |

AUD003–005 的精确断言覆盖零长度 Scan 的 state、多输出、轴、符号维和 dtype；If 的 sequence、optional、optional-of-sequence 两分支容器语义；以及 If、Loop、Scan、SequenceMap 的完整父模型 opset map 与嵌套 `ai.onnx.ml:Binarizer` 真执行。

独立审阅 `8ccf831`、`ed0bd6c`、`02bfe46` 与最终 harness 差异后无剩余阻断。AUD005 的 If-only 覆盖已扩展至四类控制流；AUD007 的 Split/Unique 精确比较缺口已在 `c5882ba` 修复，并由 common comparator、wire 范围、NPS dtype 和单/多输出回归锁定。

## 原控制流探针复跑

复用了原 `import_graph/probe_control_sequence.py` 逻辑，但将输出重定向到新报告目录，没有覆盖旧审计证据。三条产品路径均成功执行：零长度 Scan 得到 state `[3,4]` 和 `(0,2)` 空输出；If sequence 后接 SequenceLength 得到 scalar `2`；嵌套 Binarizer 得到 `[0,0,1]`。

[control_probe.rc](control_probe.rc) 记录 `python_rc=0`、`actual_case_count=3`、`semantic_rc=0`。wrapper 只检查成功执行和输出存在，不解析所有 dtype/shape/value；硬语义结论以上述 pytest 精确断言为准。ReferenceEvaluator 的零长度 Scan 仍受自身空拼接限制，产品输出则按 schema 和 pytest 验证。

## 最终门禁

| 阶段 | 原生命令 | 结果 | RC | 原始证据 |
|---|---|---:|---:|---|
| CPU 工程门禁 | `python -u tools/verify_all.py --skip-cuda --keep-artifacts` | 482 passed，11 skipped；10/10 步骤通过 | 0 | [日志](gate_cpu.stdout_stderr.log)、[RC](gate_cpu.rc) |
| 全量 CUDA 重编 | `python -u tools/cli.py compile-cuda` | source=178，compiled=178，skipped=0，inventory=178 | 0 | [日志](gate_compile_cuda.stdout_stderr.log)、[RC](gate_compile_cuda.rc) |
| fresh CUDA protocol | `python -u -m pytest -q tests/test_cuda_verifier_protocol.py` | 14 passed，0 skipped | 0 | [日志](gate_protocol.stdout_stderr.log)、[RC](gate_protocol.rc) |
| 实际 GPU 并发/多输出 | `python -u -m pytest -q tests/test_numerical_harness_regression.py::test_real_gpu_add_and_unique_multioutput_cleanup` | 1 passed，0 skipped | 0 | [日志](gate_gpu_harness.stdout_stderr.log)、[RC](gate_gpu_harness.rc) |
| 全量原生数值验证 | `python -u tools/cli.py numerical --iterations 3 --skip-plots` | 723/723 plans；2169/2169 iterations；失败与异常 0 | 0 | [日志](gate_numerical.stdout_stderr.log)、[RC/计数](gate_numerical.rc) |

CPU 的 11 个 skip 拆分为 9 个 CUDA protocol case、1 个新 GPU smoke（均因 `--skip-cuda` 清空 cache 而跳过）和 1 个既有 ONNX17 Celu float16 兼容性 skip。重编后前 10 个 GPU case 全部执行并通过。旧“34 passed”是旧版 protocol+architecture 合并口径；本报告按当前真实命令记录 `14 + 1`，architecture 已在 CPU 的 482 项中执行。

numerical 使用 `python -u` 与 `pipefail` 保存未缓冲 stdout/stderr，开始于 `2026-09-10T16:15:48+08:00`，结束于 `2026-09-10T16:26:14+08:00`。日志含 723 条 `Testing`、723 条 `Pass (3/3)`，没有 `FAILED` 或 plan 预执行异常。

## 八项修复结论

1. DynamicQuantizeLinear 以公开 float32 scale 计算 zero point 和量化值；标准 fixture、ties-to-even fixture、真实 C 与 fresh CUDA 均通过。
2. omitted Reduce axes 配合 `noop_with_empty_axes=1` 返回 identity 且不调用 C reduction。
3. 零长度 Scan 保留全部输出及正确空 shape/dtype。
4. If 按 `TypeProto` 保留 sequence/optional 容器。
5. 控制流子图完整继承父模型 domain/opset map。
6. numerical/verify_all 拒绝非正 iterations，关闭 `0/0` 假通过。
7. 所有有符号/无符号整数在普通、TopK、Split、Unique 路径精确比较，并检查 wire 范围与 NPS dtype。
8. CUDA runner 使用逐调用隔离目录，严格检查主输出、sidecar 和退出诊断并可靠清理；同进程实际 GPU 并发通过。

## 限制与使用说明

- 完成声明覆盖 `ACCEPTANCE.md` 的八项第二轮审计缺陷与上述工程门禁；未额外执行 ABI/ASan/UBSan 审计。
- 默认 723 个 numerical plans 当前没有 `uint32`/`uint64` 输出。宽整数经浮点 wire 且超出连续精确表示范围时现在会 fail closed；这项保护不代表已提供完整的 `uint64` CUDA wire 协议。
- 本地 `main` 已通过 `git merge --ff-only` 同步修复提交和本报告证据，原有未跟踪审计材料以及 `.so`、`cache/`、模型和结果均保留；未 push。
- 主目录保留的生成物尚未按同步后的源码重建。直接从主目录执行前必须至少运行 `make` 重建 C 库；需要 CUDA 验证时再运行 `python tools/cli.py compile-cuda`。也可继续使用本次已完整验收的隔离 worktree。
