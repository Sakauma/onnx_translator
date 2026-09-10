# 九组缺陷修复验收记录（2026-09-10）

## 验收范围与环境

- 基线：`42f2527347d1c02b17395e90c08b0ee7a46b2eaf`
- 已验证源码 HEAD：`4ba7f76e5608419e0b6df4cd77b2f21d48d03916`
- 分支：`codex/bugfix-audit-20260910`
- 隔离工作区：`/mnt/d/workspace/onnx_translator_bugfix_worktree`
- WSL：`ubuntu2004`；Python `3.12.12`；NumPy `2.4.6`；ONNX `1.21.0`；GCC `9.4.0`；CUDA 编译器 CUDA `12.4`。

所有构建和动态测试均在隔离工作区运行。主仓库在验收期间保持基线 HEAD，原有三组未跟踪审计材料完整保留。

## 最终门禁

| 门禁 | 结果 | 原始日志 |
| --- | --- | --- |
| `tools/verify_all.py --skip-cuda --keep-artifacts` | RC 0；437 passed，10 skipped；strict coverage、代表模型、图模型和导出模型全部通过 | [final_verify_all_skip_cuda_retry.log](final_verify_all_skip_cuda_retry.log) |
| 全量 `compile-cuda --force` | RC 0；compiled=178，skipped=0 | [final_compile_cuda_all.log](final_compile_cuda_all.log) |
| 全量编译后的 fresh CUDA protocol + runner tests | RC 0；34 passed，0 skipped | [final_cuda_protocol.log](final_cuda_protocol.log) |
| 原生 `numerical --iterations 3 --skip-plots` | RC 0；723/723 个 live plan 通过，2169/2169 次迭代通过，失败标记 0；最后触达 LSTM | [final_numerical_full.log](final_numerical_full.log) |

CPU 门禁的 10 个 skip 中，9 个来自 `--skip-cuda` 流程清理 `cache/` 后 CUDA protocol 测试找不到可执行文件，另 1 个是既有的 ONNX17 Celu float16 兼容性 skip。178 个 verifier 重新编译后，CUDA protocol 测试无 skip。

## 九组缺陷验收

- GATE-001：补齐 Slice helper 导入；五个 Slice plan 各三轮均通过；准备期异常按计划聚合并继续后续 inventory。
- IMP-001：按规范化 domain、op type 与 opset 分派；未知 domain strict/non-strict 拒绝策略通过。
- IMP-002：Softmax opset 11 flatten 语义和现代版本行为回归通过。
- IMP-003：external initializer 官方 loader、strict 失败与 non-strict 结构化诊断通过。
- GRAPH-001：声明输出名称、顺序、dtype、rank、已知维度及未知 rank 验证通过。
- NUM-001：独立精确 fixture `[-253,257,0]` 得到 `scale=2`、`zp=126`、`y=[0,254,126]`，CPU/CUDA 真路径通过。
- NUM-002：ReduceLogSumExp 的全 `-Inf`、`+Inf`、混合无穷及大有限值 CPU/Python/CUDA 回归通过。
- NUM-003：空 axes 在 constructor/runtime/shape-only 路径按 noop 标志统一，C 与 fallback 回归通过。
- CUDA-001：178 个 verifier 全部接入 runtime、launch/sync 与精确 I/O 检查；无设备 Add 探针非零退出并保留 op、exit code、stderr；Unique 四输出及 sidecar 写失败协议通过。

runner 的单输出、多输出和 TopK/Unique 等 sidecar 对缺失、截短、多字节内容均会失败并聚合；成组 sidecar 在中途失败时全部清理。相关证据见 [cuda_add_no_device_probe.log](cuda_add_no_device_probe.log)、[runner_sidecar_group_pytest.log](runner_sidecar_group_pytest.log) 和 [gate_slice_numerical.log](gate_slice_numerical.log)。

## 提交与审阅结论

主要修复提交：`7b53392`、`8ea812e`、`e206833`、`af3e16b`、`daefd16`、`7b91247`、`f61de1f`、`0a6cfd2`、`e5cc88d`、`2a39cad`、`c8cbc28`、`937fde2`、`a939823`、`1527031`、`7360021`、`682f5df`、`5063ab6`、`0c05a40`、`6900d77`、`4ba7f76`。`bf1cefb` 的无据 opset 26 anchors 已由 `6900d77` 撤销，历史组合 fixture 明确声明 opset 17。独立审阅发现的 CUDA 宏嵌套编译阻断已由 `4ba7f76` 修复，并由全量 178 编译及 723 plan 原生门禁验证。

## 未纳入本次完成声明

本次未重试此前被平台阻断的 ABI、ASan、UBSan 审计，也未关闭 `[-1,0,1]` DynamicQuantizeLinear 中间精度的独立调查。这些项目不计入九组缺陷通过结论。
