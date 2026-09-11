# REAUD-007 验证器修复交叉审阅

## 审阅范围

本次为只读交叉审阅，没有修改验证器或其测试。审阅文件：

- `tools/numerical/output_contracts.py`
- `tools/numerical/runner.py`
- `tools/numerical/runner_special_outputs.py`
- `tests/test_reaudit_special_output_contracts.py`
- 原始 fault injection：`docs/reports/reaudit_2026-09-10/harness/probe_output_contract_coercion.py`

## 结论

静态审阅未发现阻断项。修复在 lossy cast、round、clip 或 reshape 前验证已确认的特殊输出 contract，并保持合法 typed 输出的后续比较路径。

### 八个复现 case

新测试与原始探针的八次 `verify_op` 调用逐项对应：

1. DQL NPS `y`/`zero_point` 错误 dtype 与小数值，同时 `scale`/`zero_point` 为错误 rank；
2. DQL CUDA packed `y` 为小数；
3. DQL CUDA packed `y` 越界；
4. DQL CUDA packed `y` 为 NaN；
5. DQL CUDA packed `zero_point` 为小数；
6. TopK NPS indices 为小数 float32；
7. Unique NPS indices/inverse/counts 为小数 float32；
8. Dropout NPS mask 为非 bool 的 uint8(2)。

这些 case 都调用公开 `runner.verify_op(...)` 并断言最终布尔结果为 `False`，没有绕过计划级调度直接测试 helper。失败分支在增加 `pass_count` 前停止，因此单 iteration 结果不会被记为通过。

### 合法 typed 输出兼容性

- DQL 接受 schema 对应的 `uint8 y`、scalar `float32 scale`、scalar `uint8 zero_point`，CUDA packed wire 保持 float32 文件协议；只在验证 finite/integral/range 后转换语义 uint8 字段。
- TopK 保留 values 的既有数值比较，要求两侧 indices 为 int64 且 shape 等于实际 K 推导出的输出 shape。
- Unique 保留 values 的既有数值比较，要求三个辅助输出及 sidecar 为 int64，并分别校验 unique length 与 inverse shape。
- Dropout 要求 NPS mask 为 bool，CUDA sidecar 仍按既有 uint8 wire 读取并只接受 0/1，随后才转换为 bool 比较。
- 测试为上述四类各保留至少一个合法 typed `verify_op` 通过例，能够防止修复把标准 dtype/shape 一并拒绝。

底层 `run_cuda_ground_truth` 仍按 `CudaSidecarSpec` 的 dtype、shape 和精确文件字节数读取 sidecar；新增校验没有改变 sidecar 文件格式，也没有以任意 typed sidecar 绕过读取层。

## 最终兼容修复复核

验证器定向测试初轮暴露两项既有测试兼容问题。对最终源码的第二次只读复核结论如下，未发现阻断或可操作问题：

- TopK 只在 legacy 直接调用 fixture 完全省略 K 输入时，以原始 NPS values shape 作为兼容 fallback。所有真实计划均提供 `inputs_np[1]`，仍从输入 shape、规范化 axis 与实际 K 值推导期望 shape；NPS/CUDA values 及两侧 int64 indices 必须匹配该 shape。因此真实 shape/K contract 未放宽，fractional indices 仍在转换前失败。
- DQL 为保留 CUDA verifier 缺失输出的基础设施错误优先级，只先读取未经转换的 NPS `y` 元素数来确定 packed 文件长度，并在 CUDA 返回 `None` 时立即抛出 `RuntimeError`。一旦 CUDA 输出存在，NPS 三个输出与 packed wire 仍全部经过原有精确 dtype、shape、finite、integral、range 校验，随后才允许 cast 或 reshape；没有恢复 round/clip，也没有放宽缺失字段 contract。

最终验证器定向集合由负责 agent 实际运行并报告 `83 passed`、`0 skipped`、RC 0。本复核只读最终 runner delta，没有重复运行该集合。
