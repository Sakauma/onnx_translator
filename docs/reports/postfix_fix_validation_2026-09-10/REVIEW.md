# 修复后独立代码审查

审查基线为 `c5882ba16ace037e4ce6bab0a509a917db16df06`，范围覆盖原审计 AUD001–AUD008 对应的生产修改和定向回归。最终全量门禁在仅新增本审查报告的稳定 HEAD `f084e92bab08c8110d7a640360d0a6205c6d6302` 上完成。当前结论是：**未发现剩余阻断项，原审计八项问题均已修复并通过最终验收。**

## 审查范围

- AUD001–AUD002：`d5c27b3`、`c593b50`、`81e952e`，包括 DynamicQuantizeLinear 的公开 float32 scale 一致性、零范围输入，以及 Reduce omitted axes/noop 语义。
- AUD003–AUD005：`8ccf831`、`ed0bd6c`、`02bfe46`，包括零长度 Scan、If 的 sequence/optional 容器保持，以及父模型完整 domain/opset 向 If、Loop、Scan、SequenceMap 传播。
- AUD006–AUD008：`5cf957b`、`3db6b3e`、`c5882ba`，包括非正 iterations 拒绝、整数精确比较、CUDA 调用隔离、sidecar 解析和所有退出路径清理。

## 结论与依据

1. **AUD001：通过。** C 后端先物化 float32 scale，再用该公开值计算 zero point 和量化输出，避免返回参数与实际量化参数不一致。回归同时覆盖审计触发值和零范围输入，三个输出的 dtype、shape、值均作精确断言。

2. **AUD002：通过。** omitted axes 保留为空直到 `noop_with_empty_axes` 判定；noop=1 返回同 dtype、同 shape 的输入，noop=0 才展开为全轴归约。运行路径、shape-only 路径、导入路径和 fallback 路径均有覆盖。

3. **AUD003：通过。** Scan 零次迭代不会执行 dummy body，也不会虚构未知维。空 scan 输出只由运行时输入切片形状、body 输出元数据和 ONNX shape inference 推导；无法确定的维度明确失败。测试覆盖初始 state、多 scan 输出、非默认 input/output axis、负 output axis、多个 dtype 和符号维绑定。

4. **AUD004：通过。** If 根据声明的 `TypeProto` 递归转换 ReferenceEvaluator 输出。Optional 仅在 optional 分支解开 ReferenceEvaluator 的单元素容器，普通 sequence 不会被无差别剥离。`Optional[Tensor]` 与 `Optional[Sequence[Tensor]]` 的 present/empty 情况均有回归。

5. **AUD005：通过。** 导入器把完整父模型 `{domain: version}` 映射传入 If、Loop、Scan、SequenceMap；临时 ReferenceEvaluator 模型和 shape inference 使用同一映射。补充审查后，回归已从 If 扩展到四类控制流算子，并真实执行各自嵌套的 `ai.onnx.ml:Binarizer`。直接构造器未传映射时仍使用默认域 opset 17，兼容既有调用。

6. **AUD006：通过。** numerical CLI、`verify_all.py` 和直接 `verify_op` API 均在执行计划或清理产物前拒绝零和负 iterations。CLI 返回 argparse RC 2，直接 API 抛出 `ValueError`；回归还确认参数拒绝不会先删除现有 cache sentinel。

7. **AUD007：通过。** 所有适用的整数输出路径——普通输出、TopK、Split 和 Unique——共用同一比较函数。NPS 实际 dtype 必须严格等于声明 dtype；typed integer wire 在转换前以 Python 整数检查目标范围；浮点 wire 必须有限、无小数，并满足 `abs(value) < 2**p`，其中 float32 的边界为 `2**24`、float64 的边界为 `2**53`。因此边界及以上的值 fail closed，不能在转换后静默相等。回归通过真实 `verify_op` 调用链 fault injection 覆盖 NPS dtype 错误、Split uint32、Unique uint32 和 TopK uint64。

8. **AUD008：通过。** 每次 CUDA 调用使用独占 `TemporaryDirectory`；输入、参数、主输出和固定文件名 sidecar 都位于同一调用目录。主输出和全部 sidecar 在目录作用域内完成大小校验、解析和内存复制，返回后统一清理。默认无 sidecar 调用继续返回 ndarray，有 sidecar 调用返回 `CudaRunResult`。代码中已不存在 thread-local artifact 状态、keep/delete-next 协议或调用方手动清理。已逐项核对 RNN/GRU/LSTM、Dropout、BatchNormalization、LayerNormalization、SoftmaxCrossEntropyLoss、TopK 和 Unique 的 sidecar 消费点。

## 审查中发现并已修复的问题

- 初版 harness 迁移只在普通输出和 TopK 使用整数比较，Split 与非-int64 Unique 仍先转 float32。独立 probe 证明 `uint32(16777217)` 与 float32 wire `16777216` 会误判通过。`c5882ba` 将比较逻辑移到共享模块，并让 Split、Unique 走同一 fail-closed 路径。
- 初版共享比较函数先把 NPS 输出强制转换为声明整数 dtype，可能把错误的 `float32[1.5]` 截断成 `uint32[1]` 后误判相等。`c5882ba` 改为先检查 NPS 实际 dtype，并移除 reduce 输出归一化中的无条件 float32 转换；真实 `verify_op` 链回归覆盖了该场景。

上述问题均已在稳定提交中修复，最终复审未发现新的可操作问题。

## 已完成的定向验证

| 范围 | 结果 | RC | 证据 |
|---|---:|---:|---|
| DynamicQuantizeLinear 与 Reduce 语义 | 38 passed | 0 | `numeric_semantics.stdout_stderr.log`、`numeric_semantics.rc` |
| 控制流、sequence、optional | 11 passed | 0 | `control_targeted.stdout_stderr.log`、`control_targeted.rc` |
| harness 非 GPU 定向组 | 67 passed，10 deselected | 0 | `harness_finish_targeted.stdout.log`、`harness_finish_targeted.rc.txt` |
| 真实 GPU 并发 Add 与 Unique 四输出 | 1 passed | 0 | `harness_finish_gpu.stdout.log`、`harness_finish_gpu.rc.txt` |

## 最终全量门禁与限制

最终门禁全部在稳定 HEAD `f084e92bab08c8110d7a640360d0a6205c6d6302` 上执行，运行前 tracked status 为空：

| 门禁 | 结果 | RC | 证据 |
|---|---:|---:|---|
| CPU 全量、静态检查、严格覆盖、代表模型与图验证 | 482 passed，11 skipped | 0 | `gate_cpu.meta.log`、`gate_cpu.rc`、`gate_cpu.stdout_stderr.log` |
| CUDA verifier fresh 重编 | 178/178 compiled，0 skipped | 0 | `gate_compile_cuda.meta.log`、`gate_compile_cuda.rc`、`gate_compile_cuda.stdout_stderr.log` |
| fresh CUDA protocol 全组 | 14 passed | 0 | `gate_protocol.meta.log`、`gate_protocol.rc`、`gate_protocol.stdout_stderr.log` |
| 真实 GPU 并发 Add 与 Unique 四输出 | 1 passed | 0 | `gate_gpu_harness.meta.log`、`gate_gpu_harness.rc`、`gate_gpu_harness.stdout_stderr.log` |
| 原生 numerical | 723/723 plans；2169 iterations；0 failed；0 exceptions | 0 | `gate_numerical.meta.log`、`gate_numerical.rc`、`gate_numerical.stdout_stderr.log` |

CPU 全量中的 11 个 skip 均已核对：9 个 CUDA protocol 用例和 1 个真实 GPU harness 用例因 CPU 门禁发生在 fresh CUDA 编译之前而跳过，随后均由上述 fresh GPU 门禁通过；另 1 个是 ONNX 17 明确不支持的 Celu float16 情况。

ONNX ReferenceEvaluator 对零长度 Scan 的自身空堆叠限制仍存在；该项以 schema 推导和产品侧精确 pytest 断言验收。完整环境、命令、时间戳及证据说明见同目录的 `VALIDATION.md`。
