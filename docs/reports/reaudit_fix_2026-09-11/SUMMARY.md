# ONNX 边界复审修复总结

## 结论

本轮已修复 2026-09-10 边界复审确认的 7 类 P2 问题，并完成源码冻结、独立交叉审阅和完整门禁。最终源码 HEAD 为 `b8d90098176f0bec35c1ddea334781697f9881b7`；完整 numerical 最终结果为 **723/723 plans 通过，每项 3/3，共 2169 次迭代，异常 0，RC 0，stderr 为空**。

此前复审原始材料位于本地未跟踪目录 `docs/reports/reaudit_2026-09-10/`，仅作为历史审计来源，不是远端结论所依赖的链接。本轮已提交的源码、测试和本目录证据足以独立复核。

## 修复内容

1. **DynamicQuantizeLinear float32 阶段**：[`tensor_ops_dynamic_quant.c`](../../../tensor_ops/tensor_ops_dynamic_quant.c) 按 FLOAT 语义物化 extrema、range、scale、zero point quotient 和逐元素 quotient。
2. **QuantizeLinear opset 24 precision**：[`tensor_ops_quantize_linear.c`](../../../tensor_ops/tensor_ops_quantize_linear.c) 采用“显式 `precision` 优先，否则按 scale dtype”的除法精度，并在 FLOAT16/BFLOAT16 下物化操作数和 quotient。
3. **Loop 零次 symbolic scan**：[`common.py`](../../../nn/operators/common.py) 与 [`sequence_optional_control.py`](../../../nn/operators/sequence_optional_control.py) 从实际 carried state 绑定符号维，无法解析时明确失败。
4. **Loop Sequence carried state**：carried sequence 保持容器类型，不再经 `np.asarray` 错误压成 Tensor。
5. **空 SequenceMap 多输出**：按 body 声明输出数预建 buckets，空输入也返回正确数量的空 sequence。
6. **Scan negative output axis**：runtime、empty 和 shape-only 路径统一按最终 rank 规范化并校验轴。
7. **特殊输出比较契约**：[`output_contracts.py`](../../../tools/numerical/output_contracts.py)、[`runner.py`](../../../tools/numerical/runner.py) 和 [`runner_special_outputs.py`](../../../tools/numerical/runner_special_outputs.py) 在 cast、round、clip、reshape 前验证 DQL、TopK、Unique、Dropout 的 dtype、shape 和 wire 语义。

QuantizeLinear 修复依赖现有 half codec 的最小 subnormal RNE 边界，因此同步修正了 [`tensor_ops_dtype.h`](../../../tensor_ops/tensor_ops_dtype.h) 的 `shift==24` 处理。它是 REAUD-002 的必要依赖，不另计第 8 个 finding。

首轮完整 numerical 进一步暴露 CUDA oracle 仍把 FLOAT16/BFLOAT16 压成 float32 mode。第二个独立提交在 [`verify_quantize_linear.cu`](../../../cuda/verify_quantize_linear.cu) 与 [`runner_cuda_params.py`](../../../tools/numerical/runner_cuda_params.py) 中加入 0=double、1=float、2=half、3=bfloat16 的 reference precision 协议。该修改经过独立静态审阅与真实 GPU 回归，修复的是 oracle，不改变 C 端期望。

## 持久回归

- [`test_reaudit_quantization_precision.py`](../../../tests/test_reaudit_quantization_precision.py)：DQL bit-exact、QL 默认/显式 precision、half subnormal 和操作数转换边界。
- [`test_reaudit_control_boundaries.py`](../../../tests/test_reaudit_control_boundaries.py)：四类控制流最小合法模型及错误边界。
- [`test_reaudit_special_output_contracts.py`](../../../tests/test_reaudit_special_output_contracts.py)：8 个原 fault injection、合法对照及 CLI 非零退出。
- [`test_reaudit_quantization_cuda.py`](../../../tests/test_reaudit_quantization_cuda.py)：独立 half/bfloat16 staged oracle、mode 编码与真实 GPU 方向。

## 明确边界

全零 DynamicQuantizeLinear 继续采用既有 `scale=1`。schema function body、ONNX ReferenceEvaluator 与 ONNX Runtime 的该边界仍不一致，本轮没有选择新的兼容解释，也不把它列为已解决或新增 finding。

完整证据与两次 numerical 的原始结果见 [VALIDATION.md](VALIDATION.md)。

## 主目录阶段一同步

主目录已从审计基线 fast-forward 到最终源码 HEAD `b8d90098176f0bec35c1ddea334781697f9881b7`，同步前后 tracked worktree 与 index 均干净。随后主目录重新执行 C 构建，得到与隔离区一致的 `tensor_ops.so` SHA-256；单目标重编 QuantizeLinear verifier 成功，并在主目录通过 20 项真实 C/GPU 定向回归、0 skip、RC 0。完整结果见 [MAIN_SYNC_RESULT.md](MAIN_SYNC_RESULT.md)。

报告提交后的主目录第二次 fast-forward 尚未在本报告定稿时执行，因此这里不预写其完成状态。
