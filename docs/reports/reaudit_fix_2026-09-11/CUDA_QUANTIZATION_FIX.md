# QuantizeLinear CUDA precision oracle 修复

## 原始 full numerical 结果

冻结提交 `8a6f46985ad16265c5abc4283984f62ba98be2e1` 的首次完整 numerical gate 自然结束：723 plans started，721 pass，2 fail，exception 0，stderr 为空，真实 RC `1`。仅以下默认 mixed-precision QuantizeLinear plans 失败：

- FLOAT16/FLOAT16/INT8：iter 1，输入 `-697.5` 处 C 为 `-24`、CUDA 为 `-25`；
- BFLOAT16/BFLOAT16/INT8：iter 0，输入 `49852` 所在位置 C 为 `116`、CUDA 为 `117`。

原始失败日志 `numerical_gate.*` 保持不变。

## 独立判定与修复

ONNX QuantizeLinear v24 规定：未显式设置 `precision` 时，除法精度由 `y_scale` dtype 决定；显式属性覆盖默认值。ReferenceEvaluator 会先把两个操作数转换到该精度，再除法。修复后的 C 已按该顺序处理。

旧 CUDA params 只携带 double/float 布尔值，FLOAT16 与 BFLOAT16 都落入 float32 kernel，因此这两项失败来自 CUDA oracle。协议保留 `0=DOUBLE`、`1=FLOAT`，增加 `2=FLOAT16`、`3=BFLOAT16`。CUDA kernel 对低精度模式使用 RN intrinsic 依次物化两个操作数和 quotient，再进行整数 nearest-even；params 默认按 scale dtype，显式 precision 优先。DequantizeLinear 参数布局未改。

## 单目标构建与真实 GPU 回归

- 构建命令：`python -u tools/cli.py compile-cuda --op quantize_linear --force`
- 构建结果：RC `0`，`compiled=1 skipped=0`，stderr 为空；
- 新 `cache/verify_quantize_linear` SHA-256：`0c12b072e454d8e80aeefa3ac9af1ea3f7a9d03850a69e67bc0b849d2c7bf8be`；size `1051952` bytes；mtime `2026-09-11 12:22:21.863643200 +0800`；
- `tensor_ops.so` 未重建且 SHA-256 仍为 `602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66`；
- 构建后库存仍为 178 CUDA sources / 178 executables；
- 测试命令：`python -u -m pytest -q -ra tests/test_reaudit_quantization_cuda.py`；
- 测试结果：RC `0`，`9 passed in 5.68s`，无 skip，stderr 为空。

测试覆盖真实 GPU half fixture（mode 2 为 30，旧 float32 mode 1 为 29，double mode 0 为 29）、BFLOAT16 fixture（mode 3 为 80，float32 mode 1 为 81），并检查默认 FLOAT16/BFLOAT16 与显式 FLOAT/FLOAT16/DOUBLE 的 params 编码。

原始 stdout/stderr/RC 保存在 `cuda_quantize_rebuild.*` 与 `cuda_quantize_tests.*`。独立静态复核记录见 `REVIEW_CUDA_QUANTIZATION.md`。
