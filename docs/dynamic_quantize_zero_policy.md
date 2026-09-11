# DynamicQuantizeLinear 全零输入策略

## 策略

对 finite、nonempty 且所有元素均为浮点零的 DynamicQuantizeLinear 输入，本项目返回：

- `y`：与输入同 shape 的 uint8 全零 tensor；
- `y_scale`：scalar float32，bit-exact 等于 float32 `1.0f / 255.0f`；
- `y_zero_point`：scalar uint8 `0`。

浮点比较中 `+0.0 == -0.0`，因此全 `+0`、全 `-0` 和任意 `+0/-0` 混合都使用同一策略。三个输出必须由同一个分支产生，公开 scale 与量化路径使用的 scale 一致。

## 兼容来源

该策略选择对齐 ONNX 官方 Python 包的 `ReferenceEvaluator`。当前核对环境为 ONNX `1.21.0`，安装包实现位于 `onnx/reference/ops/op_dynamic_quantize_linear.py`：当 anchored extrema 满足 `maxx == minx` 时先选择 float32 `1.0`，随后除以 float32 `255`，得到 float32 `1/255`。升级追溯使用已核查的 pinned 官方源码：[op_dynamic_quantize_linear.py lines 11–27](https://github.com/onnx/onnx/blob/27d7d6890cb8bfa7ed5cda2f2656f82b6af0736a/onnx/reference/ops/op_dynamic_quantize_linear.py#L11-L27)。

这是项目选择的 ReferenceEvaluator 兼容行为。它不应表述为 opset 17 schema 强制要求；历史复审已记录 schema function 展开、ReferenceEvaluator 与其他运行时在全零边界存在差异。

## 判定边界

全零分支必须从输入值判定，并同时满足：

1. 元素数大于 0；
2. 每个值均为 finite；
3. 每个值均满足 `value == 0.0f`。

不得用 `min == max`、`range == 0` 或 `scale == 0` 替代该判定。

非零 float32 subnormal 可能在除以 255 后使 scale 下溢为 0。这类输入不是全零，继续走 normal 路径，并暂时保留现有 `scale=1.0f` 除零保护。该保护与全零 `1/255` 策略是两个独立 contract，测试必须能区分。

本策略不扩展空输入、NaN 或 infinity 的支持，也不为其声明新的输出行为。

## 回归要求

变更全零策略时至少验证：

- C runtime 与 CUDA reference 的全 `+0`、全 `-0` 或混合 `+0/-0` 三输出 bit-exact 一致；
- `y_scale` 的 float32 bits 能区分 `1/255`、`1` 和 `0`；
- 非零 subnormal 下溢继续进入 normal 保护路径；
- 既有普通 DQL 数值、REAUD-001 float32 rounding boundary 和特殊输出 contract 不回归；
- numerical harness 的全零 plans 比较全部三个输出，且不使用会掩盖 scale profile 错误的宽松容差。
