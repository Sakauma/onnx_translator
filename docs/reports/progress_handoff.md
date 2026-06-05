<!--
/**
  ******************************************************************************
  * @file        progress_handoff.md
  * @author      Egor Izmaylov
  * @brief       记录当前算子官方语义对齐、混合精度验证进度和剩余工作。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
-->

# 当前进度与剩余工作交接

> 记录时间：2026-06-05
> 当前分支：`main`
> 基准报告：`docs/reports/operator_coverage.md`

## 当前总体状态

- 当前安装 ONNX 最新默认 domain schema 名称级覆盖：`200/200`。
- ONNX opset 17 默认 domain 名称级覆盖：`178/178`。
- Python 算子类：`201` 个。
- ONNXImport 显式支持：`202` 个 ONNX op 名称，其中 `Upsample` 作为 `Resize` 别名处理。
- forward 实际接入 C 后端：`178` 个算子类。
- 合理保留 Python 调度/元数据运行时：`23` 个算子类。
- 普通数值/张量算子 Python-only 运行时：`0` 个。
- CUDA verifier：`154` 个。
- 默认 active numerical plan：`154` 个唯一算子名称，`564` 条默认计划。
- 默认 active numerical plan 混合精度覆盖：`398` 条计划。

## 本轮已完成

- 新增位运算 CUDA reference：
  - `cuda/verify_bitwise_and.cu`
  - `cuda/verify_bitwise_or.cu`
  - `cuda/verify_bitwise_xor.cu`
  - `cuda/verify_bitwise_not.cu`
  - `cuda/verify_bit_shift.cu`
- `tools/numerical/cli.py` 新增 `BitwiseAnd`、`BitwiseOr`、`BitwiseXor`、`BitwiseNot` 和 `BitShift` 的默认 numerical plan。
- `tools/numerical/runner.py` 为位运算使用固定 `int32` 样本，并保持整数二进制输入直接传给 CUDA reference，避免按浮点路径转换破坏位模式。
- `BitShift` numerical 覆盖 `LEFT` 和 `RIGHT` 两个 direction。
- 位运算属于整数/布尔位模式语义，不存在 float16、bfloat16、float8 混合精度路径；当前以 `int32` 主路径进入默认数值门禁，`uint32`/`uint64` 边界由 pytest 覆盖，后续可继续补 CUDA plan。

## 本轮已运行验证

- `python -m py_compile tools/numerical/cli.py tools/numerical/runner.py`
- `git diff --check`
- `python tools/cli.py compile-cuda`
  - 结果：`154` 个 CUDA verifier 编译成功。
- `python tools/cli.py numerical --op bitwise_and --op bitwise_or --op bitwise_xor --op bitwise_not --op bit_shift --iterations 3 --skip-plots`
  - 结果：全部通过，误差为 `0`。
- `python -m pytest -q tests/test_operator_misc_semantics.py tests/test_operator_c_backend.py -k "bitwise or bit_shift or unsigned_integer_binary_ops"`
  - 结果：`2 passed, 56 deselected`。
- `python tools/audit_ops.py --output docs/reports/operator_coverage.md`
  - 结果：覆盖报告已刷新。
- `python tools/cli.py numerical --iterations 1 --skip-plots`
  - 结果：完整默认 numerical 一轮全部通过。

## 仍未完成的普通 C-backed 数值门禁

以下算子已有 C runtime path 或 C 后端函数，但当前仍缺少 CUDA verifier / 默认 numerical plan，后续应优先补独立 reference，而不是回退到 Python 数值实现：

- 随机/采样：`Bernoulli`、`Multinomial`、`RandomNormal`、`RandomNormalLike`、`RandomUniform`。
- 损失/检测：`NegativeLogLikelihoodLoss`、`SoftmaxCrossEntropyLoss`、`NonMaxSuppression`。
- 窗函数/特征：`HannWindow`、`HammingWindow`、`BlackmanWindow`、`MelWeightMatrix`。
- 量化/阈值：`DynamicQuantizeLinear`、`Binarizer`。
- 形状/索引/集合：`Range`、`OneHot`、`ReverseSequence`、`Unique`。
- 线性代数/三角：`Det`、`Tril`、`Triu`、`Trilu`。
- 结构拆分：`Split` 当前通过 `slice_forward` 执行并已有 pytest/ONNX reference 覆盖，但还没有独立默认 numerical plan。

## 合理保留 Python 调度的范围

以下类别主要是控制流、序列、可选值、字符串、图像 IO 或元数据算子，不适合按普通 C/CUDA 数值门禁处理；后续应继续依赖 ONNX reference pytest、导入器语义测试和端到端图测试：

- 控制流：`If`、`Loop`、`Scan`。
- 序列：`SequenceEmpty`、`SequenceConstruct`、`SequenceAt`、`SequenceInsert`、`SequenceErase`、`SequenceLength`、`SequenceMap`、`ConcatFromSequence`、`SplitToSequence`。
- 可选值：`Optional`、`OptionalGetElement`、`OptionalHasElement`。
- 字符串：`RegexFullMatch`、`StringConcat`、`StringSplit`、`StringNormalizer`、`TfIdfVectorizer`。
- 元数据/常量/图像：`Shape`、`Constant`、`ImageDecoder`。

## 后续建议优先级

1. 优先补 `Tril`、`Triu`、`Trilu`：三者共用 triangular C 后端，CUDA reference 和 numerical plan 可共用同一套参数，收益高且风险低。
2. 补窗函数 `HannWindow`、`HammingWindow`、`BlackmanWindow`：输入输出简单，容易扩展 float32/float16/bfloat16 主路径。
3. 补 `Range`、`OneHot`、`ReverseSequence`：官方语义明确，但需要注意输出 shape、轴参数和整数输入 dtype。
4. 再处理损失、随机、NMS、DynamicQuantizeLinear 等高风险算子：这些需要更细的确定性 reference、随机种子或阈值边界设计。

## 当前剩余风险

- 默认 numerical 是固定 case 与随机样本的门禁，不等同于 ONNX 所有 opset schema 的穷尽证明。
- 已进入 numerical 的 mixed precision 计划主要覆盖项目当前支持和官方 type constraint 中合理的低精度路径；非官方约束内的 float8 或字符串/序列路径不应强行纳入数值门禁。
- `MaxRoiPool`、`RoiAlign`、`RNN`、`GRU`、`LSTM`、`DFT`、`STFT` 已进入默认门禁，但仍建议继续扩展更多 layout、direction、axis、window 和边界输入。
- `Attention`、`DeformConv` 等复杂算子已覆盖主 C/CUDA 路径，部分 cache、nonpad、多输出或高维 fallback 仍主要由 ONNX reference pytest 覆盖。
