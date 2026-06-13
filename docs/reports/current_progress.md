<!--
/**
  ******************************************************************************
  * @file        current_progress.md
  * @author      Egor Izmaylov
  * @brief       记录当前工程进度、未完成事项和后续风险。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
-->

# 当前工程进度与未完成事项

> 记录时间：2026-06-13
> 主要依据：`docs/reports/operator_coverage.md`、`docs/reports/progress_handoff.md`、最近一次完整 numerical 记录以及当前仓库状态。

## 当前进度

- 仓库结构已经完成整理，根目录仅保留仓库级入口文件：`README.md`、`AGENTS.md`、`Makefile` 和 `requirements.txt`。
- 旧的脚本入口已经收敛到 `tools/cli.py` 和 `tools/commands/`；`create_model`、`compile_cuda`、`numerical`、`verify_graph` 等流程通过统一 CLI 调度。
- `nn/Operators.py` 和 `nn/ONNXImport.py` 保留兼容入口；实际算子实现位于 `nn/operators/`，ONNX 导入逻辑位于 `nn/importer/`。
- C 后端已经按领域拆分到 `tensor_ops/tensor_ops_*.c`，公共 ABI 保留在 `tensor_ops/tensor_ops.h`，内部公共工具集中在 `tensor_ops/tensor_ops_internal.h`。
- 文档已经集中到 `docs/`，覆盖报告和历史验证记录位于 `docs/reports/`。

## 算子覆盖状态

- Python 算子类：`201` 个。
- ONNXImport 显式支持：`202` 个 ONNX op 名称，其中 `Upsample` 作为 `Resize` 别名处理。
- ONNX opset 17 默认 domain 名称级覆盖：`178/178`。
- 当前安装 ONNX 最新默认 domain schema 名称级覆盖：`200/200`。
- forward 实际接入 C 后端：`178` 个算子类。
- 合理保留 Python 调度、控制流、序列、可选值、字符串、图像 IO 或元数据运行时：`23` 个算子类。
- 普通数值/张量算子 Python-only 运行时：`0` 个。
- CUDA verifier：`165` 个。
- 默认 active numerical plan：`165` 个唯一算子名称，`586` 条默认计划。
- 默认 active numerical plan 混合精度覆盖：`409` 条计划。

## 最近已完成验证

- `python -m py_compile tools/numerical/cli.py tools/numerical/runner.py`
- `git diff --check`
- `python tools/cli.py compile-cuda`
  - 最近记录结果：`165` 个 CUDA verifier 编译成功。
- `python tools/cli.py numerical --op bitwise_and --op bitwise_or --op bitwise_xor --op bitwise_not --op bit_shift --iterations 3 --skip-plots`
  - 最近记录结果：位运算 targeted numerical 通过。
- `python tools/cli.py numerical --op tril --op triu --op trilu --op hann_window --op hamming_window --op blackman_window --iterations 3 --skip-plots`
  - 最近记录结果：三角矩阵和窗函数 targeted numerical 通过。
- `python tools/cli.py numerical --op range --op one_hot --op reverse_sequence --op det --op mel_weight_matrix --iterations 3 --skip-plots`
  - 最近记录结果：形状/索引、线性代数和特征矩阵 targeted numerical 通过。
- `python -m pytest -q tests/test_operator_misc_semantics.py tests/test_operator_c_backend.py -k "bitwise or bit_shift or unsigned_integer_binary_ops"`
  - 最近记录结果：相关 pytest 通过。
- `python tools/audit_ops.py --output docs/reports/operator_coverage.md`
  - 最近记录结果：覆盖报告已刷新。
- `python tools/cli.py numerical --iterations 1 --skip-plots`
  - 最近记录结果：完整默认 numerical 一轮通过。

## 已知未完成部分

以下 `12` 个算子已有 C runtime path、C 后端函数或较完整 pytest/ONNX reference 语义覆盖，但尚未全部具备独立 CUDA verifier 和默认 numerical plan。后续补强时应继续优先使用 C/CUDA reference，不应回退为普通数值路径的 Python-only 实现。

- 随机/采样：`Bernoulli`、`Multinomial`、`RandomNormal`、`RandomNormalLike`、`RandomUniform`。
- 随机掩码：`Dropout`。
- 损失/检测：`NegativeLogLikelihoodLoss`、`SoftmaxCrossEntropyLoss`、`NonMaxSuppression`。
- 量化：`DynamicQuantizeLinear`。
- 集合：`Unique`。
- 结构拆分：`Split` 当前通过 `slice_forward` 执行并已有 pytest/ONNX reference 覆盖，但还没有独立默认 numerical plan。

以下类别属于合理保留 Python 调度的范围，不适合按普通 C/CUDA 数值门禁处理；后续重点应放在 ONNX reference pytest、导入器语义测试和端到端图测试。

- 控制流：`If`、`Loop`、`Scan`。
- 序列：`SequenceEmpty`、`SequenceConstruct`、`SequenceAt`、`SequenceInsert`、`SequenceErase`、`SequenceLength`、`SequenceMap`、`ConcatFromSequence`、`SplitToSequence`。
- 可选值：`Optional`、`OptionalGetElement`、`OptionalHasElement`。
- 字符串：`RegexFullMatch`、`StringConcat`、`StringSplit`、`StringNormalizer`、`TfIdfVectorizer`。
- 元数据/常量/图像：`Shape`、`Constant`、`ImageDecoder`。

## 混合精度状态

- 当前默认 numerical 中已经包含 `409` 条混合精度计划，覆盖 float16、bfloat16、部分 float8 以及相关低精度存储路径。
- 混合精度已经能作为当前工程的常规回归门禁使用，但还不能宣称对所有 ONNX 官方 type constraint、所有属性组合和所有边界输入完成穷尽证明。
- 位运算、字符串、序列、控制流、随机采样等类别不应机械纳入浮点混合精度口径；这些算子需要按整数位模式、结构语义、随机分布或 ONNX reference 行为分别验证。

## 剩余风险

- ONNX opset 17 已达到名称级导入覆盖，但名称级覆盖不等同于所有属性、边界条件、异常路径和高维组合的官方语义穷尽验证。
- 默认 numerical 是固定 case 与随机样本组成的工程门禁，能够发现常见回归，但不是形式化证明。
- `MaxRoiPool`、`RoiAlign`、`RNN`、`GRU`、`LSTM`、`DFT`、`STFT` 已进入默认门禁，但仍建议继续扩展 layout、direction、axis、window 和边界输入。
- `Attention`、`DeformConv` 等复杂算子已覆盖主 C/CUDA 路径，部分 cache、nonpad、多输出或高维 fallback 仍主要依赖 ONNX reference pytest。
- 随机、损失、NMS、量化和集合类算子补 CUDA reference 时，需要特别注意确定性种子、阈值、排序稳定性和低精度舍入规则。

## 建议后续优先级

1. 优先处理 `Split`、`Unique`、`DynamicQuantizeLinear` 和 `Dropout`；它们相对确定，适合作为下一批收口对象。
2. 再处理 `NegativeLogLikelihoodLoss`、`SoftmaxCrossEntropyLoss` 和 `NonMaxSuppression`；这些算子需要更细的阈值、索引和归约边界设计。
3. 最后集中处理随机/采样算子：`Bernoulli`、`Multinomial`、`RandomNormal`、`RandomNormalLike`、`RandomUniform`；建议先确定 seed、分布容差和 reference 统计口径，再进入默认 numerical。
