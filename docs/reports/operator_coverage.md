<!--
/**
  ******************************************************************************
  * @file        operator_coverage.md
  * @author      Egor Izmaylov
  * @brief       记录算子实现情况评估、验证覆盖和剩余风险。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
-->

# 算子实现情况评估

> 自动生成时间：2026-06-05 08:33:30
> 生成命令：`python tools/audit_ops.py --output docs/reports/operator_coverage.md`

## 评估口径

- `ONNXImport`：`nn/ONNXImport.py` 中存在显式映射，可从 ONNX node 构造对应算子类。
- `forward`：`nn/Operators.py` 中存在运行时前向实现，继承自 `ReduceBase`/`ArgBase` 的实现也计入。
- `forward_`：存在图构建/形状推断实现，继承实现也计入。
- `C backend funcs`：Python 算子类中引用的 `<op>_forward` C 函数均能在 `tensor_ops/*.c` 中找到，且 `tensor_ops.h` 与 `.c` 声明/实现集合一致。
- `C runtime path`：`forward()` 运行路径实际引用 `<op>_forward` C 函数；仅在 `__init__`、形状推断或未调用 helper 中出现不计入。
- `CUDA verifier`：`cuda/verify_<op>.cu` 存在，仅说明有参考验证程序源码。
- `active numerical plan`：统一入口 `python tools/cli.py numerical` 对应的 `tools/numerical/cli.py` 默认计划中包含该算子，代表会被默认数值验证门禁执行。
- `独立 pytest 深度语义/混合精度覆盖`：使用独立公式或 ONNX reference 对高风险算子的官方语义、边界条件和低精度 dtype 路径进行 pytest 回归验证。
- `ONNX reference pytest 语义/混合精度覆盖`：使用本地 ONNX reference evaluator 对普通算子的官方输出和低精度 dtype 路径进行 pytest 回归验证。
- `ONNX opset 17 官方覆盖`：通过本地 `onnx.defs` 读取默认 domain 中 `since_version <= 17` 的最新 schema，并与 `ONNXImport` 的显式映射做名称级对比。
- `当前安装 ONNX 最新官方覆盖`：读取当前环境可见的最新默认 domain schema，用于暴露高版本 opset 新增算子的后续兼容风险。

## 总览

- Python 算子类：196 个。
- ONNXImport 显式支持：197 个 ONNX op 名称；`Upsample` 作为 `Resize` 别名处理，未单独计入算子类。
- C 后端声明：167 个 `<op>_forward`；C 实现可检出：175 个。
- forward 实际接入 C 后端：174 个算子类。
- 合理保留 Python 调度/元数据运行时：22 个算子类。
- 普通数值/张量算子 Python-only 运行时：0 个算子类；其中当前暂缓后端化：0 个，除暂缓项外待后端化：0 个。
- CUDA verifier：126 个。
- active numerical plan 覆盖：126 个唯一算子名称，466 条默认计划。
- active numerical plan 混合精度覆盖：334 条默认计划。
- 独立 pytest 深度语义/混合精度覆盖：51 个；`Multinomial`, `SequenceEmpty`, `SequenceConstruct`, `RNN`, `SequenceAt`, `LRN`, `SequenceInsert`, `RandomUniformLike`, `SequenceErase`, `RandomUniform`, `SequenceLength`, `MeanVarianceNormalization`, `ConcatFromSequence`, `StringNormalizer`, `DFT`, `RandomNormal`, `SplitToSequence`, `RandomNormalLike`, `GRU`, `TfIdfVectorizer`, `Bernoulli`, `Optional`, `OptionalGetElement`, `STFT`, `OptionalHasElement`, `Dropout`, `If`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `MaxRoiPool`, `ReduceLogSumExp`, `Loop`, `ReduceSumSquare`, `Unique`, `LSTM`, `Gelu`, `Tril`, `RoiAlign`, `Triu`, `Mish`, `BitwiseAnd`, `Scan`, `Binarizer`, `BitwiseOr`, `BitwiseXor`, `BitwiseNot`, `BitShift`, `SequenceMap`, `GroupNormalization`, `GlobalLpPool`。
- ONNX reference pytest 语义/混合精度覆盖：145 个；`Conv`, `Elu`, `RELU`, `Equal`, `Gather`, `Gemm`, `Softmax`, `GridSample`, `QuantizeLinear`, `Shape`, `AffineGrid`, `Flatten`, `EyeLike`, `Greater`, `MelWeightMatrix`, `Selu`, `RegexFullMatch`, `Constant`, `Less`, `DequantizeLinear`, `LeakyRelu`, `GreaterOrEqual`, `StringConcat`, `ScatterND`, `NegativeLogLikelihoodLoss`, `Reshape`, `COS`, `ConstantOfShape`, `Expand`, `PRelu`, `LessOrEqual`, `ThresholdedRelu`, `StringSplit`, `Not`, `TensorScatter`, `HardSigmoid`, `And`, `MaxPool`, `ConvTranspose`, `Det`, `Range`, `ABS`, `Or`, `Celu`, `Xor`, `MatMulInteger`, `Resize`, `Shrink`, `IsNaN`, `RMSNormalization`, `Tile`, `ReduceMean`, `Transpose`, `ADD`, `ReduceSum`, `ReduceMax`, `Sin`, `ReduceMin`, `Softplus`, `ReduceProd`, `Tan`, `Softsign`, `Atan`, `Pad`, `MaxUnpool`, `HardSwish`, `SoftmaxCrossEntropyLoss`, `Sign`, `QLinearMatMul`, `Swish`, `GatherND`, `Mean`, `Identity`, `SUB`, `Acos`, `Squeeze`, `Asin`, `Mod`, `ArgMax`, `Cosh`, `IsInf`, `GatherElements`, `ArgMin`, `Sinh`, `ConvInteger`, `Asinh`, `MatMul`, `Round`, `Acosh`, `NonZero`, `MUL`, `Erf`, `CenterCropPad`, `Size`, `Atanh`, `BatchNormalization`, `Unsqueeze`, `Where`, `TopK`, `HannWindow`, `DIV`, `HammingWindow`, `QLinearConv`, `Split`, `Concat`, `Trilu`, `BlackmanWindow`, `DynamicQuantizeLinear`, `InstanceNormalization`, `CumSum`, `NonMaxSuppression`, `DepthToSpace`, `EXP`, `CumProd`, `LayerNormalization`, `LOG`, `Clip`, `Slice`, `SQRT`, `SpaceToDepth`, `OneHot`, `SIGMOID`, `TANH`, `Einsum`, `AveragePool`, `ReverseSequence`, `Pow`, `Hardmax`, `Compress`, `Max`, `LogSoftmax`, `LpPool`, `Min`, `LpNormalization`, `Cast`, `ScatterElements`, `Neg`, `CastLike`, `Reciprocal`, `GlobalAveragePool`, `Ceil`, `BitCast`, `Floor`, `GlobalMaxPool`, `Sum`。
- ONNX opset 17 官方算子：178 个；ONNXImport 名称级覆盖：178 个。
- 当前安装 ONNX 最新官方算子：200 个；ONNXImport 名称级覆盖：195 个。

### 状态计数

| 状态 | 数量 |
| --- | ---: |
| 已 pytest 语义验证 | 70 |
| 已数值验证 | 126 |

### 关键结论

- `tensor_ops.h` 与 `tensor_ops.c` 中的 C forward 函数集合一致，没有发现声明缺实现或实现缺声明。
- 所有 196 个 Python 算子类均可被 `ONNXImport` 显式映射。
- 未发现缺少 `forward` / `forward_` / C 函数映射的显式部分实现算子。
- 未发现“有 C 函数但 forward 未调用”的算子。
- 当前没有记录暂缓后端化算子。
- 已补充独立 pytest 深度语义/混合精度覆盖的算子：`Multinomial`, `SequenceEmpty`, `SequenceConstruct`, `RNN`, `SequenceAt`, `LRN`, `SequenceInsert`, `RandomUniformLike`, `SequenceErase`, `RandomUniform`, `SequenceLength`, `MeanVarianceNormalization`, `ConcatFromSequence`, `StringNormalizer`, `DFT`, `RandomNormal`, `SplitToSequence`, `RandomNormalLike`, `GRU`, `TfIdfVectorizer`, `Bernoulli`, `Optional`, `OptionalGetElement`, `STFT`, `OptionalHasElement`, `Dropout`, `If`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `MaxRoiPool`, `ReduceLogSumExp`, `Loop`, `ReduceSumSquare`, `Unique`, `LSTM`, `Gelu`, `Tril`, `RoiAlign`, `Triu`, `Mish`, `BitwiseAnd`, `Scan`, `Binarizer`, `BitwiseOr`, `BitwiseXor`, `BitwiseNot`, `BitShift`, `SequenceMap`, `GroupNormalization`, `GlobalLpPool`。
- 已补充 ONNX reference pytest 语义/混合精度覆盖的普通算子：`Conv`, `Elu`, `RELU`, `Equal`, `Gather`, `Gemm`, `Softmax`, `GridSample`, `QuantizeLinear`, `Shape`, `AffineGrid`, `Flatten`, `EyeLike`, `Greater`, `MelWeightMatrix`, `Selu`, `RegexFullMatch`, `Constant`, `Less`, `DequantizeLinear`, `LeakyRelu`, `GreaterOrEqual`, `StringConcat`, `ScatterND`, `NegativeLogLikelihoodLoss`, `Reshape`, `COS`, `ConstantOfShape`, `Expand`, `PRelu`, `LessOrEqual`, `ThresholdedRelu`, `StringSplit`, `Not`, `TensorScatter`, `HardSigmoid`, `And`, `MaxPool`, `ConvTranspose`, `Det`, `Range`, `ABS`, `Or`, `Celu`, `Xor`, `MatMulInteger`, `Resize`, `Shrink`, `IsNaN`, `RMSNormalization`, `Tile`, `ReduceMean`, `Transpose`, `ADD`, `ReduceSum`, `ReduceMax`, `Sin`, `ReduceMin`, `Softplus`, `ReduceProd`, `Tan`, `Softsign`, `Atan`, `Pad`, `MaxUnpool`, `HardSwish`, `SoftmaxCrossEntropyLoss`, `Sign`, `QLinearMatMul`, `Swish`, `GatherND`, `Mean`, `Identity`, `SUB`, `Acos`, `Squeeze`, `Asin`, `Mod`, `ArgMax`, `Cosh`, `IsInf`, `GatherElements`, `ArgMin`, `Sinh`, `ConvInteger`, `Asinh`, `MatMul`, `Round`, `Acosh`, `NonZero`, `MUL`, `Erf`, `CenterCropPad`, `Size`, `Atanh`, `BatchNormalization`, `Unsqueeze`, `Where`, `TopK`, `HannWindow`, `DIV`, `HammingWindow`, `QLinearConv`, `Split`, `Concat`, `Trilu`, `BlackmanWindow`, `DynamicQuantizeLinear`, `InstanceNormalization`, `CumSum`, `NonMaxSuppression`, `DepthToSpace`, `EXP`, `CumProd`, `LayerNormalization`, `LOG`, `Clip`, `Slice`, `SQRT`, `SpaceToDepth`, `OneHot`, `SIGMOID`, `TANH`, `Einsum`, `AveragePool`, `ReverseSequence`, `Pow`, `Hardmax`, `Compress`, `Max`, `LogSoftmax`, `LpPool`, `Min`, `LpNormalization`, `Cast`, `ScatterElements`, `Neg`, `CastLike`, `Reciprocal`, `GlobalAveragePool`, `Ceil`, `BitCast`, `Floor`, `GlobalMaxPool`, `Sum`。
- 未发现仍需立即后端化的 Python-only 普通数值/张量算子。
- 默认数值门禁当前覆盖 126 个唯一算子；尚有 0 个已实现算子未进入 active numerical plan。

## ONNX opset 17 官方覆盖

- 官方默认 domain 算子：178 个。
- `ONNXImport` 已覆盖官方名称：178 个。
- 官方名称级缺口：0 个。

### 官方缺口

无。

### 仓库额外/非默认 domain/实验性名称

`AffineGrid`, `Binarizer`, `BitCast`, `BitwiseAnd`, `BitwiseNot`, `BitwiseOr`, `BitwiseXor`, `CenterCropPad`, `CumProd`, `Gelu`, `GroupNormalization`, `Mish`, `RegexFullMatch`, `RMSNormalization`, `StringConcat`, `StringSplit`, `Swish`, `TensorScatter`, `Tril`, `Triu`

## 当前安装 ONNX 最新官方覆盖

- 该段用于暴露高版本 opset 兼容风险，不改变当前报告中 opset 17 的历史覆盖口径。
- 当前环境默认 domain 最新官方算子：200 个。
- `ONNXImport` 已覆盖官方名称：195 个。
- 最新官方名称级缺口：5 个。

### 最新官方缺口

`Attention(since_version=24)`, `Col2Im(since_version=18)`, `DeformConv(since_version=22)`, `ImageDecoder(since_version=20)`, `RotaryEmbedding(since_version=23)`

## 明细表

| # | 算子类 | ONNXImport | forward | forward_ | C backend funcs | C runtime path | CUDA verifier | active numerical plan | 状态 | 备注 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `Conv` | yes | yes | yes | `conv2d_forward` | `conv2d_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 2 | `Elu` | yes | yes | yes | `elu_forward` | `elu_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 3 | `RELU` | yes | yes | yes | `relu_forward` | `relu_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 4 | `Equal` | yes | yes | yes | `equal_forward` | `equal_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 5 | `Gather` | yes | yes | yes | `gather_forward` | `gather_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 6 | `Multinomial` | yes | yes | yes | `multinomial_forward` | `multinomial_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 7 | `Gemm` | yes | yes | yes | `gemm_forward` | `gemm_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 8 | `Softmax` | yes | yes | yes | `softmax_forward` | `softmax_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 9 | `GridSample` | yes | yes | yes | `grid_sample_forward` | `grid_sample_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 10 | `QuantizeLinear` | yes | yes | yes | `quantize_linear_forward` | `quantize_linear_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 11 | `SequenceEmpty` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 12 | `Shape` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 13 | `AffineGrid` | yes | yes | yes | `affine_grid_forward` | `affine_grid_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 14 | `Flatten` | yes | yes | yes | `flatten_forward` | `flatten_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 15 | `EyeLike` | yes | yes | yes | `eye_like_forward` | `eye_like_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 16 | `Greater` | yes | yes | yes | `greater_forward` | `greater_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 17 | `SequenceConstruct` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 18 | `MelWeightMatrix` | yes | yes | yes | `mel_weight_matrix_forward` | `mel_weight_matrix_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 19 | `Selu` | yes | yes | yes | `selu_forward` | `selu_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 20 | `RegexFullMatch` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 21 | `RNN` | yes | yes | yes | `rnn_forward` | `rnn_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
| 22 | `Constant` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 23 | `Less` | yes | yes | yes | `less_forward` | `less_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 24 | `SequenceAt` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 25 | `DequantizeLinear` | yes | yes | yes | `dequantize_linear_forward` | `dequantize_linear_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 26 | `LRN` | yes | yes | yes | `lrn_forward` | `lrn_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 27 | `LeakyRelu` | yes | yes | yes | `leaky_relu_forward` | `leaky_relu_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 28 | `GreaterOrEqual` | yes | yes | yes | `greater_or_equal_forward` | `greater_or_equal_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 29 | `SequenceInsert` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 30 | `RandomUniformLike` | yes | yes | yes | `random_uniform_like_forward` | `random_uniform_like_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
| 31 | `StringConcat` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 32 | `ScatterND` | yes | yes | yes | `scatter_nd_forward` | `scatter_nd_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 33 | `NegativeLogLikelihoodLoss` | yes | yes | yes | `negative_log_likelihood_loss_forward` | `negative_log_likelihood_loss_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 34 | `Reshape` | yes | yes | yes | `reshape_forward` | `reshape_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 35 | `COS` | yes | yes | yes | `cos_forward` | `cos_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 36 | `ConstantOfShape` | yes | yes | yes | `constant_of_shape_forward` | `constant_of_shape_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 37 | `Expand` | yes | yes | yes | `expand_forward` | `expand_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 38 | `PRelu` | yes | yes | yes | `prelu_forward` | `prelu_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 39 | `LessOrEqual` | yes | yes | yes | `less_or_equal_forward` | `less_or_equal_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 40 | `SequenceErase` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 41 | `ThresholdedRelu` | yes | yes | yes | `thresholded_relu_forward` | `thresholded_relu_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 42 | `StringSplit` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 43 | `Not` | yes | yes | yes | `not_forward` | `not_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 44 | `TensorScatter` | yes | yes | yes | `tensor_scatter_forward` | `tensor_scatter_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 45 | `RandomUniform` | yes | yes | yes | `random_uniform_like_forward` | `random_uniform_like_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 46 | `SequenceLength` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 47 | `MeanVarianceNormalization` | yes | yes | yes | `mean_variance_normalization_forward` | `mean_variance_normalization_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 48 | `HardSigmoid` | yes | yes | yes | `hard_sigmoid_forward` | `hard_sigmoid_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 49 | `And` | yes | yes | yes | `and_forward` | `and_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 50 | `MaxPool` | yes | yes | yes | `max_pool_forward` | `max_pool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 51 | `ConvTranspose` | yes | yes | yes | `conv_transpose2d_forward` | `conv_transpose2d_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 52 | `Det` | yes | yes | yes | `det_forward` | `det_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 53 | `ConcatFromSequence` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 54 | `Range` | yes | yes | yes | `range_forward` | `range_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 55 | `StringNormalizer` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 56 | `ABS` | yes | yes | yes | `abs_forward` | `abs_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 57 | `Or` | yes | yes | yes | `or_forward` | `or_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 58 | `DFT` | yes | yes | yes | `dft_forward` | `dft_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
| 59 | `Celu` | yes | yes | yes | `celu_forward` | `celu_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 60 | `RandomNormal` | yes | yes | yes | `random_normal_forward` | `random_normal_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 61 | `Xor` | yes | yes | yes | `xor_forward` | `xor_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 62 | `MatMulInteger` | yes | yes | yes | `matmul_integer_forward` | `matmul_integer_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 63 | `Resize` | yes | yes | yes | `resize_forward` | `resize_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 64 | `Shrink` | yes | yes | yes | `shrink_forward` | `shrink_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 65 | `SplitToSequence` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 66 | `IsNaN` | yes | yes | yes | `isnan_forward` | `isnan_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 67 | `RMSNormalization` | yes | yes | yes | `rms_normalization_forward` | `rms_normalization_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 68 | `Tile` | yes | yes | yes | `tile_forward` | `tile_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 69 | `ReduceMean` | yes | yes | yes | `reduce_mean_forward` | `reduce_mean_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 70 | `Transpose` | yes | yes | yes | `transpose_forward` | `transpose_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 71 | `ADD` | yes | yes | yes | `add_forward` | `add_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 72 | `ReduceSum` | yes | yes | yes | `reduce_sum_forward` | `reduce_sum_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 73 | `ReduceMax` | yes | yes | yes | `reduce_max_forward` | `reduce_max_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 74 | `Sin` | yes | yes | yes | `sin_forward` | `sin_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 75 | `RandomNormalLike` | yes | yes | yes | `random_normal_forward` | `random_normal_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 76 | `ReduceMin` | yes | yes | yes | `reduce_min_forward` | `reduce_min_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 77 | `Softplus` | yes | yes | yes | `softplus_forward` | `softplus_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 78 | `ReduceProd` | yes | yes | yes | `reduce_prod_forward` | `reduce_prod_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 79 | `Tan` | yes | yes | yes | `tan_forward` | `tan_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 80 | `Softsign` | yes | yes | yes | `softsign_forward` | `softsign_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 81 | `Atan` | yes | yes | yes | `atan_forward` | `atan_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 82 | `Pad` | yes | yes | yes | `pad_forward` | `pad_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 83 | `MaxUnpool` | yes | yes | yes | `max_unpool_forward` | `max_unpool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 84 | `HardSwish` | yes | yes | yes | `hard_swish_forward` | `hard_swish_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 85 | `SoftmaxCrossEntropyLoss` | yes | yes | yes | `softmax_cross_entropy_loss_forward` | `softmax_cross_entropy_loss_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 86 | `GRU` | yes | yes | yes | `gru_forward` | `gru_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
| 87 | `TfIdfVectorizer` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 88 | `Bernoulli` | yes | yes | yes | `bernoulli_forward` | `bernoulli_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 89 | `Optional` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 90 | `Sign` | yes | yes | yes | `sign_forward` | `sign_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 91 | `QLinearMatMul` | yes | yes | yes | `qlinear_matmul_forward` | `qlinear_matmul_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 92 | `Swish` | yes | yes | yes | `swish_forward` | `swish_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 93 | `GatherND` | yes | yes | yes | `gather_nd_forward` | `gather_nd_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 94 | `Mean` | yes | yes | yes | `mean_forward` | `mean_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 95 | `Identity` | yes | yes | yes | `identity_forward` | `identity_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 96 | `OptionalGetElement` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 97 | `SUB` | yes | yes | yes | `sub_forward` | `sub_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 98 | `Acos` | yes | yes | yes | `acos_forward` | `acos_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 99 | `Squeeze` | yes | yes | yes | `reshape_forward` | `reshape_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 100 | `STFT` | yes | yes | yes | `stft_forward` | `stft_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
| 101 | `OptionalHasElement` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 102 | `Dropout` | yes | yes | yes | `dropout_forward` | `dropout_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 103 | `Asin` | yes | yes | yes | `asin_forward` | `asin_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 104 | `Mod` | yes | yes | yes | `mod_forward` | `mod_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 105 | `ArgMax` | yes | yes | yes | `argmax_forward` | `argmax_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 106 | `If` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 107 | `Cosh` | yes | yes | yes | `cosh_forward` | `cosh_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 108 | `IsInf` | yes | yes | yes | `isinf_forward` | `isinf_forward` | no | no | 已 pytest 语义验证 | 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 109 | `GatherElements` | yes | yes | yes | `gather_elements_forward` | `gather_elements_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 110 | `ArgMin` | yes | yes | yes | `argmin_forward` | `argmin_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 111 | `ReduceL1` | yes | yes | yes | `reduce_l1_forward` | `reduce_l1_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
| 112 | `Sinh` | yes | yes | yes | `sinh_forward` | `sinh_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 113 | `ReduceL2` | yes | yes | yes | `reduce_l2_forward` | `reduce_l2_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
| 114 | `ConvInteger` | yes | yes | yes | `conv_integer_forward` | `conv_integer_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 115 | `ReduceLogSum` | yes | yes | yes | `reduce_log_sum_forward` | `reduce_log_sum_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
| 116 | `MaxRoiPool` | yes | yes | yes | `max_roi_pool_forward` | `max_roi_pool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
| 117 | `ReduceLogSumExp` | yes | yes | yes | `reduce_log_sum_exp_forward` | `reduce_log_sum_exp_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
| 118 | `Asinh` | yes | yes | yes | `asinh_forward` | `asinh_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 119 | `MatMul` | yes | yes | yes | `matmul_forward` | `matmul_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 120 | `Loop` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 121 | `ReduceSumSquare` | yes | yes | yes | `reduce_sum_square_forward` | `reduce_sum_square_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
| 122 | `Round` | yes | yes | yes | `round_forward` | `round_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 123 | `Acosh` | yes | yes | yes | `acosh_forward` | `acosh_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 124 | `NonZero` | yes | yes | yes | `nonzero_forward` | `nonzero_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 125 | `Unique` | yes | yes | yes | `unique_forward` | `unique_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 126 | `MUL` | yes | yes | yes | `mul_forward` | `mul_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 127 | `Erf` | yes | yes | yes | `erf_forward` | `erf_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 128 | `CenterCropPad` | yes | yes | yes | `center_crop_pad_forward` | `center_crop_pad_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 129 | `Size` | yes | yes | yes | `size_forward` | `size_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 130 | `Atanh` | yes | yes | yes | `atanh_forward` | `atanh_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 131 | `LSTM` | yes | yes | yes | `lstm_forward` | `lstm_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
| 132 | `BatchNormalization` | yes | yes | yes | `batch_norm_forward` | `batch_norm_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 133 | `Unsqueeze` | yes | yes | yes | `reshape_forward` | `reshape_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 134 | `Gelu` | yes | yes | yes | `gelu_forward` | `gelu_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
| 135 | `Where` | yes | yes | yes | `where_forward` | `where_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 136 | `TopK` | yes | yes | yes | `topk_forward` | `topk_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 137 | `Tril` | yes | yes | yes | `triangular_forward` | `triangular_forward` | no | no | 已 pytest 语义验证 | 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 138 | `HannWindow` | yes | yes | yes | `hann_window_forward` | `hann_window_forward` | no | no | 已 pytest 语义验证 | 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 139 | `RoiAlign` | yes | yes | yes | `roi_align_forward` | `roi_align_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
| 140 | `Triu` | yes | yes | yes | `triangular_forward` | `triangular_forward` | no | no | 已 pytest 语义验证 | 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 141 | `Mish` | yes | yes | yes | `mish_forward` | `mish_forward` | yes | yes | 已数值验证 | 已有独立 pytest 深度语义/混合精度覆盖 |
| 142 | `DIV` | yes | yes | yes | `div_forward` | `div_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 143 | `BitwiseAnd` | yes | yes | yes | `bitwise_and_forward` | `bitwise_and_forward` | no | no | 已 pytest 语义验证 | 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 144 | `Scan` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 145 | `HammingWindow` | yes | yes | yes | `hamming_window_forward` | `hamming_window_forward` | no | no | 已 pytest 语义验证 | 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 146 | `Binarizer` | yes | yes | yes | `binarizer_forward` | `binarizer_forward` | no | no | 已 pytest 语义验证 | 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 147 | `QLinearConv` | yes | yes | yes | `qlinear_conv_forward` | `qlinear_conv_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 148 | `BitwiseOr` | yes | yes | yes | `bitwise_or_forward` | `bitwise_or_forward` | no | no | 已 pytest 语义验证 | 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 149 | `Split` | yes | yes | yes | `slice_forward` | `slice_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 150 | `Concat` | yes | yes | yes | `concat_forward` | `concat_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 151 | `Trilu` | yes | yes | yes | `triangular_forward` | `triangular_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 152 | `BlackmanWindow` | yes | yes | yes | `blackman_window_forward` | `blackman_window_forward` | no | no | 已 pytest 语义验证 | 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 153 | `BitwiseXor` | yes | yes | yes | `bitwise_xor_forward` | `bitwise_xor_forward` | no | no | 已 pytest 语义验证 | 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 154 | `DynamicQuantizeLinear` | yes | yes | yes | `dynamic_quantize_linear_forward` | `dynamic_quantize_linear_forward` | no | no | 已 pytest 语义验证 | 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 155 | `InstanceNormalization` | yes | yes | yes | `instance_norm_forward` | `instance_norm_forward` | no | no | 已 pytest 语义验证 | 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 156 | `CumSum` | yes | yes | yes | `cumsum_forward` | `cumsum_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 157 | `NonMaxSuppression` | yes | yes | yes | `non_max_suppression_forward` | `non_max_suppression_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 158 | `BitwiseNot` | yes | yes | yes | `bitwise_not_forward` | `bitwise_not_forward` | no | no | 已 pytest 语义验证 | 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 159 | `DepthToSpace` | yes | yes | yes | `depth_to_space_forward` | `depth_to_space_forward` | no | no | 已 pytest 语义验证 | 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 160 | `BitShift` | yes | yes | yes | `bit_shift_forward` | `bit_shift_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 161 | `EXP` | yes | yes | yes | `exp_forward` | `exp_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 162 | `CumProd` | yes | yes | yes | `cumprod_forward` | `cumprod_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 163 | `LayerNormalization` | yes | yes | yes | `layer_norm_forward` | `layer_norm_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 164 | `SequenceMap` | yes | yes | yes | none | Python orchestration | no | no | 已 pytest 语义验证 | Python 调度/元数据类，不要求 C 数值后端; 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 165 | `LOG` | yes | yes | yes | `log_forward` | `log_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 166 | `Clip` | yes | yes | yes | `clip_forward` | `clip_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 167 | `Slice` | yes | yes | yes | `slice_forward` | `slice_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 168 | `SQRT` | yes | yes | yes | `sqrt_forward` | `sqrt_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 169 | `SpaceToDepth` | yes | yes | yes | `space_to_depth_forward` | `space_to_depth_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 170 | `OneHot` | yes | yes | yes | `one_hot_forward` | `one_hot_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 171 | `SIGMOID` | yes | yes | yes | `sigmoid_forward` | `sigmoid_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 172 | `TANH` | yes | yes | yes | `tanh_forward` | `tanh_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 173 | `Einsum` | yes | yes | yes | `einsum_forward` | `einsum_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 174 | `AveragePool` | yes | yes | yes | `average_pool_forward` | `average_pool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 175 | `ReverseSequence` | yes | yes | yes | `reverse_sequence_forward` | `reverse_sequence_forward` | no | no | 已 pytest 语义验证 | 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 176 | `Pow` | yes | yes | yes | `pow_forward` | `pow_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 177 | `Hardmax` | yes | yes | yes | `hardmax_forward` | `hardmax_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 178 | `Compress` | yes | yes | yes | `compress_forward` | `compress_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 179 | `Max` | yes | yes | yes | `max_forward` | `max_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 180 | `LogSoftmax` | yes | yes | yes | `log_softmax_forward` | `log_softmax_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 181 | `LpPool` | yes | yes | yes | `lp_pool_forward` | `lp_pool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 182 | `Min` | yes | yes | yes | `min_forward` | `min_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 183 | `LpNormalization` | yes | yes | yes | `lp_normalization_forward` | `lp_normalization_forward` | no | no | 已 pytest 语义验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 184 | `Cast` | yes | yes | yes | `cast_forward` | `cast_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 185 | `ScatterElements` | yes | yes | yes | `scatter_elements_forward` | `scatter_elements_forward` | no | no | 已 pytest 语义验证 | 已有 ONNX reference pytest 语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 186 | `Neg` | yes | yes | yes | `neg_forward` | `neg_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 187 | `CastLike` | yes | yes | yes | `cast_forward` | `cast_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 188 | `GroupNormalization` | yes | yes | yes | `group_norm_forward` | `group_norm_forward` | no | no | 已 pytest 语义验证 | 已有独立 pytest 深度语义/混合精度覆盖; 缺少 CUDA/数值验证覆盖 |
| 189 | `Reciprocal` | yes | yes | yes | `reciprocal_forward` | `reciprocal_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 190 | `GlobalAveragePool` | yes | yes | yes | `global_average_pool_forward` | `global_average_pool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 191 | `Ceil` | yes | yes | yes | `ceil_forward` | `ceil_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 192 | `BitCast` | yes | yes | yes | `bitcast_forward` | `bitcast_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 193 | `Floor` | yes | yes | yes | `floor_forward` | `floor_forward` | yes | yes | 已数值验证 | 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 194 | `GlobalMaxPool` | yes | yes | yes | `global_max_pool_forward` | `global_max_pool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 195 | `Sum` | yes | yes | yes | `sum_forward` | `sum_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有 ONNX reference pytest 语义/混合精度覆盖 |
| 196 | `GlobalLpPool` | yes | yes | yes | `global_lp_pool_forward` | `global_lp_pool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback; 已有独立 pytest 深度语义/混合精度覆盖 |
