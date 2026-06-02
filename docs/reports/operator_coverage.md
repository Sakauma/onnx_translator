# 算子实现情况评估

> 自动生成时间：2026-06-02 13:54:10
> 生成命令：`python tools/audit_ops.py --output docs/reports/operator_coverage.md`

## 评估口径

- `ONNXImport`：`nn/ONNXImport.py` 中存在显式映射，可从 ONNX node 构造对应算子类。
- `forward`：`nn/Operators.py` 中存在运行时前向实现，继承自 `ReduceBase`/`ArgBase` 的实现也计入。
- `forward_`：存在图构建/形状推断实现，继承实现也计入。
- `C backend funcs`：Python 算子类中引用的 `<op>_forward` C 函数均能在 `tensor_ops/*.c` 中找到，且 `tensor_ops.h` 与 `.c` 声明/实现集合一致。
- `C runtime path`：`forward()` 运行路径实际引用 `<op>_forward` C 函数；仅在 `__init__`、形状推断或未调用 helper 中出现不计入。
- `CUDA verifier`：`cuda/verify_<op>.cu` 存在，仅说明有参考验证程序源码。
- `active numerical plan`：`numerical_correctness.py` 兼容入口对应的 `tools/numerical/cli.py` 默认计划中包含该算子，代表会被默认数值验证门禁执行。
- `ONNX opset 17 官方覆盖`：通过本地 `onnx.defs` 读取默认 domain 中 `since_version <= 17` 的最新 schema，并与 `ONNXImport` 的显式映射做名称级对比。

## 总览

- Python 算子类：186 个。
- ONNXImport 显式支持：187 个 ONNX op 名称；`Upsample` 作为 `Resize` 别名处理，未单独计入算子类。
- C 后端声明：160 个 `<op>_forward`；C 实现可检出：160 个。
- forward 实际接入 C 后端：167 个算子类。
- 合理保留 Python 调度/元数据运行时：19 个算子类。
- 普通数值/张量算子 Python-only 运行时：0 个算子类；其中当前暂缓后端化：0 个，除暂缓项外待后端化：0 个。
- CUDA verifier：68 个。
- active numerical plan 覆盖：68 个唯一算子名称。
- 暂缓深度语义/数值验证：7 个；`RNN`, `DFT`, `GRU`, `STFT`, `MaxRoiPool`, `LSTM`, `RoiAlign`。
- ONNX opset 17 官方算子：178 个；ONNXImport 名称级覆盖：178 个。

### 状态计数

| 状态 | 数量 |
| --- | ---: |
| 已实现未数值验证 | 111 |
| 已数值验证 | 68 |
| 暂缓深度验证 | 7 |

### 关键结论

- `tensor_ops.h` 与 `tensor_ops.c` 中的 C forward 函数集合一致，没有发现声明缺实现或实现缺声明。
- 所有 186 个 Python 算子类均可被 `ONNXImport` 显式映射。
- 未发现缺少 `forward` / `forward_` / C 函数映射的显式部分实现算子。
- 未发现“有 C 函数但 forward 未调用”的算子。
- 当前没有记录暂缓后端化算子。
- 当前暂缓深度语义/数值验证的剩余算子：`RNN`, `DFT`, `GRU`, `STFT`, `MaxRoiPool`, `LSTM`, `RoiAlign`。
- 除暂缓项外，未发现仍需立即后端化的 Python-only 普通数值/张量算子。
- 默认数值门禁当前覆盖 68 个唯一算子；尚有 118 个已实现算子未进入 active numerical plan。

## ONNX opset 17 官方覆盖

- 官方默认 domain 算子：178 个。
- `ONNXImport` 已覆盖官方名称：178 个。
- 官方名称级缺口：0 个。

### 官方缺口

无。

### 仓库额外/非默认 domain/实验性名称

`Binarizer`, `BitwiseAnd`, `BitwiseNot`, `BitwiseOr`, `BitwiseXor`, `Gelu`, `GroupNormalization`, `Mish`, `Tril`, `Triu`

## 明细表

| # | 算子类 | ONNXImport | forward | forward_ | C backend funcs | C runtime path | CUDA verifier | active numerical plan | 状态 | 备注 |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `Conv` | yes | yes | yes | `conv2d_forward` | `conv2d_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 2 | `Elu` | yes | yes | yes | `elu_forward` | `elu_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 3 | `RELU` | yes | yes | yes | `relu_forward` | `relu_forward` | yes | yes | 已数值验证 | - |
| 4 | `Equal` | yes | yes | yes | `equal_forward` | `equal_forward` | yes | yes | 已数值验证 | - |
| 5 | `Gather` | yes | yes | yes | `gather_forward` | `gather_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 6 | `Multinomial` | yes | yes | yes | `multinomial_forward` | `multinomial_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 7 | `Gemm` | yes | yes | yes | `gemm_forward` | `gemm_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 8 | `Softmax` | yes | yes | yes | `softmax_forward` | `softmax_forward` | yes | yes | 已数值验证 | - |
| 9 | `GridSample` | yes | yes | yes | `grid_sample_forward` | `grid_sample_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 10 | `QuantizeLinear` | yes | yes | yes | `quantize_linear_forward` | `quantize_linear_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 11 | `EyeLike` | yes | yes | yes | `eye_like_forward` | `eye_like_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 12 | `SequenceEmpty` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 13 | `Shape` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 14 | `Expand` | yes | yes | yes | `expand_forward` | `expand_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 15 | `Flatten` | yes | yes | yes | `flatten_forward` | `flatten_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 16 | `MelWeightMatrix` | yes | yes | yes | `mel_weight_matrix_forward` | `mel_weight_matrix_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 17 | `StringNormalizer` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 18 | `Greater` | yes | yes | yes | `greater_forward` | `greater_forward` | yes | yes | 已数值验证 | - |
| 19 | `SequenceConstruct` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 20 | `Selu` | yes | yes | yes | `selu_forward` | `selu_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 21 | `RNN` | yes | yes | yes | `rnn_forward` | `rnn_forward` | no | no | 暂缓深度验证 | 含 Python 调度或 fallback; 按当前整理阶段暂缓深度语义/数值验证，作为剩余风险跟踪; 缺少 CUDA/数值验证覆盖 |
| 22 | `Constant` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 23 | `Less` | yes | yes | yes | `less_forward` | `less_forward` | yes | yes | 已数值验证 | - |
| 24 | `SequenceAt` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 25 | `RandomUniformLike` | yes | yes | yes | `random_uniform_like_forward` | `random_uniform_like_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 26 | `DequantizeLinear` | yes | yes | yes | `dequantize_linear_forward` | `dequantize_linear_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 27 | `LRN` | yes | yes | yes | `lrn_forward` | `lrn_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 28 | `LeakyRelu` | yes | yes | yes | `leaky_relu_forward` | `leaky_relu_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 29 | `GreaterOrEqual` | yes | yes | yes | `greater_or_equal_forward` | `greater_or_equal_forward` | yes | yes | 已数值验证 | - |
| 30 | `SequenceInsert` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 31 | `ScatterND` | yes | yes | yes | `scatter_nd_forward` | `scatter_nd_forward` | yes | yes | 已数值验证 | - |
| 32 | `NegativeLogLikelihoodLoss` | yes | yes | yes | `negative_log_likelihood_loss_forward` | `negative_log_likelihood_loss_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 33 | `Reshape` | yes | yes | yes | `reshape_forward` | `reshape_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 34 | `COS` | yes | yes | yes | `cos_forward` | `cos_forward` | yes | yes | 已数值验证 | - |
| 35 | `ConstantOfShape` | yes | yes | yes | `constant_of_shape_forward` | `constant_of_shape_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 36 | `PRelu` | yes | yes | yes | `prelu_forward` | `prelu_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 37 | `LessOrEqual` | yes | yes | yes | `less_or_equal_forward` | `less_or_equal_forward` | yes | yes | 已数值验证 | - |
| 38 | `SequenceErase` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 39 | `RandomUniform` | yes | yes | yes | `random_uniform_like_forward` | `random_uniform_like_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 40 | `ThresholdedRelu` | yes | yes | yes | `thresholded_relu_forward` | `thresholded_relu_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 41 | `Resize` | yes | yes | yes | `resize_forward` | `resize_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 42 | `Not` | yes | yes | yes | `not_forward` | `not_forward` | yes | yes | 已数值验证 | - |
| 43 | `GatherND` | yes | yes | yes | `gather_nd_forward` | `gather_nd_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 44 | `SequenceLength` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 45 | `TfIdfVectorizer` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 46 | `DFT` | yes | yes | yes | `dft_forward` | `dft_forward` | no | no | 暂缓深度验证 | 含 Python 调度或 fallback; 按当前整理阶段暂缓深度语义/数值验证，作为剩余风险跟踪; 缺少 CUDA/数值验证覆盖 |
| 47 | `MeanVarianceNormalization` | yes | yes | yes | `mean_variance_normalization_forward` | `mean_variance_normalization_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 48 | `HardSigmoid` | yes | yes | yes | `hard_sigmoid_forward` | `hard_sigmoid_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 49 | `And` | yes | yes | yes | `and_forward` | `and_forward` | yes | yes | 已数值验证 | - |
| 50 | `RandomNormal` | yes | yes | yes | `random_normal_forward` | `random_normal_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 51 | `Det` | yes | yes | yes | `det_forward` | `det_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 52 | `MaxPool` | yes | yes | yes | `max_pool_forward` | `max_pool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 53 | `ConvTranspose` | yes | yes | yes | `conv_transpose2d_forward` | `conv_transpose2d_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 54 | `ConcatFromSequence` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 55 | `Range` | yes | yes | yes | `range_forward` | `range_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 56 | `ABS` | yes | yes | yes | `abs_forward` | `abs_forward` | yes | yes | 已数值验证 | - |
| 57 | `Or` | yes | yes | yes | `or_forward` | `or_forward` | yes | yes | 已数值验证 | - |
| 58 | `Celu` | yes | yes | yes | `celu_forward` | `celu_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 59 | `Xor` | yes | yes | yes | `xor_forward` | `xor_forward` | yes | yes | 已数值验证 | - |
| 60 | `GatherElements` | yes | yes | yes | `gather_elements_forward` | `gather_elements_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 61 | `RandomNormalLike` | yes | yes | yes | `random_normal_forward` | `random_normal_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 62 | `MatMulInteger` | yes | yes | yes | `matmul_integer_forward` | `matmul_integer_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 63 | `ReduceMean` | yes | yes | yes | `reduce_mean_forward` | `reduce_mean_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 64 | `Shrink` | yes | yes | yes | `shrink_forward` | `shrink_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 65 | `SplitToSequence` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 66 | `Tile` | yes | yes | yes | `tile_forward` | `tile_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 67 | `IsNaN` | yes | yes | yes | `isnan_forward` | `isnan_forward` | yes | yes | 已数值验证 | - |
| 68 | `Mean` | yes | yes | yes | `mean_forward` | `mean_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 69 | `ReduceSum` | yes | yes | yes | `reduce_sum_forward` | `reduce_sum_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 70 | `Transpose` | yes | yes | yes | `transpose_forward` | `transpose_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 71 | `ReduceMax` | yes | yes | yes | `reduce_max_forward` | `reduce_max_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 72 | `ADD` | yes | yes | yes | `add_forward` | `add_forward` | yes | yes | 已数值验证 | - |
| 73 | `ReduceMin` | yes | yes | yes | `reduce_min_forward` | `reduce_min_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 74 | `Sin` | yes | yes | yes | `sin_forward` | `sin_forward` | yes | yes | 已数值验证 | - |
| 75 | `ReduceProd` | yes | yes | yes | `reduce_prod_forward` | `reduce_prod_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 76 | `Softplus` | yes | yes | yes | `softplus_forward` | `softplus_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 77 | `Bernoulli` | yes | yes | yes | `bernoulli_forward` | `bernoulli_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 78 | `NonZero` | yes | yes | yes | `nonzero_forward` | `nonzero_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 79 | `Tan` | yes | yes | yes | `tan_forward` | `tan_forward` | yes | yes | 已数值验证 | - |
| 80 | `Softsign` | yes | yes | yes | `softsign_forward` | `softsign_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 81 | `Pad` | yes | yes | yes | `pad_forward` | `pad_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 82 | `Atan` | yes | yes | yes | `atan_forward` | `atan_forward` | yes | yes | 已数值验证 | - |
| 83 | `GRU` | yes | yes | yes | `gru_forward` | `gru_forward` | no | no | 暂缓深度验证 | 含 Python 调度或 fallback; 按当前整理阶段暂缓深度语义/数值验证，作为剩余风险跟踪; 缺少 CUDA/数值验证覆盖 |
| 84 | `IsInf` | yes | yes | yes | `isinf_forward` | `isinf_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 85 | `SoftmaxCrossEntropyLoss` | yes | yes | yes | `softmax_cross_entropy_loss_forward` | `softmax_cross_entropy_loss_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 86 | `MaxUnpool` | yes | yes | yes | `max_unpool_forward` | `max_unpool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 87 | `HardSwish` | yes | yes | yes | `hard_swish_forward` | `hard_swish_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 88 | `Optional` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 89 | `TopK` | yes | yes | yes | `topk_forward` | `topk_forward` | yes | yes | 已数值验证 | - |
| 90 | `Sign` | yes | yes | yes | `sign_forward` | `sign_forward` | yes | yes | 已数值验证 | - |
| 91 | `QLinearMatMul` | yes | yes | yes | `qlinear_matmul_forward` | `qlinear_matmul_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 92 | `Acos` | yes | yes | yes | `acos_forward` | `acos_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 93 | `Dropout` | yes | yes | yes | `dropout_forward` | `dropout_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 94 | `STFT` | yes | yes | yes | `stft_forward` | `stft_forward` | no | no | 暂缓深度验证 | 含 Python 调度或 fallback; 按当前整理阶段暂缓深度语义/数值验证，作为剩余风险跟踪; 缺少 CUDA/数值验证覆盖 |
| 95 | `Asin` | yes | yes | yes | `asin_forward` | `asin_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 96 | `Identity` | yes | yes | yes | `identity_forward` | `identity_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 97 | `OptionalGetElement` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 98 | `Round` | yes | yes | yes | `round_forward` | `round_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 99 | `Cosh` | yes | yes | yes | `cosh_forward` | `cosh_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 100 | `SUB` | yes | yes | yes | `sub_forward` | `sub_forward` | yes | yes | 已数值验证 | - |
| 101 | `Erf` | yes | yes | yes | `erf_forward` | `erf_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 102 | `Squeeze` | yes | yes | yes | `reshape_forward` | `reshape_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 103 | `OptionalHasElement` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 104 | `Size` | yes | yes | yes | `size_forward` | `size_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 105 | `Sinh` | yes | yes | yes | `sinh_forward` | `sinh_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 106 | `BatchNormalization` | yes | yes | yes | `batch_norm_forward` | `batch_norm_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 107 | `ArgMax` | yes | yes | yes | `argmax_forward` | `argmax_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 108 | `Mod` | yes | yes | yes | `mod_forward` | `mod_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 109 | `ArgMin` | yes | yes | yes | `argmin_forward` | `argmin_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 110 | `Asinh` | yes | yes | yes | `asinh_forward` | `asinh_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 111 | `If` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 112 | `ReduceL1` | yes | yes | yes | `reduce_l1_forward` | `reduce_l1_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 113 | `CumSum` | yes | yes | yes | `cumsum_forward` | `cumsum_forward` | yes | yes | 已数值验证 | - |
| 114 | `ReduceL2` | yes | yes | yes | `reduce_l2_forward` | `reduce_l2_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 115 | `Acosh` | yes | yes | yes | `acosh_forward` | `acosh_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 116 | `ReduceLogSum` | yes | yes | yes | `reduce_log_sum_forward` | `reduce_log_sum_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 117 | `Tril` | yes | yes | yes | `triangular_forward` | `triangular_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 118 | `ReduceLogSumExp` | yes | yes | yes | `reduce_log_sum_exp_forward` | `reduce_log_sum_exp_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 119 | `ConvInteger` | yes | yes | yes | `conv_integer_forward` | `conv_integer_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 120 | `MaxRoiPool` | yes | yes | yes | `max_roi_pool_forward` | `max_roi_pool_forward` | no | no | 暂缓深度验证 | 含 Python 调度或 fallback; 按当前整理阶段暂缓深度语义/数值验证，作为剩余风险跟踪; 缺少 CUDA/数值验证覆盖 |
| 121 | `ReduceSumSquare` | yes | yes | yes | `reduce_sum_square_forward` | `reduce_sum_square_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 122 | `Atanh` | yes | yes | yes | `atanh_forward` | `atanh_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 123 | `MatMul` | yes | yes | yes | `matmul_forward` | `matmul_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 124 | `Loop` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 125 | `Gelu` | yes | yes | yes | `gelu_forward` | `gelu_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 126 | `OneHot` | yes | yes | yes | `one_hot_forward` | `one_hot_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 127 | `Unique` | yes | yes | yes | `unique_forward` | `unique_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 128 | `Split` | yes | yes | yes | `slice_forward` | `slice_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 129 | `MUL` | yes | yes | yes | `mul_forward` | `mul_forward` | yes | yes | 已数值验证 | - |
| 130 | `LSTM` | yes | yes | yes | `lstm_forward` | `lstm_forward` | no | no | 暂缓深度验证 | 含 Python 调度或 fallback; 按当前整理阶段暂缓深度语义/数值验证，作为剩余风险跟踪; 缺少 CUDA/数值验证覆盖 |
| 131 | `Triu` | yes | yes | yes | `triangular_forward` | `triangular_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 132 | `Mish` | yes | yes | yes | `mish_forward` | `mish_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 133 | `Binarizer` | yes | yes | yes | `binarizer_forward` | `binarizer_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 134 | `Unsqueeze` | yes | yes | yes | `reshape_forward` | `reshape_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 135 | `Where` | yes | yes | yes | `where_forward` | `where_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 136 | `HannWindow` | yes | yes | yes | `hann_window_forward` | `hann_window_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 137 | `Trilu` | yes | yes | yes | `triangular_forward` | `triangular_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 138 | `InstanceNormalization` | yes | yes | yes | `instance_norm_forward` | `instance_norm_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 139 | `DynamicQuantizeLinear` | yes | yes | yes | `dynamic_quantize_linear_forward` | `dynamic_quantize_linear_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 140 | `HammingWindow` | yes | yes | yes | `hamming_window_forward` | `hamming_window_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 141 | `RoiAlign` | yes | yes | yes | `roi_align_forward` | `roi_align_forward` | no | no | 暂缓深度验证 | 含 Python 调度或 fallback; 按当前整理阶段暂缓深度语义/数值验证，作为剩余风险跟踪; 缺少 CUDA/数值验证覆盖 |
| 142 | `DepthToSpace` | yes | yes | yes | `depth_to_space_forward` | `depth_to_space_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 143 | `DIV` | yes | yes | yes | `div_forward` | `div_forward` | yes | yes | 已数值验证 | - |
| 144 | `BitwiseAnd` | yes | yes | yes | `bitwise_and_forward` | `bitwise_and_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 145 | `BlackmanWindow` | yes | yes | yes | `blackman_window_forward` | `blackman_window_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 146 | `Scan` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 147 | `LayerNormalization` | yes | yes | yes | `layer_norm_forward` | `layer_norm_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 148 | `QLinearConv` | yes | yes | yes | `qlinear_conv_forward` | `qlinear_conv_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 149 | `BitwiseOr` | yes | yes | yes | `bitwise_or_forward` | `bitwise_or_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 150 | `Concat` | yes | yes | yes | `concat_forward` | `concat_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 151 | `BitwiseXor` | yes | yes | yes | `bitwise_xor_forward` | `bitwise_xor_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 152 | `NonMaxSuppression` | yes | yes | yes | `non_max_suppression_forward` | `non_max_suppression_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 153 | `SpaceToDepth` | yes | yes | yes | `space_to_depth_forward` | `space_to_depth_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 154 | `BitwiseNot` | yes | yes | yes | `bitwise_not_forward` | `bitwise_not_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 155 | `BitShift` | yes | yes | yes | `bit_shift_forward` | `bit_shift_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 156 | `EXP` | yes | yes | yes | `exp_forward` | `exp_forward` | yes | yes | 已数值验证 | - |
| 157 | `SequenceMap` | yes | yes | yes | none | Python orchestration | no | no | 已实现未数值验证 | Python 调度/元数据类，不要求 C 数值后端; 缺少 CUDA/数值验证覆盖 |
| 158 | `Clip` | yes | yes | yes | `clip_forward` | `clip_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 159 | `LOG` | yes | yes | yes | `log_forward` | `log_forward` | yes | yes | 已数值验证 | - |
| 160 | `ReverseSequence` | yes | yes | yes | `reverse_sequence_forward` | `reverse_sequence_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 161 | `Slice` | yes | yes | yes | `slice_forward` | `slice_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 162 | `Hardmax` | yes | yes | yes | `hardmax_forward` | `hardmax_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 163 | `SQRT` | yes | yes | yes | `sqrt_forward` | `sqrt_forward` | yes | yes | 已数值验证 | - |
| 164 | `SIGMOID` | yes | yes | yes | `sigmoid_forward` | `sigmoid_forward` | yes | yes | 已数值验证 | - |
| 165 | `Compress` | yes | yes | yes | `compress_forward` | `compress_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 166 | `LogSoftmax` | yes | yes | yes | `log_softmax_forward` | `log_softmax_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 167 | `TANH` | yes | yes | yes | `tanh_forward` | `tanh_forward` | yes | yes | 已数值验证 | - |
| 168 | `Einsum` | yes | yes | yes | `einsum_forward` | `einsum_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 169 | `AveragePool` | yes | yes | yes | `average_pool_forward` | `average_pool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 170 | `LpNormalization` | yes | yes | yes | `lp_normalization_forward` | `lp_normalization_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 171 | `Pow` | yes | yes | yes | `pow_forward` | `pow_forward` | yes | yes | 已数值验证 | - |
| 172 | `Max` | yes | yes | yes | `max_forward` | `max_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 173 | `ScatterElements` | yes | yes | yes | `scatter_elements_forward` | `scatter_elements_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 174 | `GroupNormalization` | yes | yes | yes | `group_norm_forward` | `group_norm_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 175 | `LpPool` | yes | yes | yes | `lp_pool_forward` | `lp_pool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 176 | `Min` | yes | yes | yes | `min_forward` | `min_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 177 | `Cast` | yes | yes | yes | `cast_forward` | `cast_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 178 | `Neg` | yes | yes | yes | `neg_forward` | `neg_forward` | yes | yes | 已数值验证 | - |
| 179 | `CastLike` | yes | yes | yes | `cast_forward` | `cast_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 180 | `Reciprocal` | yes | yes | yes | `reciprocal_forward` | `reciprocal_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 181 | `GlobalAveragePool` | yes | yes | yes | `global_average_pool_forward` | `global_average_pool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 182 | `Ceil` | yes | yes | yes | `ceil_forward` | `ceil_forward` | no | no | 已实现未数值验证 | 缺少 CUDA/数值验证覆盖 |
| 183 | `Sum` | yes | yes | yes | `sum_forward` | `sum_forward` | no | no | 已实现未数值验证 | 含 Python 调度或 fallback; 缺少 CUDA/数值验证覆盖 |
| 184 | `Floor` | yes | yes | yes | `floor_forward` | `floor_forward` | yes | yes | 已数值验证 | - |
| 185 | `GlobalMaxPool` | yes | yes | yes | `global_max_pool_forward` | `global_max_pool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
| 186 | `GlobalLpPool` | yes | yes | yes | `global_lp_pool_forward` | `global_lp_pool_forward` | yes | yes | 已数值验证 | 含 Python 调度或 fallback |
