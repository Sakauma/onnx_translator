# ONNX Official Semantic Matrix

- Generated at: `2026-06-27T10:00:46.948357+00:00`
- Official latest default-domain operators: `200`
- Verified operators: `200`
- Runtime-only or weak evidence: `0`
- Deprecated official operators: `2`
- Deprecated aliases covered by canonical classes: `2`

| Op | Since | Deprecated | Import | Classes | Alias | Evidence | Status |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `Abs` | 13 | False | True | `ABS` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Acos` | 22 | False | True | `Acos` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Acosh` | 22 | False | True | `Acosh` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Add` | 14 | False | True | `ADD` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `AffineGrid` | 20 | False | True | `AffineGrid` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `And` | 7 | False | True | `And` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `ArgMax` | 13 | False | True | `ArgMax` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `ArgMin` | 13 | False | True | `ArgMin` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Asin` | 22 | False | True | `Asin` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Asinh` | 22 | False | True | `Asinh` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Atan` | 22 | False | True | `Atan` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Atanh` | 22 | False | True | `Atanh` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Attention` | 24 | False | True | `Attention` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `AveragePool` | 22 | False | True | `AveragePool` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `BatchNormalization` | 15 | False | True | `BatchNormalization` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Bernoulli` | 22 | False | True | `Bernoulli` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `BitCast` | 26 | False | True | `BitCast` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `BitShift` | 11 | False | True | `BitShift` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `BitwiseAnd` | 18 | False | True | `BitwiseAnd` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `BitwiseNot` | 18 | False | True | `BitwiseNot` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `BitwiseOr` | 18 | False | True | `BitwiseOr` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `BitwiseXor` | 18 | False | True | `BitwiseXor` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `BlackmanWindow` | 17 | False | True | `BlackmanWindow` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Cast` | 25 | False | True | `Cast` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `CastLike` | 25 | False | True | `CastLike` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Ceil` | 13 | False | True | `Ceil` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Celu` | 12 | False | True | `Celu` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `CenterCropPad` | 18 | False | True | `CenterCropPad` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Clip` | 13 | False | True | `Clip` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Col2Im` | 18 | False | True | `Col2Im` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Compress` | 11 | False | True | `Compress` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Concat` | 13 | False | True | `Concat` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `ConcatFromSequence` | 11 | False | True | `ConcatFromSequence` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `Constant` | 25 | False | True | `Constant` | - | `onnx_reference_pytest`, `python_orchestration` | `verified` |
| `ConstantOfShape` | 25 | False | True | `ConstantOfShape` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Conv` | 22 | False | True | `Conv` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `ConvInteger` | 10 | False | True | `ConvInteger` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `ConvTranspose` | 22 | False | True | `ConvTranspose` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Cos` | 22 | False | True | `COS` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Cosh` | 22 | False | True | `Cosh` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `CumProd` | 26 | False | True | `CumProd` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `CumSum` | 14 | False | True | `CumSum` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `DFT` | 20 | False | True | `DFT` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `DeformConv` | 22 | False | True | `DeformConv` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `DepthToSpace` | 13 | False | True | `DepthToSpace` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `DequantizeLinear` | 25 | False | True | `DequantizeLinear` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Det` | 22 | False | True | `Det` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Div` | 14 | False | True | `DIV` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Dropout` | 22 | False | True | `Dropout` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `DynamicQuantizeLinear` | 11 | False | True | `DynamicQuantizeLinear` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Einsum` | 12 | False | True | `Einsum` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Elu` | 22 | False | True | `Elu` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Equal` | 19 | False | True | `Equal` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Erf` | 13 | False | True | `Erf` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Exp` | 13 | False | True | `EXP` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Expand` | 13 | False | True | `Expand` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `EyeLike` | 22 | False | True | `EyeLike` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Flatten` | 25 | False | True | `Flatten` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Floor` | 13 | False | True | `Floor` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `GRU` | 22 | False | True | `GRU` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `Gather` | 13 | False | True | `Gather` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `GatherElements` | 13 | False | True | `GatherElements` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `GatherND` | 13 | False | True | `GatherND` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Gelu` | 20 | False | True | `Gelu` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `Gemm` | 13 | False | True | `Gemm` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `GlobalAveragePool` | 22 | False | True | `GlobalAveragePool` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `GlobalLpPool` | 22 | False | True | `GlobalLpPool` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `GlobalMaxPool` | 22 | False | True | `GlobalMaxPool` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Greater` | 13 | False | True | `Greater` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `GreaterOrEqual` | 16 | False | True | `GreaterOrEqual` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `GridSample` | 22 | False | True | `GridSample` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `GroupNormalization` | 21 | False | True | `GroupNormalization` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `HammingWindow` | 17 | False | True | `HammingWindow` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `HannWindow` | 17 | False | True | `HannWindow` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `HardSigmoid` | 22 | False | True | `HardSigmoid` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `HardSwish` | 22 | False | True | `HardSwish` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Hardmax` | 13 | False | True | `Hardmax` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Identity` | 25 | False | True | `Identity` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `If` | 25 | False | True | `If` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `ImageDecoder` | 20 | False | True | `ImageDecoder` | - | `onnx_reference_pytest`, `python_orchestration` | `verified` |
| `InstanceNormalization` | 22 | False | True | `InstanceNormalization` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `IsInf` | 20 | False | True | `IsInf` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `IsNaN` | 20 | False | True | `IsNaN` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `LRN` | 13 | False | True | `LRN` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `LSTM` | 22 | False | True | `LSTM` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `LayerNormalization` | 17 | False | True | `LayerNormalization` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `LeakyRelu` | 16 | False | True | `LeakyRelu` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Less` | 13 | False | True | `Less` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `LessOrEqual` | 16 | False | True | `LessOrEqual` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Log` | 13 | False | True | `LOG` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `LogSoftmax` | 13 | False | True | `LogSoftmax` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Loop` | 25 | False | True | `Loop` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `LpNormalization` | 22 | False | True | `LpNormalization` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `LpPool` | 22 | False | True | `LpPool` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `MatMul` | 13 | False | True | `MatMul` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `MatMulInteger` | 10 | False | True | `MatMulInteger` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Max` | 13 | False | True | `Max` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `MaxPool` | 22 | False | True | `MaxPool` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `MaxRoiPool` | 22 | False | True | `MaxRoiPool` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `MaxUnpool` | 22 | False | True | `MaxUnpool` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Mean` | 13 | False | True | `Mean` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `MeanVarianceNormalization` | 13 | False | True | `MeanVarianceNormalization` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `MelWeightMatrix` | 17 | False | True | `MelWeightMatrix` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Min` | 13 | False | True | `Min` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Mish` | 22 | False | True | `Mish` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `Mod` | 13 | False | True | `Mod` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Mul` | 14 | False | True | `MUL` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Multinomial` | 22 | False | True | `Multinomial` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `Neg` | 13 | False | True | `Neg` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `NegativeLogLikelihoodLoss` | 22 | False | True | `NegativeLogLikelihoodLoss` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `NonMaxSuppression` | 11 | False | True | `NonMaxSuppression` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `NonZero` | 13 | False | True | `NonZero` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Not` | 1 | False | True | `Not` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `OneHot` | 11 | False | True | `OneHot` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Optional` | 15 | False | True | `Optional` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `OptionalGetElement` | 18 | False | True | `OptionalGetElement` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `OptionalHasElement` | 18 | False | True | `OptionalHasElement` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `Or` | 7 | False | True | `Or` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `PRelu` | 16 | False | True | `PRelu` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Pad` | 25 | False | True | `Pad` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Pow` | 15 | False | True | `Pow` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `QLinearConv` | 10 | False | True | `QLinearConv` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `QLinearMatMul` | 21 | False | True | `QLinearMatMul` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `QuantizeLinear` | 25 | False | True | `QuantizeLinear` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `RMSNormalization` | 23 | False | True | `RMSNormalization` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `RNN` | 22 | False | True | `RNN` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `RandomNormal` | 22 | False | True | `RandomNormal` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `RandomNormalLike` | 22 | False | True | `RandomNormalLike` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `RandomUniform` | 22 | False | True | `RandomUniform` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `RandomUniformLike` | 22 | False | True | `RandomUniformLike` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `Range` | 11 | False | True | `Range` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Reciprocal` | 13 | False | True | `Reciprocal` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `ReduceL1` | 18 | False | True | `ReduceL1` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `ReduceL2` | 18 | False | True | `ReduceL2` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `ReduceLogSum` | 18 | False | True | `ReduceLogSum` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `ReduceLogSumExp` | 18 | False | True | `ReduceLogSumExp` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `ReduceMax` | 20 | False | True | `ReduceMax` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `ReduceMean` | 18 | False | True | `ReduceMean` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `ReduceMin` | 20 | False | True | `ReduceMin` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `ReduceProd` | 18 | False | True | `ReduceProd` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `ReduceSum` | 13 | False | True | `ReduceSum` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `ReduceSumSquare` | 18 | False | True | `ReduceSumSquare` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `RegexFullMatch` | 20 | False | True | `RegexFullMatch` | - | `onnx_reference_pytest`, `python_orchestration` | `verified` |
| `Relu` | 14 | False | True | `RELU` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Reshape` | 25 | False | True | `Reshape` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Resize` | 19 | False | True | `Resize` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `ReverseSequence` | 10 | False | True | `ReverseSequence` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `RoiAlign` | 22 | False | True | `RoiAlign` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `RotaryEmbedding` | 23 | False | True | `RotaryEmbedding` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Round` | 22 | False | True | `Round` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `STFT` | 17 | False | True | `STFT` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `Scan` | 25 | False | True | `Scan` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `Scatter` | 11 | True | True | `ScatterElements` | deprecated_alias -> `ScatterElements` | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `ScatterElements` | 18 | False | True | `ScatterElements` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `ScatterND` | 18 | False | True | `ScatterND` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Selu` | 22 | False | True | `Selu` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `SequenceAt` | 11 | False | True | `SequenceAt` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `SequenceConstruct` | 11 | False | True | `SequenceConstruct` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `SequenceEmpty` | 11 | False | True | `SequenceEmpty` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `SequenceErase` | 11 | False | True | `SequenceErase` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `SequenceInsert` | 11 | False | True | `SequenceInsert` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `SequenceLength` | 11 | False | True | `SequenceLength` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `SequenceMap` | 17 | False | True | `SequenceMap` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `Shape` | 25 | False | True | `Shape` | - | `onnx_reference_pytest`, `python_orchestration` | `verified` |
| `Shrink` | 9 | False | True | `Shrink` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Sigmoid` | 13 | False | True | `SIGMOID` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Sign` | 13 | False | True | `Sign` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Sin` | 22 | False | True | `Sin` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Sinh` | 22 | False | True | `Sinh` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Size` | 25 | False | True | `Size` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Slice` | 13 | False | True | `Slice` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Softmax` | 13 | False | True | `Softmax` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `SoftmaxCrossEntropyLoss` | 13 | False | True | `SoftmaxCrossEntropyLoss` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Softplus` | 22 | False | True | `Softplus` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Softsign` | 22 | False | True | `Softsign` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `SpaceToDepth` | 13 | False | True | `SpaceToDepth` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Split` | 18 | False | True | `Split` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `SplitToSequence` | 24 | False | True | `SplitToSequence` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `Sqrt` | 13 | False | True | `SQRT` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Squeeze` | 25 | False | True | `Squeeze` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `StringConcat` | 20 | False | True | `StringConcat` | - | `onnx_reference_pytest`, `python_orchestration` | `verified` |
| `StringNormalizer` | 10 | False | True | `StringNormalizer` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `StringSplit` | 20 | False | True | `StringSplit` | - | `onnx_reference_pytest`, `python_orchestration` | `verified` |
| `Sub` | 14 | False | True | `SUB` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Sum` | 13 | False | True | `Sum` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Swish` | 24 | False | True | `Swish` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Tan` | 22 | False | True | `Tan` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Tanh` | 13 | False | True | `TANH` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `TensorScatter` | 24 | False | True | `TensorScatter` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `TfIdfVectorizer` | 9 | False | True | `TfIdfVectorizer` | - | `deep_semantic_pytest`, `python_orchestration` | `verified` |
| `ThresholdedRelu` | 22 | False | True | `ThresholdedRelu` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Tile` | 13 | False | True | `Tile` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `TopK` | 24 | False | True | `TopK` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Transpose` | 25 | False | True | `Transpose` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Trilu` | 14 | False | True | `Trilu` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Unique` | 11 | False | True | `Unique` | - | `c_runtime`, `cuda_verifier`, `deep_semantic_pytest`, `numerical_plan` | `verified` |
| `Unsqueeze` | 25 | False | True | `Unsqueeze` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Upsample` | 10 | True | True | `Resize` | deprecated_alias -> `Resize` | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Where` | 16 | False | True | `Where` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
| `Xor` | 7 | False | True | `Xor` | - | `c_runtime`, `cuda_verifier`, `numerical_plan`, `onnx_reference_pytest` | `verified` |
