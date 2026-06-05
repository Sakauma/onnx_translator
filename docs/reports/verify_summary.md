<!--
/**
  ******************************************************************************
  * @file        verify_summary.md
  * @author      Egor Izmaylov
  * @brief       记录阶段性算子数值验证和图结构验证总结。
  * @details     2026.06.02  V1.0.0  创建
  * @details     2026.06.05  V1.0.1  补充完整 numerical 收尾验证记录
  * @details     2026.06.05  V1.0.2  补充 Transpose CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.3  补充 Flatten 与 Reshape CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.4  补充 Tile 与 Concat CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.5  补充 Expand 与 Pad CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.6  补充 ConstantOfShape 与 EyeLike CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.7  补充 Mean 与 Sum CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.8  补充 Cast 与 CastLike CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.9  补充 Ceil、Reciprocal、Softplus、Softsign 与 HardSigmoid CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.10  补充最终完整 numerical 复跑记录
  * @details     2026.06.05  V1.0.11  补充 Elu、LeakyRelu、PRelu、Selu、Celu 与 ThresholdedRelu CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.12  补充 HardSwish、Shrink、Gelu 与 Mish CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.13  补充 Gelu approximate=tanh 官方属性语义和完整 numerical 收尾记录
  * @details     2026.06.05  V1.0.14  补充 Hardmax 与 LogSoftmax CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.15  补充 Reduce 公式类算子 CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.16  补充一元数学算子 CUDA 数值门禁记录
  * @details     2026.06.05  V1.0.17  补充基础元素级算子 ONNX reference 语义记录
  * @details     2026.06.05  V1.0.18  修复 Reduce keepdims 坐标映射并补充语义记录
  * @details     2026.06.05  V1.0.19  补充索引和排序类算子 ONNX reference 语义记录
  * @details     2026.06.05  V1.0.20  补充当前工作收尾完整 numerical 复跑记录
  * @details     2026.06.05  V1.0.21  补充核心卷积、矩阵、池化和量化算子官方语义记录
  * @details     2026.06.05  V1.0.22  补充复杂属性算子官方语义和 Einsum 混合精度修复记录
  * @details     2026.06.05  V1.0.23  补充当前工作结束前完整 numerical 一轮记录
  * @details     2026.06.05  V1.0.24  补充 AveragePool 覆盖登记和最新 ONNX schema 风险审计记录
  * @details     2026.06.05  V1.0.25  补充 Swish opset 24 官方语义和混合精度验证记录
  * @details     2026.06.05  V1.0.26  补充 CumProd opset 26 和完整 numerical 收尾记录
  * @details     2026.06.05  V1.0.27  补充 RMSNormalization opset 23 和完整 numerical 收尾记录
  * @details     2026.06.05  V1.0.28  补充当前请求的完整 numerical 一轮收尾记录
  * @details     2026.06.05  V1.0.29  补充 BitCast opset 26 和最终 numerical 收尾记录
  * @details     2026.06.05  V1.0.30  补充 CenterCropPad opset 18 语义和混合精度验证记录
  * @details     2026.06.05  V1.0.31  补充 TensorScatter opset 24 语义和混合精度验证记录
  * @details     2026.06.05  V1.0.32  补充 RegexFullMatch、StringConcat 和 StringSplit opset 20 语义验证记录
  * @details     2026.06.05  V1.0.33  补充当前工作结束前完整 numerical 收尾记录
  * @details     2026.06.05  V1.0.34  补充 RotaryEmbedding opset 23 语义和混合精度验证记录
  * @details     2026.06.05  V1.0.35  补充 GridSample CUDA 数值门禁和完整 numerical 复跑记录
  * @details     2026.06.05  V1.0.36  补充 LRN CUDA 数值门禁和当前工作收尾记录
  * @details     2026.06.05  V1.0.37  补充 MeanVarianceNormalization CUDA 数值门禁记录
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
-->

# 算子验证总结说明

## 一、验证背景
本次验证工作对算子出现频率排序开展，  
按出现次数从高到低优先验证高频算子，目标覆盖高频算子为优先。

验证重点为：
- Python 侧算子执行正确性
- CUDA 侧数值计算正确性
- 模型图构建与 shape 推导正确性

在保证结果正确的前提下，未对所有算子做极端 corner case 覆盖。

---

## 二、数值验证（Python + CUDA）

数值验证采用工程内统一验证入口 `python tools/cli.py numerical`：

- Python 侧通过 `nn/Operators.py` 调用 `tensor_ops.so` 作为参考实现
- CUDA 侧通过 `cache/verify_*` 可执行文件作为 ground truth
- 对 Python 与 CUDA 结果逐元素数值对比
- iterations 从默认 200 调整为 20（稳定高效）

### 2.1 说明

- **Mod**：验证时采用 Python/ONNX 风格的 mod 语义（结果与除数同号），避免与 C 的 `fmod` 语义差异带来大误差。
- **RandomUniformLike**：由于不同后端的随机数实现未必逐元素一致，验证中采用固定 RNG 参考实现进行逐元素一致性验证。

注：`ReduceProd` 在随机输入下可能出现溢出；脚本对 `NaN/Inf/Overflow` 做逻辑匹配，当出现 “all values were NaN/Inf/Overflow matched” warning 时，仍表示两侧结果在该情形下匹配，验证可稳定通过。

### 2.2 已完成数值验证的算子（Python + CUDA）

#### 基础算子 / 一元算子
- Neg
- Floor
- Sign
- IsNaN

#### 基础算子 / 二元算子
- Add
- Sub
- Mul
- Div
- Mod
- Max
- Min

#### 基础算子 / 多输入广播算子
- Mean
- Sum

#### 类型转换算子
- Cast
- CastLike

#### 激活 / 数学函数
- Relu
- Sigmoid
- Tanh
- Sin
- Cos
- Tan
- Atan
- Exp
- Log
- Sqrt
- Pow
- Ceil
- Reciprocal
- Softplus
- Softsign
- HardSigmoid
- Elu
- LeakyRelu
- PRelu
- Selu
- Celu
- ThresholdedRelu
- HardSwish
- Shrink
- Gelu
- Mish
- Round
- Erf
- Acos
- Asin
- Cosh
- Sinh
- Asinh
- Acosh
- Atanh

#### 线性代数 / 卷积 / 池化 / Softmax
- Conv
- Gemm
- MatMul
- Einsum
- MaxPool
- Softmax
- Hardmax
- LogSoftmax

#### 比较 / 逻辑
- Equal
- Greater
- Less
- GreaterOrEqual
- LessOrEqual
- Not
- And
- Or
- Xor

#### Reduce
- ReduceMean
- ReduceSum
- ReduceMax
- ReduceMin
- ReduceProd
- ReduceL1
- ReduceL2
- ReduceLogSum
- ReduceLogSumExp
- ReduceSumSquare

#### 索引 / Scatter-Gather
- Gather
- GatherElements
- GatherND
- ScatterND
- NonZero
- TopK
- ArgMin
- ArgMax

#### 扫描 / 随机 / 采样
- CumSum
- Resize
- RandomUniformLike

以上算子均通过 Python 与 CUDA 数值一致性验证。

---

## 三、图结构与 Shape 验证

针对算子中以图结构和 shape 推导为主的算子，
采用模型图验证方式进行覆盖：

- 使用 `python tools/cli.py create-model` 生成模型
- 使用 `python tools/cli.py verify-graph` 验证图构建、节点连接及 shape 推导过程
- 验证模型图能够正确生成且无报错

图结构与 shape 验证通过 `python tools/cli.py create-graph-model` 构造覆盖 Cast / Shape / ConstantOfShape / Unsqueeze / Squeeze / Slice / Transpose / Concat / Reshape / Expand / Where / Range 等算子的 ONNX 模型，并使用 `python tools/cli.py verify-graph` 完成图导入、节点连接与 shape 推导验证，生成图结构可视化结果（result/nps_verification/nps_verification.pdf）

---

## 四、2026.06.05 完整 numerical 收尾记录

本轮收尾前已完成一次完整数值门禁：

- 命令：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`
- 结果：全部通过，无失败算子。
- 覆盖：默认 active numerical plan 覆盖 93 个唯一算子名称、295 条默认计划，其中混合精度计划 198 条。
- 覆盖 dtype：包含 float32、float16、bfloat16、float8_e4m3、float8_e5m2，以及 int8、uint8、int64、bool 等量化、索引和比较相关类型。
- 高风险算子：MaxRoiPool、RoiAlign、RNN、GRU、LSTM、DFT、STFT 已进入完整 numerical 默认门禁，并在本轮一轮验证中通过。

本轮同时修正 `QuantizeLinear` 在 float32 scale 下的 nearest-even 舍入细节：当输入和 scale 均不是 float64 时，C 后端使用 float32 中间值与 `rintf`，避免将 float32 scale 提升到 double 后改变 ONNX 参考实现的半点舍入结果。新增回归测试覆盖 `axis=-1` 的 per-axis uint8/int8 量化，并使用 ONNX `ReferenceEvaluator` 对齐官方参考输出。

后续补充 `Transpose` 的 CUDA 参考验证程序，并将 float32、float16、bfloat16、float8_e4m3、float8_e5m2 五条计划接入默认 numerical 门禁。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op transpose --iterations 3 --skip-plots` 与完整一轮 numerical 均已通过。

继续补充 `Flatten` 与 `Reshape` 的 CUDA 参考验证程序，并将二者的 float32、float16、bfloat16、float8_e4m3、float8_e5m2 计划接入默认 numerical 门禁。`Reshape` 的数值计划使用 `[0, -1]` 目标 shape，覆盖 ONNX 中 0 复制输入维度和 -1 自动推断维度的主路径。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op flatten --op reshape --iterations 3 --skip-plots` 与完整一轮 numerical 均已通过。

继续补充 `Tile` 与 `Concat` 的 CUDA 参考验证程序，并将二者的 float32、float16、bfloat16、float8_e4m3、float8_e5m2 计划接入默认 numerical 门禁。`Tile` 的 CUDA 参考按输出坐标对输入维度取模，`Concat` 的 CUDA 参考按 concat axis 段落定位来源输入。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op tile --op concat --iterations 3 --skip-plots` 与完整一轮 numerical 均已通过。

继续补充 `Expand` 与 `Pad` 的 CUDA 参考验证程序，并将二者的 float32、float16、bfloat16、float8_e4m3、float8_e5m2 计划接入默认 numerical 门禁。`Expand` 的 CUDA 参考按广播后的输出坐标映射回输入坐标，`Pad` 当前数值门禁覆盖 ONNX 标准 constant mode。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op expand --op pad --iterations 3 --skip-plots` 与完整一轮 numerical 均已通过。

继续补充 `ConstantOfShape` 与 `EyeLike` 的 CUDA 参考验证程序，并将二者的 float32、float16、bfloat16、float8_e4m3、float8_e5m2 计划接入默认 numerical 门禁。`ConstantOfShape` 的 CUDA 参考根据 shape 参数生成目标张量并填充统一常量，`EyeLike` 的 CUDA 参考覆盖二维输入形状和 `k=1` 对角线偏移。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op constant_of_shape --op eye_like --iterations 3 --skip-plots` 与完整一轮 numerical 均已通过。

继续补充 `Mean` 与 `Sum` 的 CUDA 参考验证程序，并将二者的 float32、float16、bfloat16、float8_e4m3、float8_e5m2 计划接入默认 numerical 门禁。两者的数值计划均使用三输入互相广播的形状组合，覆盖 ONNX variadic elementwise 与 multidirectional broadcast 主路径。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op mean --op sum --iterations 3 --skip-plots` 与完整一轮 numerical 均已通过。

继续补充 `Cast` 与 `CastLike` 的 CUDA 参考验证程序，并将二者的 float32、float16、bfloat16、float8_e4m3、float8_e5m2、int64、bool 计划接入默认 numerical 门禁。`CastLike` 的 runner 路径同步调整为按第二个 target tensor 的 dtype 决定输出类型，避免用构造参数绕开核心语义。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op cast --op cast_like --iterations 3 --skip-plots` 与完整一轮 numerical 均已通过。

继续补充 `Ceil`、`Reciprocal`、`Softplus`、`Softsign` 与 `HardSigmoid` 的 CUDA 参考验证程序，并将五者的 float32、float16、bfloat16、float8_e4m3、float8_e5m2 计划接入默认 numerical 门禁。数值计划使用有限固定样本，避免 `Reciprocal` 零点和 `Softplus` 随机极大值干扰主路径验证；`HardSigmoid` 覆盖 ONNX 默认 `alpha=0.2`、`beta=0.5`。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op ceil --op reciprocal --op softplus --op softsign --op hard_sigmoid --iterations 3 --skip-plots` 与完整一轮 numerical 均已通过。

最终按收尾要求再次执行完整 numerical：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`。本次复跑覆盖 93 个默认数值验证算子、295 条默认计划和 198 条混合精度计划，所有计划均通过，无失败算子。随后重新执行 `tools/audit_ops.py`，并将最新覆盖报告落盘到 `docs/reports/operator_coverage.md`。

继续补充 `Elu`、`LeakyRelu`、`PRelu`、`Selu`、`Celu` 与 `ThresholdedRelu` 的 CUDA 参考验证程序，并将六者的 float32、float16、bfloat16、float8_e4m3、float8_e5m2 计划接入默认 numerical 门禁。`PRelu` 数值计划使用输入 `(2, 3, 4)` 与 slope `(1, 3, 1)`，覆盖 ONNX 多向广播主路径；其余激活算子通过 params.bin 传递 alpha/gamma 属性，避免用固定默认值绕过属性语义。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op elu --op leaky_relu --op prelu --op selu --op celu --op thresholded_relu --iterations 3 --skip-plots` 与完整一轮 numerical 均已通过。本轮后默认 active numerical plan 覆盖 99 个唯一算子名称、325 条默认计划，其中混合精度计划 222 条。

继续补充 `HardSwish`、`Shrink`、`Gelu` 与 `Mish` 的 CUDA 参考验证程序，并将四者的 float32、float16、bfloat16、float8_e4m3、float8_e5m2 计划接入默认 numerical 门禁。`HardSwish` 固定样本覆盖 `-3/3` 分段边界，`Shrink` 固定样本覆盖 `±lambd` 分段边界且通过 params.bin 传递 `bias/lambd` 属性，`Gelu` 当前 numerical 覆盖精确 erf 公式路径，`Mish` 覆盖 `x * tanh(softplus(x))` 主路径。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op hard_swish --op shrink --op gelu --op mish --iterations 3 --skip-plots` 与完整一轮 numerical 均已通过。本轮后默认 active numerical plan 覆盖 103 个唯一算子名称、345 条默认计划，其中混合精度计划 238 条。

随后补齐 `Gelu` 的 ONNX opset 20 `approximate` 属性语义：导入器保留 `approximate="tanh"`，Python runtime 优先调用 C 后端新增的 `gelu_forward_mode`，CUDA verifier 通过 params.bin 区分 exact erf 与 tanh 近似路径。新增 pytest 使用 ONNX `ReferenceEvaluator` 校验 importer 与 bfloat16 混合精度路径，并将 tanh 近似的 float32、float16、bfloat16、float8_e4m3、float8_e5m2 计划接入默认 numerical 门禁。按收尾要求再次执行完整 numerical：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，所有计划均通过，无失败算子。本轮后默认 active numerical plan 覆盖 103 个唯一算子名称、350 条默认计划，其中混合精度计划 242 条。

继续补充 `Hardmax` 与 `LogSoftmax` 的 CUDA 参考验证程序，复用 Softmax 族的 `outer/inner/remaining` axis 参数布局，并将二者的 float32、float16、bfloat16、float8_e4m3、float8_e5m2 计划接入默认 numerical 门禁。`Hardmax` 覆盖 ONNX 第一最大值 one-hot 语义，`LogSoftmax` 使用稳定形式 `x - max - log(sum(exp(x - max)))`。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op hardmax --op log_softmax --iterations 3 --skip-plots` 与完整一轮 `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots` 均已通过。本轮后默认 active numerical plan 覆盖 105 个唯一算子名称、360 条默认计划，其中混合精度计划 250 条。

继续补充 `ReduceL1`、`ReduceL2`、`ReduceLogSum`、`ReduceLogSumExp` 与 `ReduceSumSquare` 的 CUDA 参考验证程序，并将五者的 float32、float16、bfloat16、float8_e4m3、float8_e5m2 计划接入默认 numerical 门禁。当前数值计划覆盖 axes=None、keepdims=0 的全量归约主路径；`ReduceLogSum` 使用正输入样本，`ReduceLogSumExp` 使用稳定公式避免指数溢出。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op reduce_l1 --op reduce_l2 --op reduce_log_sum --op reduce_log_sum_exp --op reduce_sum_square --iterations 3 --skip-plots`、相关 pytest 公式语义测试和完整一轮 numerical 均已通过。本轮后默认 active numerical plan 覆盖 110 个唯一算子名称、385 条默认计划，其中混合精度计划 270 条。

继续补充 `Round`、`Erf`、`Acos`、`Asin`、`Cosh`、`Sinh`、`Asinh`、`Acosh` 与 `Atanh` 的 CUDA 参考验证程序，并将九者的 float32、float16、bfloat16、float8_e4m3、float8_e5m2 计划接入默认 numerical 门禁。数值计划为 `Acos`/`Asin`/`Atanh`/`Acosh` 使用受控定义域样本，避免随机输入越界产生 NaN 干扰主路径；`Round` 覆盖 `±0.5`、`±1.5`、`±2.5` 等 ties-to-even 舍入样本。`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op round --op erf --op acos --op asin --op cosh --op sinh --op asinh --op acosh --op atanh --iterations 3 --skip-plots`、相关激活语义 pytest 和完整一轮 `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots` 均已通过。本轮后默认 active numerical plan 覆盖 119 个唯一算子名称、430 条默认计划，其中混合精度计划 306 条。

继续补充基础元素级算子的 ONNX reference pytest 语义覆盖，新增 `tests/test_operator_elementwise_semantics.py`，覆盖 `Add`、`Sub`、`Mul`、`Div`、`Pow`、`Max`、`Min`、比较算子、布尔逻辑算子、`Relu`、`Abs`、`Neg`、`Floor`、`Sign`、`IsNaN`、`Sin`、`Cos`、`Tan`、`Atan`、`Exp`、`Log`、`Sqrt`、`Sigmoid`、`Tanh` 与 `Softmax`。测试同时覆盖 ONNX 广播、variadic 输入、bool 输出、负 axis，以及 bfloat16 位存储读写路径。`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_elementwise_semantics.py` 与完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests` 均已通过。本轮后 ONNX reference pytest 语义/混合精度覆盖增至 101 个算子。

继续补充基础 Reduce 算子的 ONNX reference pytest 语义覆盖，新增 `tests/test_operator_reduce_semantics.py`，覆盖 `ReduceMean`、`ReduceSum`、`ReduceMax`、`ReduceMin` 与 `ReduceProd` 的属性 axes、运行时 axes 输入、空 axes、`noop_with_empty_axes`、默认 `keepdims=1` 和 bfloat16 位存储路径。新增测试暴露并修复了 C 后端在 keepdims 输出布局下的归约坐标映射问题：保留维度时输出坐标需要与输入维度对齐，只有非 keepdims 输出才压缩非归约维度坐标。`make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests`、Reduce 定向 numerical 和完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots` 均已通过。本轮后 ONNX reference pytest 语义/混合精度覆盖增至 106 个算子。

继续补充索引、Arg、Scatter、TopK 与 CumSum 算子的 ONNX reference pytest 语义覆盖，新增 `tests/test_operator_index_semantics.py`，覆盖 `Gather`、`GatherElements`、`GatherND`、`ScatterND`、`NonZero`、`ArgMax`、`ArgMin`、`TopK` 与 `CumSum`。测试覆盖负 axis、负索引、`GatherND batch_dims=1`、`ScatterND reduction=none/add/mul`、`Arg* select_last_index`、`TopK largest=0 sorted=1`、`CumSum exclusive+reverse` 和 bfloat16 位存储路径。`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_index_semantics.py`、完整 pytest 和索引组定向 numerical 均已通过。本轮后 ONNX reference pytest 语义/混合精度覆盖增至 115 个算子。

按当前收尾要求再次执行完整 numerical：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`。本次复跑覆盖默认 active numerical plan 中的 float32、float16、bfloat16、float8_e4m3、float8_e5m2、量化、索引和布尔类型路径，所有计划均通过，无失败算子；MaxRoiPool、RoiAlign、RNN、GRU、LSTM、DFT、STFT 等高风险算子也在本次完整一轮中通过。

继续补充核心数值算子的官方语义覆盖，新增 `tests/test_operator_core_numeric_semantics.py`，覆盖 `Conv`、`ConvInteger`、`QLinearConv`、`ConvTranspose`、`Gemm`、`MatMul`、`MatMulInteger`、`QLinearMatMul`、`MaxPool`、`AveragePool`、`LpPool`、`GlobalAveragePool`、`GlobalMaxPool`、`GlobalLpPool`、`QuantizeLinear`、`DequantizeLinear`、`Clip` 与 `Mod`。测试覆盖 group/dilation/pads、ConvTranspose output_padding、Gemm 转置和 alpha/beta/C 广播、MatMul batch broadcast 和一维输入、Pool ceil/pad/count_include_pad、量化负 axis per-axis、整数零点和 bfloat16 位存储路径。`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_core_numeric_semantics.py`、完整 pytest 和核心数值组定向 numerical 均已通过。本轮后 ONNX reference pytest 语义/混合精度覆盖增至 131 个算子，独立 pytest 深度语义/混合精度覆盖增至 51 个算子。

继续补充复杂属性算子的官方语义覆盖，新增 `tests/test_operator_complex_attribute_semantics.py`，覆盖 `Resize`、`Pad`、`GridSample`、`MaxUnpool` 与 `Einsum` 的高风险属性组合。测试覆盖 Resize 的 `align_corners`、`nearest_mode=ceil`、cubic 与 `round_prefer_ceil` fallback，Pad 的 edge/reflect/负 pad 裁剪，GridSample 的 nearest/border 与 linear/reflection，MaxUnpool 的显式 `output_shape`，以及 Einsum 的 ellipsis、重复标签和 bfloat16 位存储路径。本轮同时修复 `Einsum` 混合精度路径：低精度输入先解码为 float32 临时 CTensor 后交给 C stride planner 累加，C 输出已按目标 dtype 写回时不再二次 `_cast_numeric_to_dtype`，并将 C 后端累加改为 double 临时缓冲后统一写回，避免低精度输出缓冲参与中间累加。`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_complex_attribute_semantics.py`、完整 pytest 和 `Resize/Pad/Einsum/MaxUnpool` 定向 numerical 均已通过。本轮后 ONNX reference pytest 语义/混合精度覆盖增至 134 个算子。

按当前结束要求执行完整一轮 numerical：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`。本次验证覆盖默认 active numerical plan 中的基础数值、卷积、池化、量化、归约、索引、形状变换、谱、序列和混合精度路径，所有计划均通过，无失败算子；MaxRoiPool、RoiAlign、RNN、GRU、LSTM、DFT、STFT 等此前标记为高风险的算子也在本次完整一轮中通过。当前收尾未新增运行逻辑，仅将本次验证结果落盘，作为提交前状态记录。

继续校准覆盖审计口径：`AveragePool` 已在 `tests/test_operator_core_numeric_semantics.py` 中通过 ONNX reference 覆盖 padding、ceil_mode、count_include_pad 和 bfloat16 位存储路径，本轮将其补入审计登记，`tools/audit_ops.py` 当前显示 186 个算子类均已有 pytest 语义/混合精度覆盖记录。审计脚本同时新增“当前安装 ONNX 最新官方覆盖”段，除既有 opset 17 官方名称级覆盖 178/178 外，额外暴露当前环境最新默认 domain schema 的 15 个高版本缺口：AffineGrid、Attention、BitCast、CenterCropPad、Col2Im、CumProd、DeformConv、ImageDecoder、RegexFullMatch、RMSNormalization、RotaryEmbedding、StringConcat、StringSplit、Swish、TensorScatter。

继续处理当前安装 ONNX 最新 schema 缺口，补齐 `Swish` opset 24：导入器读取 `alpha` 属性并构造 `Swish` 算子，Python runtime 调用 C 后端新增 `swish_forward`，C 后端按 `x * sigmoid(alpha * x)` 写回目标 dtype，CUDA verifier 通过 `params.bin` 读取 alpha 并作为 numerical reference。默认 numerical plan 新增 float32、float16、bfloat16、float8_e4m3、float8_e5m2 五条计划，pytest 使用 ONNX `ReferenceEvaluator` 校验默认 alpha、非默认 alpha、导入器属性保留和 bfloat16 位存储路径。验证命令：`make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_activation_semantics.py`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op swish --iterations 3 --skip-plots` 均已通过；随后完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests` 结果为 241 passed、1 skipped，完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots` 也全部通过。本轮后当前安装 ONNX 最新官方名称级覆盖从 185/200 提升到 186/200，最新 schema 缺口从 15 个降至 14 个。

继续处理当前安装 ONNX 最新 schema 缺口，补齐 `CumProd` opset 26：导入器复用扫描类 factory 并保留 `exclusive`、`reverse` 属性，Python runtime 调用 C 后端新增 `cumprod_forward`，C 后端支持负 axis、exclusive、reverse、整数 wrap 写回和低精度 dtype 写回，CUDA verifier 使用 `[N, exclusive, reverse]` 参数布局作为一维累计乘积 reference。默认 numerical plan 新增 float32、float16、bfloat16、float8_e4m3、float8_e5m2 五条计划，pytest 使用 ONNX `ReferenceEvaluator` 校验二维负轴、exclusive+reverse、导入器属性保留和 bfloat16 位存储路径。验证命令：`make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_index_semantics.py`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op cumprod --iterations 3 --skip-plots` 均已通过；随后完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests` 结果为 243 passed、1 skipped。按当前收尾要求执行完整一轮 `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，默认 active numerical plan 的 121 个唯一算子、440 条默认计划、314 条混合精度计划全部通过。本轮后当前安装 ONNX 最新官方名称级覆盖从 186/200 提升到 187/200，最新 schema 缺口从 14 个降至 13 个。

继续处理当前安装 ONNX 最新 schema 缺口，补齐 `RMSNormalization` opset 23：导入器读取 `axis`、`epsilon`、`stash_type` 属性并构造 `RMSNormalization` 算子，Python runtime 调用 C 后端新增 `rms_normalization_forward`，C 后端支持任意合法 axis 后缀 RMS、scale 单向广播、float32/double stash 计算和低精度 dtype 写回，CUDA verifier 以已广播 scale 和 `[row_count, normalized_size, epsilon]` 参数作为 reference。默认 numerical plan 新增 float32、float16、bfloat16、float8_e4m3、float8_e5m2 五条计划，pytest 使用 ONNX `ReferenceEvaluator` 校验 axis=-1、axis=1、scale 广播、导入器属性保留和 bfloat16 位存储路径。验证命令：`make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_normalization_semantics.py`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op rms_normalization --iterations 3 --skip-plots` 均已通过；随后完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests` 结果为 246 passed、1 skipped。按当前收尾要求执行完整一轮 `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，默认 active numerical plan 的 122 个唯一算子、445 条默认计划、318 条混合精度计划全部通过。本轮后当前安装 ONNX 最新官方名称级覆盖从 187/200 提升到 188/200，最新 schema 缺口从 13 个降至 12 个。

继续处理当前安装 ONNX 最新 schema 缺口，补齐 `AffineGrid` opset 20：导入器读取 `align_corners` 属性并构造 `AffineGrid` 算子，Python runtime 优先调用 C 后端新增 `affine_grid_forward`，在旧共享库未重编译时保留 ONNX reference fallback；C 后端支持 2D `[N,H,W,2]` 与 3D `[N,D,H,W,3]` 网格生成，并按输出 dtype 写回低精度位模式；CUDA verifier 使用 `[spatial_rank, N, D, H, W, align_corners]` 参数作为 reference。默认 numerical plan 新增 float32、float16、bfloat16、float8_e4m3、float8_e5m2 五条计划，pytest 覆盖 2D、3D `align_corners=1`、导入器属性保留和 bfloat16 位存储路径。验证命令：`make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_shape_semantics.py -k affine_grid`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op affine_grid --iterations 3 --skip-plots` 均已通过；随后完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests` 结果为 250 passed、1 skipped。按当前收尾要求执行完整一轮 `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，默认 active numerical plan 的 123 个唯一算子、450 条默认计划、322 条混合精度计划全部通过。本轮后当前安装 ONNX 最新官方名称级覆盖从 188/200 提升到 189/200，最新 schema 缺口从 12 个降至 11 个。

按当前请求在结束前再次执行完整默认 numerical 一轮：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`。本次执行覆盖默认 active numerical plan 的 123 个唯一算子、450 条默认计划和 322 条混合精度计划，450/450 全部通过，无失败算子；覆盖 dtype 包含 float32、float16、bfloat16、float8_e4m3、float8_e5m2、int8、uint8、int64 和 bool。MaxRoiPool、RoiAlign、RNN、GRU、LSTM、DFT、STFT 等此前重点关注算子均在本轮完整默认门禁中通过。随后执行 `tools/audit_ops.py --output docs/reports/operator_coverage.md` 刷新覆盖报告，当前 opset 17 官方名称级覆盖保持 178/178，当前安装 ONNX 最新默认 domain schema 覆盖保持 189/200。

继续处理当前安装 ONNX 最新 schema 缺口，补齐 `BitCast` opset 26：导入器读取 `to` 属性并构造 `BitCast` 算子，Python runtime 调用 C 后端新增 `bitcast_forward`，C 后端按等宽非字符串 dtype 做原始字节复制，不对数值进行转换。CUDA verifier 使用字节级拷贝作为 reference，numerical runner 对 `BitCast` 改用原始字节比较，避免把位模式重新解释为浮点值后引入非语义差异。默认 numerical plan 新增 float32/int32 双向、float16->int16、bfloat16->uint16、float8_e4m3->uint8 和 uint8->float8_e5m2 六条计划，pytest 使用 ONNX `ReferenceEvaluator` 校验官方参考语义、导入器属性保留和低精度位存储路径。按当前要求在结束前执行完整默认 numerical 一轮：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`。本次执行覆盖默认 active numerical plan 的 124 个唯一算子、456 条默认计划和 326 条混合精度计划，456/456 全部通过，无失败算子；覆盖 dtype 包含 float32、float16、bfloat16、float8_e4m3、float8_e5m2、int8、uint8、int64 和 bool。随后 `docs/reports/operator_coverage.md` 已刷新，当前 opset 17 官方名称级覆盖保持 178/178，当前安装 ONNX 最新默认 domain schema 覆盖提升到 190/200。

继续处理当前安装 ONNX 最新 schema 缺口，补齐 `CenterCropPad` opset 18：导入器读取 `axes` 属性并构造 `CenterCropPad` 算子，Python runtime 调用 C 后端新增 `center_crop_pad_forward`，C 后端根据输入和输出 shape 执行官方中心裁剪/零填充语义，奇数裁剪差值按向下取整选择左侧窗口，奇数 padding 的额外像素落在右侧。Python fallback 覆盖 string 等 C 后端不承载的 dtype。CUDA verifier 使用同样的中心坐标映射作为 reference，默认 numerical plan 新增 float32、float16、bfloat16、float8_e4m3、float8_e5m2 五条计划，pytest 使用 ONNX `ReferenceEvaluator` 校验全 axes、负 axes 子集、导入器属性保留和低精度位存储路径。验证命令：`make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_shape_semantics.py`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op center_crop_pad --iterations 3 --skip-plots` 均已通过。随后刷新 `docs/reports/operator_coverage.md`，默认 active numerical plan 覆盖提升到 125 个唯一算子、461 条默认计划和 330 条混合精度计划，当前安装 ONNX 最新默认 domain schema 覆盖提升到 191/200。

继续处理当前安装 ONNX 最新 schema 缺口，补齐 `TensorScatter` opset 24：导入器读取 `axis` 与 `mode` 属性并构造 `TensorScatter` 算子，Python runtime 调用 C 后端新增 `tensor_scatter_forward`，C 后端先复制 `past_cache`，再按 batch 级 `write_indices` 将 `update` 写入 sequence 轴，支持 `linear` 和 `circular` 两种模式；optional `write_indices` 缺省时按全 0 起点处理。低精度路径使用元素原始字节复制，避免 bfloat16/float8 位模式被二次数值转换。CUDA verifier 使用完整 update 坐标映射作为 reference，默认 numerical plan 新增 float32、float16、bfloat16、float8_e4m3、float8_e5m2 五条计划，pytest 使用 ONNX `ReferenceEvaluator` 校验 linear、circular、缺省 write_indices、导入器属性保留和低精度位存储路径。验证命令：`make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_index_semantics.py -k tensor_scatter`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op tensor_scatter --iterations 3 --skip-plots` 均已通过；随后完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests` 结果为 263 passed、1 skipped，完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots` 也全部通过。随后刷新 `docs/reports/operator_coverage.md`，默认 active numerical plan 覆盖提升到 126 个唯一算子、466 条默认计划和 334 条混合精度计划，当前安装 ONNX 最新默认 domain schema 覆盖提升到 192/200。

继续处理当前安装 ONNX 最新 schema 缺口，补齐 `RegexFullMatch`、`StringConcat` 和 `StringSplit` opset 20：导入器分别读取 `pattern`、`delimiter` 和 `maxsplit` 属性并构造对应字符串算子。`RegexFullMatch` 按 fullmatch 逐元素输出 bool，`StringConcat` 按 NumPy-style broadcasting 逐元素拼接字符串，`StringSplit` 按 delimiter 或默认连续空白拆分，并将每个输入元素的拆分结果用空字符串补齐到统一最后维，同时输出 int64 拆分数量。该组三个算子是字符串语义算子，不存在 float16/bfloat16/float8 混合精度路径，也不适合接入数值 C/CUDA 门禁；本轮使用 ONNX `ReferenceEvaluator` 覆盖广播拼接、正则 fullmatch、空输入、delimiter/maxsplit、默认空白拆分、padding 和导入器属性保留。验证命令：`/home/sakauma/data/miniconda3/envs/egor/bin/python -m py_compile nn/operators/text.py nn/importer/node_factories_04.py tools/audit_ops.py tests/test_operator_text_semantics.py`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_text_semantics.py` 均已通过。随后刷新 `docs/reports/operator_coverage.md`，当前安装 ONNX 最新默认 domain schema 覆盖提升到 195/200。

按当前收尾要求执行完整默认 numerical 一轮：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`。本次执行覆盖默认 active numerical plan 的 126 个唯一算子、466 条默认计划和 334 条混合精度计划，466/466 全部通过，无失败算子；覆盖 dtype 包含 float32、float16、bfloat16、float8_e4m3、float8_e5m2、int8、uint8、int64 和 bool。随后执行 `tools/audit_ops.py --output docs/reports/operator_coverage.md` 刷新覆盖报告，当前安装 ONNX 最新默认 domain schema 覆盖保持 195/200，剩余缺口保持为 `Attention(since_version=24)`、`Col2Im(since_version=18)`、`DeformConv(since_version=22)`、`ImageDecoder(since_version=20)`、`RotaryEmbedding(since_version=23)`。本轮未新增运行逻辑，仅将结束前完整验证结果落盘。

继续处理当前安装 ONNX 最新 schema 缺口，补齐 `RotaryEmbedding` opset 23：新增 `nn/operators/embedding_ops.py` 分组并实现 3D/4D 输入、`position_ids` 查表、无 position 的 3D cache、`interleaved` 和 partial `rotary_embedding_dim` 语义；导入器读取 `num_heads`、`rotary_embedding_dim` 和 `interleaved` 属性；C 后端新增 `rotary_embedding_forward`，按原始输入布局写回结果，并对未旋转尾部维度直接复制元素位模式；CUDA verifier 新增 `verify_rotary_embedding.cu`，作为 numerical 的独立参考。默认 numerical plan 新增 float32、float16、bfloat16 三条计划，未添加 float8 计划，因为官方 type constraint 只声明 float32、float16、bfloat16。验证命令：`make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_rotary_embedding_semantics.py`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op rotary_embedding --iterations 3 --skip-plots`、完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests` 和完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots` 均已通过。本轮后默认 active numerical plan 覆盖 127 个唯一算子、469 条默认计划和 336 条混合精度计划，当前安装 ONNX 最新默认 domain schema 覆盖提升到 196/200。

继续处理当前安装 ONNX 最新 schema 缺口，补齐 `Col2Im` opset 18：导入器读取 `pads`、`strides`、`dilations` 属性并构造 `Col2Im` 算子，Python runtime 优先调用 C 后端新增 `col2im_forward`，C 后端按 N-D fold 语义将 `[N, C * prod(block_shape), L]` 列块累加回 `[N, C, *image_shape]`，Python fallback 复用 ONNX reference 的 `col2im_naive_implementation`。CUDA verifier 新增 `verify_col2im.cu`，以输出元素反查列块贡献的方式作为 numerical reference，避免重叠写入的原子累加差异。默认 numerical plan 新增 float32、float16、bfloat16 三条计划，pytest 使用 ONNX `ReferenceEvaluator` 校验带 pads/strides/dilations 的官方语义、导入器属性保留、Python fallback 和 bfloat16 位存储路径。验证命令：`make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_col2im_semantics.py`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op col2im --iterations 3 --skip-plots` 均已通过；随后完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests` 结果为 279 passed、1 skipped。按当前收尾要求执行完整一轮 `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，默认 active numerical plan 的 128 个唯一算子、472 条默认计划和 338 条混合精度计划全部通过。本轮后当前安装 ONNX 最新默认 domain schema 覆盖提升到 197/200，剩余最新官方名称级缺口为 `Attention(since_version=24)`、`DeformConv(since_version=22)`、`ImageDecoder(since_version=20)`。

继续处理当前安装 ONNX 最新 schema 缺口，补齐 `ImageDecoder` opset 20：新增图像算子分组 `nn/operators/image_ops.py`，实现 uint8 encoded stream 到 channel-last uint8 图像的解码；`pixel_format` 支持官方 `RGB`、`BGR` 和 `Grayscale`，其中 BGR 按最后通道反转，Grayscale 转 `L` 后扩为 `(H, W, 1)`。导入器读取 `pixel_format` 属性并构造 `ImageDecoder`；解码失败时按 schema 文档返回空 uint8 图像矩阵。该算子是图像 IO/解码类，不属于普通数值 C/CUDA 后端或混合精度门禁范围，因此登记为合理保留 Python runtime。验证命令：`/home/sakauma/data/miniconda3/envs/egor/bin/python -m py_compile nn/operators/image_ops.py nn/operators/__init__.py nn/importer/node_factories_04.py tools/audit_ops.py tests/operator_test_context.py tests/test_operator_image_decoder_semantics.py`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_image_decoder_semantics.py` 和完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests` 均已通过；完整 pytest 结果为 284 passed、1 skipped。随后刷新 `docs/reports/operator_coverage.md`，当前安装 ONNX 最新默认 domain schema 覆盖提升到 198/200，剩余最新官方名称级缺口为 `Attention(since_version=24)`、`DeformConv(since_version=22)`。

继续处理当前安装 ONNX 最新 schema 缺口，补齐 `DeformConv` opset 22：导入器读取 `strides`、`pads`、`dilations`、`group`、`kernel_shape` 和 `offset_group` 属性并构造算子；Python runtime 对 2D 主路径调用 C 后端新增 `deform_conv2d_forward`，覆盖 group、offset_group、bias、mask 和双线性采样语义，非 C 可承载路径继续复用 ONNX reference fallback。CUDA verifier 新增 `verify_deform_conv.cu`，用独立 reference kernel 与 C 后端输出对比。默认 numerical plan 新增 float32、float16、bfloat16 三条计划，pytest 使用 ONNX `ReferenceEvaluator` 校验 group/offset_group/mask/bias 官方语义、导入器属性保留、Python fallback 和 bfloat16 位存储路径。验证命令：`make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_deform_conv_semantics.py`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op deform_conv --iterations 3 --skip-plots` 均已通过；随后完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests` 结果为 288 passed、1 skipped。按当前收尾要求执行完整一轮 `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，默认 active numerical plan 的 129 个唯一算子、475 条默认计划和 340 条混合精度计划全部通过。本轮后当前安装 ONNX 最新默认 domain schema 覆盖提升到 199/200，剩余最新官方名称级缺口为 `Attention(since_version=24)`。

继续处理当前安装 ONNX 最新 schema 最后一个缺口，补齐 `Attention` opset 24：导入器读取 `q_num_heads`、`kv_num_heads`、`scale`、`is_causal`、`softmax_precision`、`softcap` 和 `qk_matmul_output_mode` 属性并构造算子；Python runtime 复用 ONNX reference 覆盖 3D/4D、MHA/GQA/MQA、KV cache、`nonpad_kv_seqlen`、mask、causal、softcap 和 qk 中间输出，4D 无 cache 主路径调用 C 后端新增 `attention_forward`，覆盖 QK/V matmul、GQA 头映射、float/bool mask、causal、softcap 和低精度 dtype 写回。CUDA verifier 新增 `verify_attention.cu`，默认 numerical plan 新增 float32、float16、bfloat16 三条计划。pytest 覆盖 C 后端 4D GQA+float mask+causal+softcap、bfloat16 位存储、Python fallback 的 3D cache/qk 输出和导入器属性保留。验证命令：`make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_attention_semantics.py`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op attention --iterations 3 --skip-plots` 均已通过。随后刷新 `docs/reports/operator_coverage.md`，默认 active numerical plan 覆盖提升到 130 个唯一算子、478 条默认计划和 342 条混合精度计划；当前安装 ONNX 最新默认 domain schema 覆盖提升到 200/200，最新官方名称级缺口为 0。

按当前收尾要求再次执行完整默认 numerical 一轮。首次复跑时暴露 `QuantizeLinear` 的 CUDA verifier 在 float32 输入下使用 double 中间除法，和 C 后端/ONNX reference 的 float32 中间舍入规则在半点附近可能差 1；本轮已修正 CUDA verifier，并由 runner 显式传递量化计算精度模式。修复后执行 `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op quantize_linear --iterations 3 --skip-plots`，float32、float16、bfloat16 三组均通过；随后执行完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，默认 active numerical plan 的 130 个唯一算子、478 条默认计划和 342 条混合精度计划全部通过，无失败算子。

继续补强已有 C 后端但未进入 CUDA numerical 的高风险数值算子，补充 `GridSample` CUDA 参考验证程序，并将 float32、float16、bfloat16 三条计划接入默认 numerical 门禁。CUDA reference 覆盖 4D NCHW 输入、`mode="linear"`、`padding_mode="reflection"`、`align_corners=0`、归一化坐标映射和低精度写回；pytest 仍覆盖 nearest/border、linear/reflection 等官方属性组合。验证命令：`/home/sakauma/data/miniconda3/envs/egor/bin/python -m py_compile tools/numerical/cli.py tools/numerical/runner.py`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op grid_sample --iterations 3 --skip-plots`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_complex_attribute_semantics.py -k grid_sample` 均已通过。随后执行完整 `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，默认 active numerical plan 提升到 131 个唯一算子、481 条默认计划和 344 条混合精度计划，全部通过，无失败算子；当前安装 ONNX 最新默认 domain schema 名称级覆盖保持 200/200。

继续补强 `LRN` 的 C/CUDA 数值门禁：新增 `cuda/verify_lrn.cu`，按 ONNX schema 公式对 NCHW 及展开 spatial 维度执行跨通道平方和窗口归一化；numerical runner 新增固定有限样本和 `[N, C, spatial, size] + [alpha, beta, bias]` 参数包，默认计划新增 float32、float16、bfloat16 三条验证路径。验证命令：`/home/sakauma/data/miniconda3/envs/egor/bin/python -m py_compile tools/numerical/cli.py tools/numerical/runner.py`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op lrn --iterations 3 --skip-plots`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_normalization_semantics.py -k lrn` 均已通过。按当前请求最后执行完整默认 numerical 一轮：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，默认 active numerical plan 提升到 132 个唯一算子、484 条默认计划和 346 条混合精度计划，全部通过，无失败算子；随后执行 `tools/audit_ops.py --output docs/reports/operator_coverage.md` 刷新覆盖报告，当前安装 ONNX 最新默认 domain schema 名称级覆盖保持 200/200。

继续补强 `MeanVarianceNormalization` 的 C/CUDA 数值门禁：新增 `cuda/verify_mean_variance_normalization.cu`，按 rank/axes 参数对任意展开坐标计算均值、方差和 `(x - mean) / sqrt(variance)`；numerical runner 新增按通道分布不同的固定有限样本，并通过参数包传递 rank、shape 和归约 axes。默认计划新增 float32、float16、bfloat16 三条验证路径。验证命令：`/home/sakauma/data/miniconda3/envs/egor/bin/python -m py_compile tools/numerical/cli.py tools/numerical/runner.py`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op mean_variance_normalization --iterations 3 --skip-plots`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_normalization_semantics.py -k "mean_variance or mvn"` 均已通过。随后执行完整默认 numerical 一轮：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，默认 active numerical plan 提升到 133 个唯一算子、487 条默认计划和 348 条混合精度计划，全部通过，无失败算子；`tools/audit_ops.py --output docs/reports/operator_coverage.md` 已刷新，当前安装 ONNX 最新默认 domain schema 名称级覆盖保持 200/200。

继续补强 `BatchNormalization` 的 C/CUDA 数值门禁：新增 `cuda/verify_batch_normalization.cu`，按 ONNX 推理模式公式 `scale * (x - mean) / sqrt(var + epsilon) + bias` 作为独立 CUDA reference；numerical runner 新增固定有限样本、`[N, C, spatial_size] + epsilon` 参数包，并为 float32、float16、bfloat16 三条默认计划接入门禁。验证命令：`/home/sakauma/data/miniconda3/envs/egor/bin/python -m py_compile tools/numerical/cli.py tools/numerical/runner.py`、`git diff --check`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op batch_normalization --iterations 3 --skip-plots`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_normalization_semantics.py -k batch_normalization` 均已通过。按当前收尾要求最后执行完整默认 numerical 一轮：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，默认 active numerical plan 提升到 134 个唯一算子、490 条默认计划和 350 条混合精度计划，全部通过，无失败算子；`tools/audit_ops.py --output docs/reports/operator_coverage.md` 已刷新，当前安装 ONNX 最新默认 domain schema 名称级覆盖保持 200/200。

继续补强归一化组的 C/CUDA 数值门禁：新增 `cuda/verify_instance_normalization.cu` 和 `cuda/verify_layer_normalization.cu`。`InstanceNormalization` CUDA reference 覆盖 NCHW 输入的 per-instance/per-channel spatial 均值方差、scale/bias 通道参数和低精度写回；`LayerNormalization` CUDA reference 覆盖当前 C 后端承载的 `axis=-1` 单输出主路径，并通过 `[row_count, normalized_size, has_scale, has_bias] + epsilon` 参数包对齐 runner。默认计划新增两者的 float32、float16、bfloat16 六条验证路径。验证命令：`/home/sakauma/data/miniconda3/envs/egor/bin/python -m py_compile tools/numerical/cli.py tools/numerical/runner.py`、`git diff --check`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op instance_normalization --op layer_normalization --iterations 3 --skip-plots`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_normalization_semantics.py` 均已通过。随后执行完整默认 numerical 一轮：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，默认 active numerical plan 提升到 136 个唯一算子、496 条默认计划和 354 条混合精度计划，全部通过，无失败算子；`tools/audit_ops.py --output docs/reports/operator_coverage.md` 已刷新，当前安装 ONNX 最新默认 domain schema 名称级覆盖保持 200/200。

继续补强 `LpNormalization` 与 `GroupNormalization` 的 C/CUDA 数值门禁：新增 `cuda/verify_lp_normalization.cu` 和 `cuda/verify_group_normalization.cu`。`LpNormalization` CUDA reference 按 `[outer, inner, remaining, p]` 参数沿指定 axis 计算 `x / ||x||_p`，默认计划覆盖 p=1、p=2 的 float32 以及 p=2 的 float16/bfloat16；`GroupNormalization` CUDA reference 按 `[N, C, spatial_size, num_groups] + epsilon` 参数计算每个 batch/group 的均值方差，并应用通道 scale/bias，默认计划覆盖 float32、float16、bfloat16。验证命令：`/home/sakauma/data/miniconda3/envs/egor/bin/python -m py_compile tools/numerical/cli.py tools/numerical/runner.py`、`git diff --check`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op lp_normalization --op group_normalization --iterations 3 --skip-plots`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_normalization_semantics.py tests/test_operator_domain_ops.py -k "LpNormalization or lp_normalization or normalization"`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_misc_semantics.py::test_c_backend_extra_formula_ops_match_expected_mixed_precision` 均已通过。随后执行完整默认 numerical 一轮：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，默认 active numerical plan 提升到 138 个唯一算子、503 条默认计划和 358 条混合精度计划，全部通过，无失败算子；`tools/audit_ops.py --output docs/reports/operator_coverage.md` 已刷新，当前安装 ONNX 最新默认 domain schema 名称级覆盖保持 200/200。

继续补强基础图结构与形状算子的 C/CUDA 数值门禁：新增 `cuda/verify_identity.cu`、`cuda/verify_where.cu`、`cuda/verify_isinf.cu`、`cuda/verify_size.cu`、`cuda/verify_squeeze.cu` 和 `cuda/verify_unsqueeze.cu`。`Identity`、`Squeeze`、`Unsqueeze` CUDA reference 验证连续内存元素保持不变；`Where` 验证 runner 按广播物化后的条件选择路径；`IsInf` 通过 `[detect_positive, detect_negative]` 参数覆盖正负无穷检测；`Size` 通过 int64 参数验证标量元素数输出。默认计划新增六者的 float32 主路径，并为 `Identity`、`Where`、`Squeeze`、`Unsqueeze` 增加 float16、bfloat16、float8_e4m3、float8_e5m2 混合精度计划，为 `IsInf`、`Size` 增加 float16/bfloat16 输入计划。验证命令：`/home/sakauma/data/miniconda3/envs/egor/bin/python -m py_compile tools/numerical/cli.py tools/numerical/runner.py`、`git diff --check`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op identity --op where --op isinf --op size --op squeeze --op unsqueeze --iterations 3 --skip-plots`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_shape_semantics.py tests/test_operator_import_and_shape.py tests/test_operator_misc_semantics.py` 均已通过。随后执行完整默认 numerical 一轮：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，默认 active numerical plan 提升到 144 个唯一算子、529 条默认计划和 378 条混合精度计划，全部通过，无失败算子；`tools/audit_ops.py --output docs/reports/operator_coverage.md` 已刷新，当前安装 ONNX 最新默认 domain schema 名称级覆盖保持 200/200。

继续补强 NCHW 布局重排算子的 C/CUDA 数值门禁：新增 `cuda/verify_depth_to_space.cu` 和 `cuda/verify_space_to_depth.cu`。`DepthToSpace` CUDA reference 按 `[N, C, H, W, blocksize, mode]` 参数从输出坐标反推输入坐标，覆盖 DCR 与 CRD 两种 mode；`SpaceToDepth` CUDA reference 按 `[N, C, H, W, blocksize]` 参数将空间 block 展开到通道维。默认计划新增 `DepthToSpace` 的 float32 DCR/CRD 主路径、`SpaceToDepth` 的 float32 主路径，并为两者增加 float16、bfloat16、float8_e4m3、float8_e5m2 混合精度计划。验证命令：`/home/sakauma/data/miniconda3/envs/egor/bin/python -m py_compile tools/numerical/cli.py tools/numerical/runner.py`、`git diff --check`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`、`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op depth_to_space --op space_to_depth --iterations 3 --skip-plots`、`/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_shape_semantics.py tests/test_operator_domain_ops.py` 均已通过。随后执行完整默认 numerical 一轮：`/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`，默认 active numerical plan 提升到 146 个唯一算子、540 条默认计划和 386 条混合精度计划，全部通过，无失败算子；`tools/audit_ops.py --output docs/reports/operator_coverage.md` 已刷新，当前安装 ONNX 最新默认 domain schema 名称级覆盖保持 200/200。

### 剩余风险

- 当前 numerical 是随机样本与固定 case 的默认门禁，不等价于 ONNX 每个 opset schema 的穷尽证明。
- 当前安装 ONNX 最新默认 domain schema 名称级缺口为 0，ONNXImport 已覆盖当前环境可见的 200 个最新默认 domain 官方算子。
- QuantizeLinear/DequantizeLinear 的新版属性，例如 block quantization、output_dtype、precision、saturate 等，仍需结合目标 opset 再决定是否扩展。
- `BatchNormalization` 当前 C/CUDA numerical 覆盖推理模式主路径；training_mode 的 running mean/var 多输出语义仍主要由 Python runtime 负责，后续如目标模型依赖训练模式导出，需要再补 C/CUDA reference 和 pytest 边界集合。
- `LayerNormalization` 当前 C/CUDA numerical 覆盖 `axis=-1` 且只输出 `Y` 的主路径；非最后轴、多输出 `mean/inv_std` 路径仍主要由 Python fallback 与 pytest 语义测试覆盖，后续可继续扩展 C/CUDA reference。
- `LpNormalization` 当前 mixed precision numerical 覆盖 p=2，float32 同时覆盖 p=1/p=2；更多 axis、空维度和全零范数组合仍主要依赖 pytest 语义测试，后续可继续扩展默认计划。
- `GroupNormalization` 当前 C/CUDA numerical 覆盖 2 组 4 通道 NCHW 主路径；更多 group 数、非 4D 输入和极小方差边界仍建议后续继续补 case。
- `Where`、`Identity`、`Squeeze`、`Unsqueeze`、`Size` 和 `IsInf` 已进入默认 C/CUDA mixed precision numerical；字符串张量、复杂广播形状、运行时 axes 输入等更宽语义仍主要由 ONNX reference pytest 覆盖。
- `DepthToSpace` 与 `SpaceToDepth` 已进入默认 C/CUDA mixed precision numerical，当前数值门禁覆盖 4D NCHW、blocksize=2 和主要 mode；更多 blocksize、极小空间尺寸和异常 shape 仍主要由 pytest/导入器边界处理覆盖。
- `Gelu` 的 exact erf 与 `approximate="tanh"` 已接入导入器、Python/C 后端、CUDA verifier、pytest 和默认 mixed precision numerical；后续风险主要在更多极端输入分布和跨 opset 兼容性穷尽。
- `DeformConv` 当前 C/CUDA 数值门禁覆盖 2D 主路径，fallback 跟随本地 ONNX reference；如果后续目标模型需要更高维 deformable convolution，需要再扩展 C/CUDA reference 和 pytest 边界集合。
- `Attention` 当前 C/CUDA 数值门禁覆盖 4D 主路径，完整 cache、nonpad 和 qk 中间输出语义由 Python fallback 跟随本地 ONNX reference；后续可继续把 cache 更新和 nonpad 剪枝路径下沉到 C/CUDA。
- `GridSample` 已进入默认 C/CUDA mixed precision numerical，当前数值门禁覆盖 4D linear/reflection/align_corners=0 主路径；bicubic、nearest、border 等组合仍主要由 ONNX reference pytest 覆盖。
- ROI、序列和谱算子已进入默认门禁，但仍建议后续继续补充更多 corner case，例如多方向序列、不同布局、异常轴、空张量和边界 window 参数。
