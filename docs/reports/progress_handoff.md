<!--
/**
  ******************************************************************************
  * @file        progress_handoff.md
  * @author      Egor Izmaylov
  * @brief       记录当前算子官方语义对齐、混合精度验证进度和剩余工作。
  * @details     2026.06.05  V1.0.0  创建
  * @details     2026.06.13  V1.0.1  补充 GridSample 属性数值覆盖记录
  * @details     2026.06.13  V1.0.2  补充 ROI 算子边界属性数值覆盖记录
  * @details     2026.06.14  V1.0.3  补充循环算子 activation 和 clip 数值覆盖记录
  * @details     2026.06.14  V1.0.4  补充 Attention mask 和 scale 数值覆盖记录
  * @details     2026.06.14  V1.0.5  补充 DeformConv 分组和可选输入数值覆盖记录
  * @details     2026.06.14  V1.0.6  补充 QuantizeLinear/DequantizeLinear per-axis 数值覆盖记录
  * @details     2026.06.14  V1.0.7  补充 QuantizeLinear/DequantizeLinear 负轴和省略 zero_point 数值覆盖记录
  * @details     2026.06.14  V1.0.8  补充 QuantizeLinear/DequantizeLinear output_dtype 属性覆盖记录
  * @details     2026.06.14  V1.0.9  补充 QuantizeLinear/DequantizeLinear block_size 属性覆盖记录
  * @details     2026.06.14  V1.0.10 补充 QuantizeLinear/DequantizeLinear 16 位整数量化 dtype 覆盖记录
  * @details     2026.06.14  V1.0.11 补充 QuantizeLinear precision 属性覆盖记录
  * @details     2026.06.14  V1.0.12 补充 QuantizeLinear/DequantizeLinear 负轴尾块 blocked 数值覆盖记录
  * @details     2026.06.14  V1.0.13 补充 DequantizeLinear int32 输入 dtype 覆盖记录
  * @details     2026.06.14  V1.0.14 补充 RNN/GRU/LSTM 零长度 sequence_lens 边界覆盖记录
  * @details     2026.06.14  V1.0.15 补充 DFT/STFT 高维 axis 和前缀维数值覆盖记录
  * @details     2026.06.14  V1.0.16 补充 QuantizeLinear float8 saturate 属性数值覆盖记录
  * @details     2026.06.14  V1.0.17 补充 QuantizeLinear saturate 后完整 numerical 收尾记录
  * @details     2026.06.14  V1.0.18 补充 float8 FNUZ 量化/反量化数值覆盖记录
  * @details     2026.06.14  V1.0.19 补充 2-bit/4-bit 整数量化/反量化数值覆盖记录
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
-->

# 当前进度与剩余工作交接

> 记录时间：2026-06-14
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
- CUDA verifier：`178` 个。
- 默认 active numerical plan：`178` 个唯一算子名称，`719` 条默认计划。
- 默认 active numerical plan 混合精度覆盖：`484` 条计划。

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
- 新增 `Tril`、`Triu`、`Trilu`、`HannWindow`、`HammingWindow`、`BlackmanWindow` 的独立 CUDA verifier 和默认 numerical plan。
- 三角类通过 `params.bin` 独立传入 `k` 和 `upper`，覆盖 float32 与低精度主路径；窗口类覆盖 periodic/non-periodic 以及 float32、float16、bfloat16 存储路径。
- 新增 `Range`、`OneHot`、`ReverseSequence`、`Det`、`MelWeightMatrix` 的独立 CUDA verifier 和默认 numerical plan。
- `Range` 和 `MelWeightMatrix` 的 numerical plan 使用 `(1,)` 标量输入形状，避开空 shape 随机数据生成路径；窗函数和 `MelWeightMatrix` 通过 ONNX `output_datatype` 编码显式控制输出类型。
- 新增 `DynamicQuantizeLinear` 的独立 CUDA verifier 和默认 numerical plan，并在 runner 中按 `y`、`y_scale`、`y_zero_point` 三个输出分别校验。
- 新增 `Split` 的独立 CUDA verifier 和默认 numerical plan，并在 runner 中支持多输出逐一对比。
- 新增 `Unique` 的独立 CUDA verifier 和默认 numerical plan，并在 runner 中支持 `values`、`indices`、`inverse`、`counts` 四输出逐一对比。
- 新增 `Dropout` 的独立 CUDA verifier 和默认 numerical plan，并在 runner 中支持 `y` 与 `mask` 双输出逐一对比。
- 新增 `RandomUniform`、`RandomNormal`、`RandomNormalLike` 和 `Bernoulli` 的独立 CUDA verifier 和默认 numerical plan。
- `RandomUniformLike` 的 numerical 不再绕过 C 后端 reference，而是统一走 C-backed forward 后与 CUDA verifier 对比。
- 随机 uniform/normal/Bernoulli 的 C 后端改为按元素由 seed 派生确定性随机状态，避免 OpenMP 线程数或调度顺序影响验证结果。
- 新增 `Multinomial` 的独立 CUDA verifier 和默认 numerical plan，覆盖 int64/int32 输出、零概率、one-hot 和非归一化概率行。
- 新增 `NegativeLogLikelihoodLoss`、`SoftmaxCrossEntropyLoss` 和 `NonMaxSuppression` 的独立 CUDA verifier 和默认 numerical plan，覆盖 ignore_index、reduction、权重、log_prob 多输出、score 阈值、IoU 抑制和 center_point_box 主路径。
- 继续扩展 `NonMaxSuppression` 默认 numerical plan：新增多 batch/class、排序 tie、score 阈值等值包含、跨 batch/class 输出顺序和 float16 空输出边界 case。
- 继续扩展 `LpNormalization` 默认 numerical plan：新增 ONNX reference 明确要求的零范数返回 0 边界，并补充 axis=2、p=1、bfloat16 位写回路径。
- 修正 `LayerNormalization` C 后端的 axis 语义，从仅按 `shape[axis]` 归一化扩展为按 `axis` 后缀维度归一化；默认 numerical plan 新增 axis=1 的 float32/bfloat16 单输出路径。
- 补强 `LayerNormalization` 的 `mean/inv_std` aux 多输出语义：新增 C 后端 `layer_norm_multi_output_forward`，Python runtime 在请求辅助输出时优先调用 C 后端，CUDA verifier 输出 mean/inv_std sidecar，runner 按三路输出分别比较。
- 新增 `Binarizer` 的独立 CUDA verifier 和默认 numerical plan，覆盖 threshold 两侧和恰好等于 threshold 的严格大于边界，并补入 float16、bfloat16、float8_e4m3、float8_e5m2 混合精度门禁。
- 补强 `BatchNormalization` 的 training_mode 多输出语义：新增 C 后端 `batch_norm_training_forward`，Python runtime 训练态优先调用 C 后端，CUDA verifier 同时输出 `Y/running_mean/running_var`，runner 按三路输出分别比较。
- 默认 numerical plan 新增 BatchNormalization training_mode 的 float32、float16、bfloat16 三条计划，默认计划总数提升到 `625`，混合精度计划提升到 `429`。
- 默认 numerical plan 新增 LayerNormalization aux 输出的 float32、float16、bfloat16 三条计划，默认计划总数提升到 `628`，混合精度计划提升到 `431`。
- 继续扩展 `GridSample` 默认 numerical plan：新增 float32 `nearest + border + align_corners=0` 和 float32 `cubic + zeros + align_corners=1` 两条属性组合，并补充 float16 nearest/border 与 bfloat16 cubic/zeros 混合精度计划。
- `tools/numerical/runner.py` 为 GridSample 新增 `grid_variant` 内部测试参数，仅用于选择固定有限网格样本；构造算子前会移除该参数，公共算子接口和运行逻辑不变。
- 本轮后默认 numerical plan 保持 `178` 个唯一算子名称，默认计划提升到 `632` 条，混合精度计划提升到 `433` 条。
- 继续扩展 ROI 默认 numerical plan：新增 MaxRoiPool float32/bfloat16 的 `spatial_scale=0.5`、越界裁剪和空 ROI 输出计划；新增 RoiAlign float32/float16 的 `mode=max`、`coordinate_transformation_mode=output_half_pixel`、`sampling_ratio=0` 自适应采样和 `spatial_scale=0.75` 计划。
- `tools/numerical/runner.py` 为 ROI 算子新增 `roi_variant` 内部测试参数，仅用于选择固定 ROI 坐标和 batch indices；构造算子前会移除该参数，公共算子接口和运行逻辑不变。
- 本轮后默认 numerical plan 保持 `178` 个唯一算子名称，默认计划提升到 `636` 条，混合精度计划提升到 `435` 条。
- 继续扩展谱算子默认 numerical plan：新增 DFT full spectrum 复数输入、DFT inverse onesided、STFT full spectrum window、STFT full spectrum 复数输入且无 window 分支。
- `tools/numerical/runner.py` 为 DFT/STFT 新增 `dft_variant` 和 `stft_variant` 内部测试参数，仅用于选择固定谱样本；构造算子前会移除这些参数，公共算子接口和运行逻辑不变。
- STFT numerical runner 现在显式保留可选 `window=None` 的参数位置，避免过滤 None 后把 `frame_length` 误传为 window。
- 本轮后默认 numerical plan 保持 `178` 个唯一算子名称，默认计划提升到 `644` 条，混合精度计划提升到 `439` 条。
- 继续扩展循环算子默认 numerical plan：新增 RNN reverse、RNN bidirectional+layout=1、GRU `linear_before_reset=0`、GRU reverse+layout=1、LSTM `input_forget=0`、LSTM bidirectional+layout=1 分支，并补入 float16/bfloat16 低精度路径。
- `tools/numerical/runner.py` 的 RNN/GRU/LSTM 固定输入现在按 plan 的 `direction`、`layout`、`hidden_size` 动态生成，避免新增 plan 仍实际复用旧 forward/layout=0 样本。
- RNN/GRU/LSTM CUDA verifier 现在会写出最终状态 sidecar；numerical runner 已同时比较 RNN/GRU 的 `Y/Y_h` 和 LSTM 的 `Y/Y_h/Y_c`。
- 本轮后默认 numerical plan 保持 `178` 个唯一算子名称，默认计划提升到 `656` 条，混合精度计划提升到 `445` 条。
- 继续扩展循环算子 activation/clip 默认 numerical plan：新增 RNN `Relu + clip`、GRU `HardSigmoid/ScaledTanh + activation_alpha/beta + clip`、LSTM `HardSigmoid/Tanh/Relu + activation_alpha/beta + clip` 分支，并同步补入 float16/bfloat16 低精度路径。
- `tools/numerical/runner.py` 现在会把 recurrent activation code、activation alpha/beta 和 clip 编码到 CUDA 参数块；RNN/GRU/LSTM CUDA verifier 使用同一参数块解码并执行 ONNX recurrent activation 语义。
- 本轮后默认 numerical plan 保持 `178` 个唯一算子名称，默认计划提升到 `662` 条，混合精度计划提升到 `448` 条。
- 继续扩展 Attention 默认 numerical plan：新增 float mask、bool broadcast mask、显式 scale、非 causal、无 softcap，以及 float16/bfloat16 低精度写回路径。
- `tools/numerical/runner.py` 为 Attention 新增固定 mask 样本生成，确保默认门禁覆盖 CUDA verifier 已支持的 mask 广播和布尔屏蔽语义。
- 本轮后默认 numerical plan 保持 `178` 个唯一算子名称，默认计划提升到 `666` 条，混合精度计划提升到 `450` 条。
- 继续扩展 DeformConv 默认 numerical plan：新增 `group=2/offset_group=2` 分组采样计划，以及无 bias/无 mask、非默认 stride/pad/dilation 计划，并同步补入 float16/bfloat16 低精度路径。
- `tools/numerical/runner.py` 的 DeformConv 固定样本生成现在会按可选输入是否存在分别生成 bias 和 mask，CUDA 参数块也显式编码 `has_bias/has_mask`。
- 本轮后默认 numerical plan 保持 `178` 个唯一算子名称，默认计划提升到 `670` 条，混合精度计划提升到 `452` 条。
- 继续扩展 QuantizeLinear/DequantizeLinear 默认 numerical plan：新增 `axis=1` per-axis scale/zero_point 的 uint8 量化和反量化计划，并同步补入 float16/bfloat16 低精度路径。
- `cuda/verify_quantize_linear.cu` 与 `cuda/verify_dequantize_linear.cu` 现在可读取原始 1D scale/zero_point，并按输出坐标映射到对应 axis 参数下标，避免把 per-axis 参数误当成线性逐元素数组。
- 继续扩展 QuantizeLinear/DequantizeLinear 默认 numerical plan：新增 `axis=-1` signed int8 per-axis 且省略 `zero_point` 的量化和反量化计划，覆盖 ONNX 可选 `zero_point` 缺省时默认零点语义，并同步补入 float16/bfloat16 低精度路径。
- `tools/numerical/runner.py` 在该内部测试参数下仍为 CUDA reference 生成显式零点张量，但调用 Python/C runtime 时只传入 `x` 和 `scale`，从而真实验证算子可选输入缺省路径。
- 本轮后默认 numerical plan 保持 `178` 个唯一算子名称，默认计划提升到 `678` 条，混合精度计划提升到 `456` 条。
- 继续补强 QuantizeLinear/DequantizeLinear 最新 schema 的 `output_dtype` 属性：导入器在缺省 zero point 时优先使用 `output_dtype` 决定 QuantizeLinear 输出 dtype，DequantizeLinear 会保存并使用该属性选择输出 dtype。
- 新增 pytest 覆盖无中间 `value_info` 时的 QuantizeLinear `output_dtype=int8` 导入，以及缺省 `zero_point` 时 QuantizeLinear int8 输出和 DequantizeLinear float16 输出的主路径。
- 继续补强 QuantizeLinear/DequantizeLinear 最新 schema 的 `block_size` 属性：导入器读取 `block_size`，Python runtime 将 blocked scale/zero_point 按输入形状展开后交给现有 C 后端完成逐元素量化/反量化，CUDA verifier 独立按 blocked 坐标映射读取原始 scale/zero_point。
- 默认 numerical 新增 float32 QuantizeLinear blocked、float32 DequantizeLinear blocked、float16 QuantizeLinear blocked 和 bfloat16 DequantizeLinear blocked 四条计划；本轮后默认计划提升到 `682` 条，混合精度计划提升到 `458` 条。
- 继续补强 QuantizeLinear/DequantizeLinear 官方 dtype 约束：CUDA verifier 的量化饱和范围从 int8/uint8 扩展到 int16/uint16，默认 numerical 新增 uint16/int16 量化和反量化计划；本轮后默认计划提升到 `686` 条，混合精度计划提升到 `460` 条。
- 继续补强 QuantizeLinear 最新 schema 的 `precision` 属性：C 后端新增 `quantize_linear_forward_precision` 入口，显式 `precision=DOUBLE` 时使用 double division；默认 numerical 新增 float32 scale 下 double precision 舍入边界计划，本轮后默认计划提升到 `687` 条，混合精度计划保持 `460` 条。
- 继续扩大 QuantizeLinear/DequantizeLinear 的 blocked quantization 边界矩阵：新增 `axis=-1`、输入 `(2, 3, 5)`、scale/zero_point `(2, 3, 3)`、`block_size=2` 的尾块不满映射 case，并同步补入 float16 QuantizeLinear 与 bfloat16 DequantizeLinear 混合精度路径；本轮后默认计划提升到 `691` 条，混合精度计划提升到 `462` 条。
- 继续补强 DequantizeLinear 官方 dtype 约束中的 int32 输入路径：默认 numerical 新增 int32 输入、int32 zero point、float32 scale 的极值反量化计划，runner 现在对整数 `input_values` 直接按目标整数 dtype 构造输入，避免 int32 极值先经过 float32 时丢精度；本轮后默认计划提升到 `692` 条，混合精度计划保持 `462` 条。
- 继续补强循环算子 `sequence_lens` 边界：默认 numerical 为 RNN、GRU、LSTM 分别新增 float32 和低精度的 `sequence_lens=[0, 2]` 计划，固定验证零长度 batch 不执行任何时间步，并保持 `initial_h/initial_c` 状态；本轮后默认计划提升到 `698` 条，混合精度计划提升到 `465` 条。
- 继续补强谱算子 axis 和高维前缀边界：`cuda/verify_dft.cu` 从扁平 `batch x length x complex` 参数升级为 rank/shape/axis 坐标映射协议，默认 numerical 新增 DFT 4D 中间轴 `axis=1`、`dft_length=5` 的 float32/float16 计划，并新增 STFT 4D 多前缀维输入的 float32/bfloat16 计划；本轮后默认计划提升到 `702` 条，混合精度计划提升到 `467` 条。
- 继续补强 QuantizeLinear 最新 schema 的 `saturate` 属性：C 后端新增 `quantize_linear_forward_precision_saturate` 入口，float8 输出按 `saturate=1/0` 区分最大有限值、Inf 和 NaN 溢出行为；CUDA verifier 同步接收 saturate 参数并对 float8_e4m3/float8_e5m2 使用独立溢出阈值。默认 numerical 新增 3 条 float8 saturate 计划；本轮后默认计划提升到 `705` 条，混合精度计划提升到 `470` 条。
- 继续补强 float8 FNUZ dtype 语义：Python dtype 映射、C 后端读写、CUDA QuantizeLinear reference、numerical dtype 编解码和 pytest 均新增 float8_e4m3fnuz/float8_e5m2fnuz；QuantizeLinear 覆盖 `saturate=1/0`、最大有限值、FNUZ NaN 位模式，DequantizeLinear 覆盖 raw uint8 位模式解码。默认 numerical 新增 6 条 FNUZ 计划；本轮后默认计划提升到 `711` 条，混合精度计划提升到 `476` 条。
- 继续补强低 bit 整数量化 dtype 语义：Python dtype 映射、C 后端读写、CUDA QuantizeLinear reference、numerical dtype 编解码和 pytest 均新增 `uint4`、`int4`、`uint2`、`int2`；当前按项目内部一元素一字节容器执行值语义，QuantizeLinear/DequantizeLinear 覆盖饱和边界和反量化解码。默认 numerical 新增 8 条低 bit 整数计划；本轮后默认计划提升到 `719` 条，混合精度计划提升到 `484` 条。

## 本轮已运行验证

- `python -m py_compile tools/numerical/cli.py tools/numerical/runner.py`
- `git diff --check`
- `python tools/cli.py compile-cuda`
  - 结果：`178` 个 CUDA verifier 编译成功；`verify_dft` 已按新的 rank/shape/axis 参数协议完成全量编译。
- `python tools/cli.py numerical --op dft --op stft --iterations 3 --skip-plots`
  - 结果：全部通过，DFT/STFT 各 `9` 个 active plan 样本均与 CUDA reference 对齐，覆盖 DFT 高维中间轴和 STFT 多前缀维输入。
- `python -m pytest -q tests/test_operator_spectral_semantics.py`
  - 结果：`4 passed`，新增独立公式验证 DFT 高维中间轴和 STFT 多前缀维切帧。
- `python tools/cli.py numerical --op rnn --op gru --op lstm --iterations 3 --skip-plots`
  - 结果：全部通过，RNN/GRU/LSTM 各 `10` 组 active plan、各 `30` 个样本均与 CUDA reference 对齐，且覆盖 `sequence_lens=[0, 2]` 零长度 batch、非默认 activation、activation alpha/beta、clip、RNN/GRU `Y_h` 和 LSTM `Y_h/Y_c` sidecar 输出。
- `python -m pytest -q tests/test_operator_recurrent_semantics.py`
  - 结果：`3 passed`，新增独立公式验证 `sequence_lens=0` 时 RNN/GRU/LSTM 的输出和最终状态保持 initial state。
- `python tools/cli.py numerical --op bitwise_and --op bitwise_or --op bitwise_xor --op bitwise_not --op bit_shift --iterations 3 --skip-plots`
  - 结果：全部通过，误差为 `0`。
- `python tools/cli.py numerical --op tril --op triu --op trilu --op hann_window --op hamming_window --op blackman_window --iterations 3 --skip-plots`
  - 结果：全部通过，误差为 `0`。
- `python tools/cli.py numerical --op range --op one_hot --op reverse_sequence --op det --op mel_weight_matrix --iterations 3 --skip-plots`
  - 结果：全部通过。
- `python tools/cli.py numerical --op dynamic_quantize_linear --iterations 3 --skip-plots`
  - 结果：全部通过。
- `python tools/cli.py numerical --op split --iterations 3 --skip-plots`
  - 结果：全部通过。
- `python tools/cli.py numerical --op unique --iterations 3 --skip-plots`
  - 结果：全部通过。
- `python tools/cli.py numerical --op dropout --iterations 3 --skip-plots`
  - 结果：全部通过，`y` 与 `mask` 均对齐 CUDA reference。
- `python tools/cli.py numerical --op random_uniform --op random_uniform_like --op random_normal --op random_normal_like --op bernoulli --iterations 3 --skip-plots`
  - 结果：全部通过。
- `python tools/cli.py numerical --op multinomial --iterations 3 --skip-plots`
  - 结果：全部通过，int64 与 int32 输出均对齐 CUDA reference。
- `python tools/cli.py numerical --op binarizer --op negative_log_likelihood_loss --op softmax_cross_entropy_loss --op non_max_suppression --iterations 3 --skip-plots`
  - 结果：全部通过，Binarizer 的五条计划、NLLLoss 的三条计划、SCE Loss 的三条计划和 NMS 的四条计划均对齐 CUDA reference。
- `python tools/cli.py numerical --op lp_normalization --iterations 3 --skip-plots`
  - 结果：全部通过，6 条 LpNormalization 默认计划均对齐 CUDA reference。
- `python tools/cli.py numerical --op layer_normalization --iterations 3 --skip-plots`
  - 结果：全部通过，8 条 LayerNormalization 默认计划均对齐 CUDA reference，包含 `Y/mean/inv_std` 三路输出。
- `python tools/cli.py numerical --op grid_sample --iterations 3 --skip-plots`
  - 结果：全部通过，7 条 GridSample 默认计划、21 个样本均对齐 CUDA reference。
- `python tools/cli.py numerical --op attention --iterations 3 --skip-plots`
  - 结果：全部通过，7 条 Attention 默认计划、21 个样本均对齐 CUDA reference，覆盖 float mask、bool broadcast mask、显式 scale、causal/非 causal、softcap/无 softcap 和低精度写回。
- `python tools/cli.py numerical --op max_roi_pool --op roi_align --iterations 3 --skip-plots`
  - 结果：全部通过，MaxRoiPool 与 RoiAlign 各 15 个样本均对齐 CUDA reference。
- `python -m pytest -q tests/test_operator_roi_semantics.py`
  - 结果：`3 passed`。
- `python -m pytest -q tests/test_operator_c_backend.py::test_c_backend_grid_sample_matches_onnx_reference tests/test_operator_complex_attribute_semantics.py::test_c_backend_grid_sample_modes_match_onnx_reference`
  - 结果：`2 passed`。
- `python tools/cli.py numerical --op batch_normalization --iterations 3 --skip-plots`
  - 结果：全部通过，推理态与 training_mode 三输出计划均对齐 CUDA reference。
- `python -m pytest -q tests/test_operator_normalization_semantics.py`
  - 结果：`16 passed`，包含 BatchNormalization training_mode 和 LayerNormalization aux 输出的 C 后端路径断言。
- `python -m pytest -q tests/test_operator_misc_semantics.py tests/test_operator_c_backend.py -k "bitwise or bit_shift or unsigned_integer_binary_ops"`
  - 结果：`2 passed, 56 deselected`。
- `python tools/audit_ops.py --output docs/reports/operator_coverage.md`
  - 结果：覆盖报告已刷新。
- `python tools/cli.py numerical --op quantize_linear --op dequantize_linear --iterations 3 --skip-plots`
  - 结果：全部通过，QuantizeLinear/DequantizeLinear 共 14 组计划、42 个样本均与 CUDA reference 对齐，覆盖 scalar scale/zero_point、`axis=1` per-axis uint8、`axis=-1` per-axis signed int8、省略 `zero_point` 默认零点、float16 和 bfloat16 低精度路径。
- `python -m pytest -q tests/test_operator_import_and_shape.py::test_quantize_linear_forward_shape_and_optional_zero_point_import tests/test_operator_core_numeric_semantics.py::test_c_backend_quantize_dequantize_output_dtype_without_zero_point_matches_onnx_reference tests/test_operator_core_numeric_semantics.py::test_c_backend_quantize_and_dequantize_negative_axis_match_onnx_reference`
  - 结果：`3 passed`，覆盖 `output_dtype` 属性导入、缺省 `zero_point` 的 QuantizeLinear int8 输出和 DequantizeLinear float16 输出。
- `python -m pytest -q tests/test_operator_import_and_shape.py::test_quantize_linear_forward_shape_and_optional_zero_point_import tests/test_operator_core_numeric_semantics.py::test_c_backend_quantize_and_dequantize_block_size_match_onnx_reference tests/test_operator_core_numeric_semantics.py::test_c_backend_quantize_dequantize_output_dtype_without_zero_point_matches_onnx_reference`
  - 结果：`3 passed`，覆盖 `block_size=2` 属性导入、输入 `(2, 3, 4)`、scale/zero_point `(2, 2, 4)`、`axis=1` 的 blocked QuantizeLinear/DequantizeLinear 官方参考语义，以及缺省 `zero_point` 的 output_dtype 主路径。
- `python tools/cli.py numerical --op quantize_linear --op dequantize_linear --iterations 3 --skip-plots`
  - 结果：全部通过，QuantizeLinear/DequantizeLinear 共 `18` 组计划、`54` 个样本均与 CUDA reference 对齐，覆盖 scalar、per-axis、负轴、省略 zero_point、output_dtype 相关主路径和 `block_size=2` blocked scale/zero_point 映射。
- `python -m pytest -q tests/test_operator_core_numeric_semantics.py::test_c_backend_quantize_and_dequantize_16bit_integer_dtypes_match_onnx_reference`
  - 结果：`1 passed`，覆盖 QuantizeLinear int16/uint16 饱和量化和 DequantizeLinear int16/uint16 输入的 ONNX reference 对齐。
- `python tools/cli.py numerical --op quantize_linear --op dequantize_linear --iterations 3 --skip-plots`
  - 结果：全部通过，QuantizeLinear/DequantizeLinear 共 `22` 组计划、`66` 个样本均与 CUDA reference 对齐，新增 float32 uint16、float16 int16、float32 int16 反量化和 bfloat16 uint16 反量化路径。
- `python -m pytest -q tests/test_operator_import_and_shape.py::test_quantize_linear_forward_shape_and_optional_zero_point_import tests/test_operator_core_numeric_semantics.py::test_c_backend_quantize_linear_precision_double_uses_double_division`
  - 结果：`2 passed`，覆盖 QuantizeLinear `precision=DOUBLE` 和 `saturate` 属性导入保留，以及 `precision=DOUBLE` 真实切换 double division 的舍入边界。
- `python tools/cli.py numerical --op quantize_linear --op dequantize_linear --iterations 3 --skip-plots`
  - 结果：全部通过，QuantizeLinear/DequantizeLinear 共 `23` 组计划、`69` 个样本均与 CUDA reference 对齐，新增 `precision=DOUBLE` 独立 C/CUDA 数值门禁。
- `python -m pytest -q tests/test_operator_core_numeric_semantics.py::test_c_backend_quantize_and_dequantize_negative_axis_block_tail_matches_onnx_reference`
  - 结果：`1 passed`，覆盖 `axis=-1`、`block_size=2` 且最后一维长度不能被块大小整除时，QuantizeLinear/DequantizeLinear 的 blocked scale/zero_point 坐标映射官方语义。
- `python -m pytest -q tests/test_operator_core_numeric_semantics.py::test_c_backend_dequantize_int32_dtype_matches_onnx_reference`
  - 结果：`1 passed`，覆盖 DequantizeLinear int32 输入和 int32 zero point 的官方 dtype 约束，以及 int32 最小值、最大值和大整数反量化边界。
- `python tools/cli.py numerical --op quantize_linear --op dequantize_linear --iterations 3 --skip-plots`
  - 结果：全部通过，QuantizeLinear/DequantizeLinear 共 `28` 组计划、`84` 个样本均与 CUDA reference 对齐，新增 DequantizeLinear int32 输入 dtype 路径，并保留负轴尾块不满 blocked 映射和对应低精度写回路径。
- `python -m pytest -q tests/test_operator_core_numeric_semantics.py -k 'quantize_linear_float8_saturate or precision_double'`
  - 结果：`4 passed, 17 deselected`，覆盖 QuantizeLinear float8_e5m2 `saturate=1/0` 和 float8_e4m3 `saturate=0` 输出位模式与 ONNX ReferenceEvaluator 对齐。
- `python tools/cli.py compile-cuda`
  - 结果：`178` 个 CUDA verifier 编译成功，包含新增 saturate 参数协议后的 `verify_quantize_linear`。
- `python tools/cli.py numerical --op quantize_linear --iterations 3 --skip-plots`
  - 结果：全部通过，QuantizeLinear 共 `17` 组计划、`51` 个样本均与 CUDA reference 对齐，新增 float8_e5m2 `saturate=1/0` 和 float8_e4m3 `saturate=0` 溢出边界计划。
- `python tools/cli.py numerical --iterations 1 --skip-plots`
  - 结果：本轮新增 QuantizeLinear/DequantizeLinear FNUZ 计划后默认计划提升到 `711` 条；完整 numerical 一轮已全部通过，覆盖 `178` 个唯一算子和 `476` 条混合精度计划。
- `/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_operator_core_numeric_semantics.py -k "low_bit or quantize or dequantize"`
  - 结果：`16 passed, 11 deselected`；新增 `uint4`、`int4`、`uint2`、`int2` 的 ONNX ReferenceEvaluator 对齐测试。
- `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --op quantize_linear --op dequantize_linear --iterations 3 --skip-plots`
  - 结果：QuantizeLinear/DequantizeLinear targeted numerical 共 `45` 组计划、`135` 个样本全部通过；新增低 bit 整数 C 后端值语义与 CUDA reference 对齐。
- `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda`
  - 结果：`178` 个 CUDA verifier 全部编译成功。
- `/home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests`
  - 结果：`316 passed, 1 skipped`。
- `make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python check`
  - 结果：静态 Python 编译检查通过。
- `/home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py numerical --iterations 1 --skip-plots`
  - 结果：默认 active numerical plan 的 `719` 条计划全部通过，覆盖 `178` 个唯一算子和 `484` 条混合精度计划。
- `python -m pytest -q tests`
  - 结果：`315 passed, 1 skipped`。
- `make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python check`
  - 结果：静态 Python 编译检查通过。

## 仍未完成的普通 C-backed 数值门禁

当前未发现仍需立即后端化或接入默认 numerical 的普通 C-backed 数值/张量算子。后续剩余工作转为扩大 case matrix：对已接入门禁的复杂算子继续补全属性组合、dtype 组合和边界输入，而不是回退到 Python 数值实现。

## 合理保留 Python 调度的范围

以下类别主要是控制流、序列、可选值、字符串、图像 IO 或元数据算子，不适合按普通 C/CUDA 数值门禁处理；后续应继续依赖 ONNX reference pytest、导入器语义测试和端到端图测试：

- 控制流：`If`、`Loop`、`Scan`。
- 序列：`SequenceEmpty`、`SequenceConstruct`、`SequenceAt`、`SequenceInsert`、`SequenceErase`、`SequenceLength`、`SequenceMap`、`ConcatFromSequence`、`SplitToSequence`。
- 可选值：`Optional`、`OptionalGetElement`、`OptionalHasElement`。
- 字符串：`RegexFullMatch`、`StringConcat`、`StringSplit`、`StringNormalizer`、`TfIdfVectorizer`。
- 元数据/常量/图像：`Shape`、`Constant`、`ImageDecoder`。

## 后续建议优先级

1. 继续扩大已进入默认门禁的复杂算子 case matrix，尤其是损失/NMS、ROI、序列、谱、Attention、DeformConv、量化和随机采样类边界。
2. 审计所有“含 Python 调度或 fallback”的算子，确认普通数值主路径都由 C 后端承载，Python 只保留导入、shape、调度、ONNX reference 对照或 C 不承载 dtype 的兜底。

## 当前剩余风险

- 默认 numerical 是固定 case 与随机样本的门禁，不等同于 ONNX 所有 opset schema 的穷尽证明。
- 已进入 numerical 的 mixed precision 计划主要覆盖项目当前支持和官方 type constraint 中合理的低精度路径；非官方约束内的 float8 或字符串/序列路径不应强行纳入数值门禁。
- `BatchNormalization` 已覆盖推理态和 training_mode 三输出主路径；更多 rank、极小方差、不同 momentum/epsilon、空维度和异常 shape 组合仍建议继续扩展。
- `LayerNormalization` 已覆盖单输出和 `mean/inv_std` aux 多输出 C/CUDA 主路径；更多 rank、stash_type、极小方差、空维度和异常 axis 组合仍建议继续扩展。
- `QuantizeLinear`/`DequantizeLinear` 已覆盖 scalar、`axis=1` uint8 per-axis scale/zero_point、`axis=-1` signed int8 per-axis、省略 `zero_point` 默认零点、`output_dtype` 属性、`block_size=2` 正轴和负轴尾块不满 blocked scale/zero_point、int16/uint16 量化 dtype、DequantizeLinear int32 输入 dtype、`precision=DOUBLE` 除法精度、QuantizeLinear float8_e4m3/float8_e5m2 `saturate=1/0` 溢出边界、float8_e4m3fnuz/float8_e5m2fnuz 的 FNUZ NaN/饱和/反量化解码，以及 `uint4`、`int4`、`uint2`、`int2` 的低 bit 整数值语义 C/CUDA numerical 与 pytest；更多 precision dtype、更多 block 形状/轴组合、官方 packed TensorProto 存储，以及 FLOAT4E2M1/FLOAT8E8M0 仍需继续扩展。
- `GridSample` 已覆盖 linear/reflection、nearest/border 与 cubic/zeros 的 C/CUDA numerical 路径；更多 5D 输入、极端越界坐标、边界点插值和 align_corners 组合仍建议继续扩展。
- `MaxRoiPool` 已覆盖默认 ROI、spatial_scale=0.5、越界裁剪、空 ROI 输出和 bfloat16 写回；`RoiAlign` 已覆盖 avg/half_pixel、max/output_half_pixel、自适应采样和 float16 写回。更多 ROI 数量、边界点采样、异常 batch index 和不同输出尺寸仍建议继续补充。
- `DFT`/`STFT` 已补充 full spectrum、复数输入、inverse onesided、STFT 无 window、低精度分支、DFT 4D 中间轴和 STFT 多前缀维输入；后续仍建议继续扩展更多轴组合、不同长度、异常输入和边界 window 参数。`RNN`、`GRU`、`LSTM` 已补充 reverse、bidirectional、layout=1、GRU `linear_before_reset=0/1`、LSTM `input_forget=0/1`、非默认 activation、activation alpha/beta、clip、`sequence_lens=0` 保持初始状态边界，并补齐 RNN/GRU `Y_h` 与 LSTM `Y_h/Y_c` 的 C/CUDA sidecar 对比；后续仍建议继续扩展更多 activation 组合、不同 sequence_lens 分布、更多 initial state 极端值和极端状态边界。
- `Attention` 已从基础 4D GQA causal/softcap 主路径扩展到 float mask、bool broadcast mask、显式 scale、非 causal、无 softcap 和 float16/bfloat16 低精度 C/CUDA numerical；cache、nonpad、3D 输入和 qk 中间输出仍主要由 ONNX reference pytest 覆盖。`DeformConv` 已补充分组、offset group、无 bias/无 mask、非默认 stride/pad/dilation 和低精度 C/CUDA numerical；更高维、更多 dilation/pad 边界和特殊 fallback 仍主要由 ONNX reference pytest 覆盖。
