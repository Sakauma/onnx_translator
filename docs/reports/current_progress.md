<!--
/**
  ******************************************************************************
  * @file        current_progress.md
  * @author      Egor Izmaylov
  * @brief       记录当前工程进度、未完成事项和后续风险。
  * @details     2026.06.13  V1.0.0  创建
  * @details     2026.06.13  V1.0.1  补充 GridSample 属性数值覆盖记录
  * @details     2026.06.13  V1.0.2  补充 ROI 算子边界属性数值覆盖记录
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
- CUDA verifier：`178` 个。
- 默认 active numerical plan：`178` 个唯一算子名称，`656` 条默认计划。
- 默认 active numerical plan 混合精度覆盖：`445` 条计划。

## 最近已完成验证

- `python -m py_compile tools/numerical/cli.py tools/numerical/runner.py`
- `git diff --check`
- `python tools/cli.py compile-cuda`
  - 最近记录结果：`178` 个 CUDA verifier 编译成功。
- `python tools/cli.py numerical --op dft --op stft --iterations 3 --skip-plots`
  - 最近记录结果：DFT/STFT 新增 full spectrum、复数输入、inverse onesided、STFT 无 window 分支均通过。
- `python -m pytest -q tests/test_operator_spectral_semantics.py`
  - 最近记录结果：`2 passed`。
- `python tools/cli.py numerical --op rnn --op gru --op lstm --iterations 3 --skip-plots`
  - 最近记录结果：RNN/GRU/LSTM 新增 reverse、bidirectional、layout=1、GRU `linear_before_reset=0/1` 和 LSTM `input_forget=0/1` 分支均通过；当前 targeted numerical 已同时比较 RNN/GRU 的 `Y/Y_h` 和 LSTM 的 `Y/Y_h/Y_c`。
- `python -m pytest -q tests/test_operator_recurrent_semantics.py`
  - 最近记录结果：`2 passed`。
- `python tools/cli.py numerical --op bitwise_and --op bitwise_or --op bitwise_xor --op bitwise_not --op bit_shift --iterations 3 --skip-plots`
  - 最近记录结果：位运算 targeted numerical 通过。
- `python tools/cli.py numerical --op tril --op triu --op trilu --op hann_window --op hamming_window --op blackman_window --iterations 3 --skip-plots`
  - 最近记录结果：三角矩阵和窗函数 targeted numerical 通过。
- `python tools/cli.py numerical --op range --op one_hot --op reverse_sequence --op det --op mel_weight_matrix --iterations 3 --skip-plots`
  - 最近记录结果：形状/索引、线性代数和特征矩阵 targeted numerical 通过。
- `python tools/cli.py numerical --op dynamic_quantize_linear --iterations 3 --skip-plots`
  - 最近记录结果：DynamicQuantizeLinear 三输出 targeted numerical 通过。
- `python tools/cli.py numerical --op split --iterations 3 --skip-plots`
  - 最近记录结果：Split 多输出 targeted numerical 通过。
- `python tools/cli.py numerical --op unique --iterations 3 --skip-plots`
  - 最近记录结果：Unique 四输出 targeted numerical 通过。
- `python tools/cli.py numerical --op dropout --iterations 3 --skip-plots`
  - 最近记录结果：Dropout 的 `y` 和 `mask` 双输出 targeted numerical 通过。
- `python tools/cli.py numerical --op random_uniform --op random_uniform_like --op random_normal --op random_normal_like --op bernoulli --iterations 3 --skip-plots`
  - 最近记录结果：随机 uniform/normal 和 Bernoulli 的 C-vs-CUDA targeted numerical 通过。
- `python tools/cli.py numerical --op multinomial --iterations 3 --skip-plots`
  - 最近记录结果：Multinomial 的 int64/int32 输出、零概率和非归一化概率 targeted numerical 通过。
- `python tools/cli.py numerical --op binarizer --op negative_log_likelihood_loss --op softmax_cross_entropy_loss --op non_max_suppression --iterations 3 --skip-plots`
  - 最近记录结果：Binarizer、NegativeLogLikelihoodLoss、SoftmaxCrossEntropyLoss 和 NonMaxSuppression targeted numerical 通过；NonMaxSuppression 当前包含多 batch/class、排序 tie、阈值等值包含和 float16 空输出边界计划。
- `python tools/cli.py numerical --op lp_normalization --iterations 3 --skip-plots`
  - 最近记录结果：LpNormalization 的 p=1/p=2、axis=1/axis=2/axis=-1、bfloat16 和零范数边界 targeted numerical 通过。
- `python tools/cli.py numerical --op layer_normalization --iterations 3 --skip-plots`
  - 最近记录结果：LayerNormalization 的 axis=-1 与 axis=1 后缀归一化、float32/float16/bfloat16 单输出和 `mean/inv_std` aux 多输出 C/CUDA targeted numerical 通过。
- `python tools/cli.py numerical --op grid_sample --iterations 3 --skip-plots`
  - 最近记录结果：GridSample 的 linear/reflection、nearest/border、cubic/zeros 属性组合，以及 float32、float16、bfloat16 C/CUDA targeted numerical 通过，共 7 条计划、21 个样本。
- `python tools/cli.py numerical --op max_roi_pool --op roi_align --iterations 3 --skip-plots`
  - 最近记录结果：MaxRoiPool 的 spatial_scale=0.5、越界裁剪、空 ROI 输出，以及 RoiAlign 的 max/output_half_pixel/自适应采样属性组合 targeted numerical 通过；两类算子各 15 个样本。
- `python tools/cli.py numerical --op batch_normalization --iterations 3 --skip-plots`
  - 最近记录结果：BatchNormalization 的推理态和 training_mode 三输出路径均通过；训练态 `Y/running_mean/running_var` 已由 C 后端计算，并与 CUDA sidecar reference 对齐，覆盖 float32、float16、bfloat16。
- `python -m pytest -q tests/test_operator_normalization_semantics.py`
  - 最近记录结果：`16 passed`，包含 BatchNormalization training_mode 和 LayerNormalization aux 输出的 C 后端路径断言。
- `python -m pytest -q tests/test_operator_misc_semantics.py tests/test_operator_c_backend.py -k "bitwise or bit_shift or unsigned_integer_binary_ops"`
  - 最近记录结果：相关 pytest 通过。
- `python tools/audit_ops.py --output docs/reports/operator_coverage.md`
  - 最近记录结果：覆盖报告已刷新。
- `python tools/cli.py numerical --iterations 1 --skip-plots`
  - 最近记录结果：`656` 条默认计划完整 numerical 一轮通过，当前 recurrent 计划包含状态输出 sidecar 对比。
- `python -m pytest -q tests`
  - 最近记录结果：`298 passed, 1 skipped`。
- `make PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python check`
  - 最近记录结果：静态 Python 编译检查通过。

## 已知未完成部分

当前未发现仍需立即后端化或接入默认 numerical 的普通 C-backed 数值/张量算子。本轮已将 `BatchNormalization` 的 `training_mode` 多输出语义和 `LayerNormalization` 的 `mean/inv_std` aux 多输出语义从 Python 数值路径下沉到 C 后端，并接入独立 CUDA verifier 与默认 numerical 门禁。

后续未完成重点不再是“是否有 C/CUDA 门禁”，而是继续扩展官方语义的全属性、全 dtype 和全边界 case matrix。补强时仍应优先使用 C/CUDA reference，不应回退为普通数值路径的 Python-only 实现。

以下类别属于合理保留 Python 调度的范围，不适合按普通 C/CUDA 数值门禁处理；后续重点应放在 ONNX reference pytest、导入器语义测试和端到端图测试。

- 控制流：`If`、`Loop`、`Scan`。
- 序列：`SequenceEmpty`、`SequenceConstruct`、`SequenceAt`、`SequenceInsert`、`SequenceErase`、`SequenceLength`、`SequenceMap`、`ConcatFromSequence`、`SplitToSequence`。
- 可选值：`Optional`、`OptionalGetElement`、`OptionalHasElement`。
- 字符串：`RegexFullMatch`、`StringConcat`、`StringSplit`、`StringNormalizer`、`TfIdfVectorizer`。
- 元数据/常量/图像：`Shape`、`Constant`、`ImageDecoder`。

## 混合精度状态

- 当前默认 numerical 中已经包含 `445` 条混合精度计划，覆盖 float16、bfloat16、部分 float8 以及相关低精度存储路径。
- 混合精度已经能作为当前工程的常规回归门禁使用，但还不能宣称对所有 ONNX 官方 type constraint、所有属性组合和所有边界输入完成穷尽证明。
- 位运算、字符串、序列、控制流、随机采样等类别不应机械纳入浮点混合精度口径；这些算子需要按整数位模式、结构语义、随机分布或 ONNX reference 行为分别验证。

## 剩余风险

- ONNX opset 17 已达到名称级导入覆盖，但名称级覆盖不等同于所有属性、边界条件、异常路径和高维组合的官方语义穷尽验证。
- 默认 numerical 是固定 case 与随机样本组成的工程门禁，能够发现常见回归，但不是形式化证明。
- `LayerNormalization` 已将单输出 C 后端从最后一维扩展到任意 axis 后缀归一化，并补充 `mean/inv_std` aux 多输出 C/CUDA 门禁；后续仍建议继续扩展更多 rank、stash_type、极小方差、空维度和异常 axis 组合。
- `BatchNormalization` 已将推理态和 training_mode 三输出主路径都接入 C/CUDA numerical；后续仍建议继续补充更多 rank、极小方差、空维度和不同 momentum/epsilon 组合。
- `LpNormalization` 已补充零范数官方边界和非通道 axis 的 bfloat16 C/CUDA 门禁；更多空维度、不同 rank 和异常 axis 仍建议继续扩展。
- `GridSample` 已将 numerical 从 linear/reflection 主路径扩展到 nearest/border 与 cubic/zeros 属性组合，并覆盖 float32、float16、bfloat16；后续仍建议继续补充 5D、更多坐标边界、极端越界坐标和更多 align_corners 组合。
- `MaxRoiPool` 已补充 spatial_scale=0.5、越界裁剪、空 ROI 输出和 bfloat16 低精度 C/CUDA numerical；`RoiAlign` 已补充 max 模式、output_half_pixel、自适应 sampling_ratio=0、spatial_scale=0.75 和 float16 低精度 C/CUDA numerical。后续仍建议继续扩展更多 ROI 数量、不同 pooled/output 尺寸、边界点采样和异常 batch index。
- `DFT`/`STFT` 已从基础 onesided 实数样本扩展到 full spectrum、复数输入、inverse onesided、STFT 无 window 和 float16/bfloat16 分支；后续仍建议继续扩展更多 axis、高 rank、不同长度和异常输入。`RNN`、`GRU`、`LSTM` 已从基础 forward/layout=0 样本扩展到 reverse、bidirectional、layout=1、GRU `linear_before_reset=0/1`、LSTM `input_forget=0/1`，并补齐 RNN/GRU `Y_h` 与 LSTM `Y_h/Y_c` 的 C/CUDA sidecar 对比；后续仍建议继续扩展非默认 activation、clip、更多 sequence_lens/initial state。
- `Attention`、`DeformConv` 等复杂算子已覆盖主 C/CUDA 路径，部分 cache、nonpad、多输出或高维 fallback 仍主要依赖 ONNX reference pytest。
- 随机、损失、NMS、量化和集合类算子继续扩展 CUDA reference case 时，需要特别注意确定性种子、阈值、排序稳定性和低精度舍入规则。

## 建议后续优先级

1. 继续扩展已进入默认门禁的复杂算子属性矩阵，尤其是 ROI、序列、谱、Attention、DeformConv、量化、损失和 NMS 类边界。
2. 对仍带 Python fallback 的复杂路径逐项审计，只保留调度、ONNX reference 对照或 C 后端暂不承载 dtype 的兜底，不允许普通数值主路径退回 Python-only。
