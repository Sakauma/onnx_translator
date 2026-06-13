<!--
/**
  ******************************************************************************
  * @file        session_closeout_2026_06_13.md
  * @author      Egor Izmaylov
  * @brief       记录 2026-06-13 当前工程进度、未完成事项和收尾验证结果。
  * @details     2026.06.13  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
-->

# 2026-06-13 工程进度与未完成事项

> 记录时间：2026-06-13
> 当前分支：`main`
> 依据：`tools/audit_ops.py`、`docs/reports/operator_coverage.md`、当前工作树改动和本轮 numerical/CUDA 验证记录。

## 当前总体进度

- ONNX opset 17 默认 domain 名称级覆盖：`178/178`。
- 当前安装 ONNX 最新默认 domain schema 名称级覆盖：`200/200`。
- Python 算子类：`201` 个。
- ONNXImport 显式支持：`202` 个 ONNX op 名称，其中 `Upsample` 作为 `Resize` 别名处理。
- forward 实际接入 C 后端：`178` 个算子类。
- 合理保留 Python 调度、控制流、序列、可选值、字符串、图像 IO 或元数据运行时：`23` 个算子类。
- 普通数值/张量算子 Python-only 运行时：`0` 个。
- CUDA verifier：`178` 个。
- 默认 active numerical plan：`178` 个唯一算子名称，`618` 条默认计划。
- 默认 active numerical plan 混合精度覆盖：`425` 条计划。

## 本轮已落盘内容

- 新增并接入 `Tril`、`Triu`、`Trilu` 的 CUDA reference，其中三角矩阵公共逻辑放在 `cuda/verify_triangular_common.cuh`。
- 新增并接入 `HannWindow`、`HammingWindow`、`BlackmanWindow` 的 CUDA reference；`HammingWindow` 使用项目/ONNX 当前公式 `alpha=25/46`、`beta=21/46`。
- 新增并接入 `Range`、`OneHot`、`ReverseSequence`、`Det`、`MelWeightMatrix` 的 CUDA reference 和默认 numerical plan。
- 新增并接入 `DynamicQuantizeLinear` 的 CUDA reference 和默认 numerical plan，runner 按 `y`、`y_scale`、`y_zero_point` 三个输出分别校验。
- 新增并接入 `Split` 的 CUDA reference 和默认 numerical plan，runner 按多个输出逐一校验。
- 新增并接入 `Unique` 的 CUDA reference 和默认 numerical plan，runner 按 `values`、`indices`、`inverse`、`counts` 四个输出逐一校验。
- 新增并接入 `Dropout` 的 CUDA reference 和默认 numerical plan，runner 按 `y` 和 `mask` 双输出分别校验。
- 新增并接入 `RandomUniform`、`RandomNormal`、`RandomNormalLike` 和 `Bernoulli` 的 CUDA reference 和默认 numerical plan。
- `RandomUniformLike` 的 numerical 改为真实 C-backed forward 对比 CUDA reference，不再使用 Python 侧 reference 绕过 C 后端。
- 随机 uniform/normal/Bernoulli 的 C 后端随机状态改为按元素由 seed 派生，保证 OpenMP 并行调度不影响可复现结果。
- 新增并接入 `Multinomial` 的 CUDA reference 和默认 numerical plan，覆盖 int64/int32 输出、零概率、one-hot 和非归一化概率行。
- 新增并接入 `NegativeLogLikelihoodLoss`、`SoftmaxCrossEntropyLoss` 和 `NonMaxSuppression` 的 CUDA reference 和默认 numerical plan，覆盖 ignore_index、reduction、权重、log_prob 多输出、score 阈值、IoU 抑制和 center_point_box 主路径。
- 继续扩展 `NonMaxSuppression` 默认 numerical plan，新增多 batch/class、稳定排序 tie、阈值等值包含和 float16 空输出边界 case。
- 新增并接入 `Binarizer` 的 CUDA reference 和默认 numerical plan，覆盖 threshold 两侧和等于 threshold 的严格大于边界，并补充 float16、bfloat16、float8_e4m3、float8_e5m2 混合精度计划。
- `tools/numerical/runner.py` 补充了这些算子的输入准备、参数打包、ONNX `output_datatype` 映射和 reference 调用适配。
- `docs/reports/operator_coverage.md`、`docs/reports/current_progress.md` 和 `docs/reports/progress_handoff.md` 已按最新统计同步。

## 本轮提交前验证

- `git diff --check`：通过。
- `python -m py_compile tools/numerical/cli.py tools/numerical/runner.py`：通过。
- `python tools/audit_ops.py --output docs/reports/operator_coverage.md`：通过，覆盖报告已刷新。
- `python tools/cli.py compile-cuda`：通过，`178` 个 CUDA verifier 编译成功。
- `python tools/cli.py numerical --op dynamic_quantize_linear --iterations 3 --skip-plots`：通过。
- `python tools/cli.py numerical --op split --iterations 3 --skip-plots`：通过。
- `python tools/cli.py numerical --op unique --iterations 3 --skip-plots`：通过。
- `python tools/cli.py numerical --op dropout --iterations 3 --skip-plots`：通过。
- `python tools/cli.py numerical --op random_uniform --op random_uniform_like --op random_normal --op random_normal_like --op bernoulli --iterations 3 --skip-plots`：通过。
- `python tools/cli.py numerical --op multinomial --iterations 3 --skip-plots`：通过。
- `python tools/cli.py numerical --op binarizer --op negative_log_likelihood_loss --op softmax_cross_entropy_loss --op non_max_suppression --iterations 3 --skip-plots`：通过。
- `python tools/cli.py numerical --iterations 1 --skip-plots`：通过，`618` 条默认计划完整 numerical 一轮全部通过。
- `python -m pytest -q tests`：通过，`292 passed, 1 skipped`。

## 已知还没有完成

当前未发现仍需立即后端化或接入默认 numerical 的普通 C-backed 数值/张量算子。后续应继续扩展 C/CUDA reference 的属性、dtype 和边界 case matrix，而不是回退为普通数值路径的 Python-only 实现。

以下类别属于合理保留 Python 调度的范围，不按普通 C/CUDA 数值门禁衡量；后续应继续通过 ONNX reference pytest、导入器语义测试和端到端图测试补强。

- 控制流：`If`、`Loop`、`Scan`。
- 序列：`SequenceEmpty`、`SequenceConstruct`、`SequenceAt`、`SequenceInsert`、`SequenceErase`、`SequenceLength`、`SequenceMap`、`ConcatFromSequence`、`SplitToSequence`。
- 可选值：`Optional`、`OptionalGetElement`、`OptionalHasElement`。
- 字符串：`RegexFullMatch`、`StringConcat`、`StringSplit`、`StringNormalizer`、`TfIdfVectorizer`。
- 元数据/常量/图像：`Shape`、`Constant`、`ImageDecoder`。

## 剩余风险

- 名称级覆盖已经闭合，但并不等同于所有 ONNX 属性组合、类型约束、异常路径和边界输入的穷尽证明。
- 默认 numerical 是工程回归门禁，不是形式化证明；后续仍需要围绕官方 schema 增加高风险边界 case。
- 损失函数和 NMS 已进入默认 C/CUDA numerical 门禁，且 NMS 已补充排序 tie、阈值等值包含和空输出 case；后续仍需要继续扩展 ignore index、reduction、更多阈值组合和极端权重等边界集合。

## 后续建议

1. 按高风险属性矩阵扩展损失/NMS、ROI、序列、谱、Attention、DeformConv、量化和采样类算子的边界集合。
2. 审计所有“含 Python 调度或 fallback”的复杂算子路径，确认普通数值主路径持续由 C 后端承载。
