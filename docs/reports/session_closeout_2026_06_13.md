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
- CUDA verifier：`166` 个。
- 默认 active numerical plan：`166` 个唯一算子名称，`587` 条默认计划。
- 默认 active numerical plan 混合精度覆盖：`409` 条计划。

## 本轮已落盘内容

- 新增并接入 `Tril`、`Triu`、`Trilu` 的 CUDA reference，其中三角矩阵公共逻辑放在 `cuda/verify_triangular_common.cuh`。
- 新增并接入 `HannWindow`、`HammingWindow`、`BlackmanWindow` 的 CUDA reference；`HammingWindow` 使用项目/ONNX 当前公式 `alpha=25/46`、`beta=21/46`。
- 新增并接入 `Range`、`OneHot`、`ReverseSequence`、`Det`、`MelWeightMatrix` 的 CUDA reference 和默认 numerical plan。
- 新增并接入 `DynamicQuantizeLinear` 的 CUDA reference 和默认 numerical plan，runner 按 `y`、`y_scale`、`y_zero_point` 三个输出分别校验。
- `tools/numerical/runner.py` 补充了这些算子的输入准备、参数打包、ONNX `output_datatype` 映射和 reference 调用适配。
- `docs/reports/operator_coverage.md`、`docs/reports/current_progress.md` 和 `docs/reports/progress_handoff.md` 已按最新统计同步。

## 本轮提交前验证

- `git diff --check`：通过。
- `python -m py_compile tools/numerical/cli.py tools/numerical/runner.py`：通过。
- `python tools/audit_ops.py --output docs/reports/operator_coverage.md`：通过，覆盖报告已刷新。
- `python tools/cli.py compile-cuda`：通过，`166` 个 CUDA verifier 编译成功。
- `python tools/cli.py numerical --op dynamic_quantize_linear --iterations 3 --skip-plots`：通过。
- `python tools/cli.py numerical --iterations 1 --skip-plots`：通过，完整默认 numerical 一轮全部通过。
- `python -m pytest -q tests`：通过，`292 passed, 1 skipped`。

## 已知还没有完成

以下 `11` 个普通 C-backed 算子仍缺少独立 CUDA verifier 或默认 active numerical plan，后续应继续补 C/CUDA reference 和默认数值门禁，而不是回退为普通数值路径的 Python-only 实现。

- 随机/采样：`Bernoulli`、`Multinomial`、`RandomNormal`、`RandomNormalLike`、`RandomUniform`。
- 随机掩码：`Dropout`。
- 损失/检测：`NegativeLogLikelihoodLoss`、`SoftmaxCrossEntropyLoss`、`NonMaxSuppression`。
- 集合：`Unique`。
- 结构拆分：`Split`。

以下类别属于合理保留 Python 调度的范围，不按普通 C/CUDA 数值门禁衡量；后续应继续通过 ONNX reference pytest、导入器语义测试和端到端图测试补强。

- 控制流：`If`、`Loop`、`Scan`。
- 序列：`SequenceEmpty`、`SequenceConstruct`、`SequenceAt`、`SequenceInsert`、`SequenceErase`、`SequenceLength`、`SequenceMap`、`ConcatFromSequence`、`SplitToSequence`。
- 可选值：`Optional`、`OptionalGetElement`、`OptionalHasElement`。
- 字符串：`RegexFullMatch`、`StringConcat`、`StringSplit`、`StringNormalizer`、`TfIdfVectorizer`。
- 元数据/常量/图像：`Shape`、`Constant`、`ImageDecoder`。

## 剩余风险

- 名称级覆盖已经闭合，但并不等同于所有 ONNX 属性组合、类型约束、异常路径和边界输入的穷尽证明。
- 默认 numerical 是工程回归门禁，不是形式化证明；后续仍需要围绕官方 schema 增加高风险边界 case。
- `Split` 和 `Unique` 的主要风险在多输出、排序稳定性、inverse/counts 输出和 dtype 对齐。
- `Dropout` 和随机采样算子需要先确定 deterministic seed、统计容差和训练/推理模式口径。
- 损失函数和 NMS 需要重点验证 ignore index、reduction、阈值比较、排序 tie-break 和空输出。

## 后续建议

1. 先处理 `Split`、`Unique` 和 `Dropout`，尽快把确定性较强的缺口收掉。
2. 再处理 `NegativeLogLikelihoodLoss`、`SoftmaxCrossEntropyLoss` 和 `NonMaxSuppression`，每个算子先补官方 reference pytest 边界，再接 CUDA numerical。
3. 最后集中处理随机/采样算子，先制定 seed 和统计容差策略，再纳入默认门禁。
