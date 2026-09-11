# 推送后修复汇总

日期：2026-09-11

分支：`codex/post-push-fixes-20260911`

基线：`96d070e6f1a26f61ee0cf727c9616148b8680629`
冻结源提交：`e0a03c6b3e71ba5c9562624f783c3ac6e4165322`

## 范围与结论

本轮修复覆盖 ONNX Resize-17 默认属性、空维 Slice、LayerNormalization-17 分阶段精度和可选输出、CUDA 编译缓存身份、普通数值计划输出 dtype 契约，以及 LayerNormalization 数值计划与 CUDA 协议。定向回归、原生构建、全部 CUDA verifier 编译、Resize 图门禁及 LayerNormalization 真实 CUDA 门禁均已通过。

修复工作树的最终验证状态为 **PASS**：完整 pytest 和完整 numerical 均已在同一冻结提交及二进制身份下通过。主检出目录也已快进到同一提交，并通过重新构建和指定定向门禁；报告提交仍待后续阶段完成。验收明细见[验收记录](ACCEPTANCE.md)。

## 修复内容

### PP-NATIVE-001：Resize-17 默认属性

导入器和直接包装器在属性缺省时使用 ONNX Resize-17 默认值：`coordinate_transformation_mode="half_pixel"`、`nearest_mode="round_prefer_floor"`；模型显式提供的属性继续优先。详细说明见[形状与导入修复](SHAPE_FIX.md)。

### PP-GRAPH-002：Slice 空维语义

空维度在正步长和负步长下均产生零长度输出，C 路径、Python 回退和 `forward_` 共用规范化参数与输出形状。非空维度继续遵循 ONNX 的正式裁剪规则。特别地，对 `x=[0,1,2]`、`starts=ends=INT64_MIN`、`step=-1`，正式规则结果为 `[0]`；ONNX 1.21 `ReferenceEvaluator` 因 Python 切片行为返回 `[]`。实现有意遵循 schema，而非该非空极值上的 ReferenceEvaluator 行为。

### PP-NATIVE-002（P2）：LayerNormalization-17 stash 精度

`stash_type=1` 的第一阶段逐步物化为 float32；`stash_type=16` 支持 bfloat16 物化。归一化结果随后转换到输入类型 `T`，缩放和偏置阶段也按 `T` 物化，与 stash 类型分离。大偏移输入 `[[1e8,1e8,1e8+8]]` 的正确结果为 `[0,0,1.7320505]`，修复了旧双精度阶段产生 `[-0.7071065,-0.7071065,1.4142131]` 的问题；普通 `[1,2,3]` 用例原先仅差 1 ULP。

新原生入口当前覆盖 FLOAT stash，且 float32/float64 输入走 C；float16/bfloat16 输入走已修正的 Python 回退。低精度语义已有正确性覆盖，但本轮未做性能基准，因此这一路径选择可能带来的性能影响尚未量化。真实 GPU dtype 门禁通过 CUDA verifier 执行，不代表 NPS 的低精度输入走了原生 C。

CUDA 参数由七个 int32 字段 `row_count, normalized_size, has_scale, has_bias, emit_stats, stash_type, input_dtype` 和一个 float32 epsilon 组成；输入类型码为 FLOAT=1、FLOAT16=10、DOUBLE=11、BFLOAT16=16。非法 stash 值会被拒绝；缺少新符号的旧 `.so` 会安全回退。详细说明见[LayerNormalization 修复](NORMALIZATION_FIX.md)。

### PP-GRAPH-001：LayerNormalization 可选输出槽位

LayerNormalization 的可选输出保持 ONNX 声明位置：例如 `outputs=["y", "", "inv_std"]` 返回 `(y, None, inv_std)`，使 `Graph.forward` 和 `Graph.forward_` 能按输出索引正确绑定。`outputs=[]` 继续兼容，形状路径与数值路径遵循同一槽位规则。

### PP-TOOL-002：识别编译器的 CUDA 缓存

编译身份保存在 `cache/.compile-identities/`，不会混入 `cache/verify_*`。身份包含解析后的编译器路径、编译器二进制 `cksum` 和 `--version`。旧缓存缺少元数据时重编一次；重编开始前移除旧身份，因此编译器部分覆盖输出后失败时，下次仍会重编。`--force`、`--op` 和同编译器的新鲜度判断保持有效。

### PP-TOOL-003：普通输出物理 dtype 门禁

普通 NPS 的非整数输出在标量归一化、解码、量化或数值比较前检查实际 NumPy dtype。检查遵循项目物理存储：bool 使用 `np.bool_`，float16/32/64 使用对应 NumPy dtype，bfloat16 使用 `uint16`，float8 使用 `uint8`。即使数值相同，错误 dtype 也会失败。`INTEGER_DTYPES` 继续由既有 `compare_integer_output` 检查 dtype、范围、浮点 wire 整数性和精度，没有弱化原整数协议；特殊输出继续由各自 schema 处理。

### LayerNormalization 数值计划与协议覆盖

默认计划加入固定的大偏移 `input_values`；该辅助字段在构造算子前移除。真实且有界的 NPS 门禁覆盖输入准备、Tensor 构造、LayerNormalization 构造和 forward。单输出计划显式使用 `outputs=["y"]`。工具修复详情见[验证工具修复](VALIDATION_FIX.md)。

## 已完成验证

| 门禁 | 结果 |
| --- | --- |
| 原生 `make` | PASS，RC 0；最终 `tensor_ops.so` SHA-256 `6f35ec47feff1a5e833030548a6a6642b7160d698d789ffecae80f2b7202de89` |
| 全量 CUDA 编译 | PASS，RC 0；178 个 verifier，`compiled=178 skipped=0` |
| LayerNormalization 强制重编 | PASS，RC 0；`compiled=1 skipped=0`；SHA-256 `cc31ed88f8509fb01812b9a1e641ecd912b43d94c987f9df14e36ed9195dd074` |
| Resize-17 `verify-graph` | PASS，RC 0，stderr 为空 |
| Resize/Slice 定向测试 | PASS，7 passed in 1.45s |
| LayerNormalization 原生/真实 CUDA 定向测试 | PASS，13 passed in 3.68s；包含 FLOAT32、FLOAT16、DOUBLE 独立 oracle 及可选输出 |
| 工具定向测试 | PASS，25 passed in 2.58s |
| LayerNormalization CLI numerical | PASS，9 个计划 × 3 次迭代 = 27/27，RC 0，报告误差为 0 |
| 补修定向测试 | PASS，30 passed；证据为 `evidence/numeric/layernorm_post_shard_targeted.*` |
| 整数诊断回归与工具门禁 | PASS，26 passed in 2.67s |
| 完整 pytest attempt2 | PASS，601 passed、1 skipped in 20.72s，RC 0；唯一 skip 为 ONNX17 不支持 Celu float16 |
| 完整 numerical | PASS，726/726 个计划，每项 3 次迭代，共 2178 次，RC 0；无失败、崩溃或跳过 |

构建、图门禁与产物身份见[集成验证](INTEGRATION_VALIDATION.md)，主目录结果见[主检出目录集成验证](MAIN_SYNC.md)；原始日志位于本目录及 `evidence/tooling/`、`evidence/numeric/`、`evidence/main/`。

## 最终门禁与交付状态

| 项目 | 状态 |
| --- | --- |
| 源提交 | `e0a03c6b3e71ba5c9562624f783c3ac6e4165322` |
| 最终 `tensor_ops.so` | `6f35ec47feff1a5e833030548a6a6642b7160d698d789ffecae80f2b7202de89` |
| 完整 pytest | PASS，601 passed、1 skipped，RC 0 |
| 完整 numerical（三次迭代） | PASS，726/726 个计划、2178 次迭代，RC 0 |
| 主检出目录快进与定向验证 | PASS，HEAD `e0a03c6b3e71ba5c9562624f783c3ac6e4165322`；20 项 pytest、27/27 LayerNormalization、15/15 Resize 均通过 |
| 主检出目录报告提交 | PASS，本轮目录随独立 report-only commit 交付 |

完整门禁已在同一源提交和上述二进制身份下通过。未执行本轮性能基准；LayerNormalization 低精度 Python 回退的潜在性能影响是当前明确限制。
