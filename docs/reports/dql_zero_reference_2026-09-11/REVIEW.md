# DynamicQuantizeLinear 全零策略交叉审阅

## 审阅身份

- 基线：`45fc62b926324375d5247e3a59dc665a8673887c`
- 模式：只读实现审阅；未修改 numeric/harness source，未运行测试或构建。
- 范围：C runtime、CUDA verifier、既有/新增 DQL tests、numerical plans/input/NPS/special-output 路径及其测试。

## C 与 CUDA 实现

最终审阅通过以下合同：

- C 使用 `x->size > 0` 初始化 all-zero predicate；CUDA 使用 `n > 0`。两者均逐值要求 `isfinite(value)` 且 `value == 0.0f`，因此接受 `+0/-0`，拒绝 NaN、infinity、非零常量和空输入。
- 全零 scale 在两侧都直接物化为 C/CUDA float 表达式 `1.0f / 255.0f`。本地 ONNX 1.21 官方包 `onnx/reference/ops/op_dynamic_quantize_linear.py` 同样将 float32 `1.0` 除以 float32 `255`；新增 test 还读取真实 ReferenceEvaluator 输出并断言 uint32 bits。
- 非零输入继续使用 anchored min/max 与 float32 range/255。非零 subnormal 使 division scale 下溢为 0 时不会命中 all-zero predicate，随后仍由既有 `scale == 0` 保护变为 `1.0f`。
- normal DQL 路径的逐阶段 float32 计算、公开 scale、ties-to-even 与 REAUD-001 fixture 没有被改写。新增 normal-path tests 包含正/负常量、普通混合值和既有 rounding boundary。
- 选择说明限定为 ONNX 1.21 ReferenceEvaluator compatibility，没有将其写成 opset 17 schema 强制要求；没有声明新增 empty/NaN/infinity 支持。

## Numerical harness

最终审阅确认：

- 默认 DQL plans 从 1 个增加到 3 个；新增纯 `+0` 和混合 `+0/-0` 两个固定计划，普通 plan 未改变。
- profile 输入生成器要求 finite、nonempty、all-zero；`runner_nps` 在构造算子前剥离内部 marker。
- 显式 profile 会将 NPS/C 的三个输出和 CUDA packed 三个字段分别与独立 oracle 比较。scale 使用 uint32 bit pattern，y/zp exact，因此双方同时返回旧 scale `1`、scale `0`、1 ULP 错误或相同错误 y/zp 都不能通过。普通 DQL 的既有容差没有放宽。

## 初轮 finding 与处理

`runner_special_outputs.py` 初版只在 `dql_zero_reference_profile` marker 存在时启用独立 oracle。若运行时实际输入是 finite nonempty all-zero、但调用计划没有 marker，处理器仍只比较 C 与 CUDA，两侧同错可假通过。

owner 已采纳：最终实现直接根据实际输入的 `size > 0 && all finite && all == 0.0` 启用严格 oracle。marker 存在但输入不符合 profile 时 fail closed；marker 不再是 strict oracle 的必要条件。新增回归覆盖无 marker 时双方同时返回 scale `1`、`0` 和正确值相邻 1 ULP 均失败，并确认 marker 在 NPS 构造算子前被剥离、原始 `init_args` 仍保留。

## 最终结论

只读最终 diff 未发现阻断或可操作问题。C、CUDA 与 harness 的 profile 判据一致排除 empty、NaN、infinity 和任何非零值；非零 subnormal `/255` 下溢继续命中既有 scale `1` 保护。普通 DQL plan、normal tolerance、REAUD-001 bit-exact fixture 与输出 contract 均未放宽。

源码审阅 verdict 为 **PASS**。统一构建已由 numeric agent 报告成功：`make` RC 0；DQL 单目标 CUDA fresh build RC 0，`compiled=1 skipped=0`；`tensor_ops.so` SHA256 为 `4f94df5ccbe2df3b1df054733265d1a57d7bae993a854d92745b3973d248a0e6`，DQL verifier SHA256 为 `2c39a72d9b79686a184405f641a4345850bbfebbf4f7a761ab15383ffb911d2e`。动态测试结果仍等待两个 owner 的定向运行，不以本静态审阅替代。

## Source snapshot 精确清单（待定向结果后提交）

source snapshot 仅包含以下 11 个生产源码、测试和稳定策略文件：

1. `tensor_ops/tensor_ops_dynamic_quant.c`
2. `cuda/verify_dynamic_quantize_linear.cu`
3. `tools/numerical/cli.py`
4. `tools/numerical/runner_inputs.py`
5. `tools/numerical/runner_nps.py`
6. `tools/numerical/runner_special_outputs.py`
7. `tests/test_operator_misc_semantics.py`
8. `tests/test_reaudit_quantization_precision.py`
9. `tests/test_dynamic_quantize_zero_reference.py`
10. `tests/test_numerical_dql_zero_reference.py`
11. `docs/dynamic_quantize_zero_policy.md`

不纳入 `docs/reports/dql_zero_reference_2026-09-11/` 的新报告与 build logs，不纳入旧审计、cache、binary、模型或其他生成物。提交操作等待两组定向测试结果和根 agent 明确调度。
