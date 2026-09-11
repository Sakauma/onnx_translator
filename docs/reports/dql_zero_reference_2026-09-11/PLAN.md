# DynamicQuantizeLinear 全零输入 ReferenceEvaluator 对齐计划

## 基线与选择

- 基线提交：`45fc62b926324375d5247e3a59dc665a8673887c`
- 工作目录：`D:\workspace\onnx_translator_bugfix_worktree`
- 用户选择：全零 DynamicQuantizeLinear 对齐 ONNX 官方 Python 包的 `ReferenceEvaluator` 行为，输出 `scale=float32(1/255)`、`y=0`、`zero_point=0`。

本地安装的 ONNX 官方包文件 `/home/sakauma/data/miniconda3/envs/egor/lib/python3.12/site-packages/onnx/reference/ops/op_dynamic_quantize_linear.py` 在 `maxx == minx` 时先选择 float32 `1.0`，再除以 float32 `255`。本项目采用该实现行为作为产品选择。既有复审已记录 schema function 展开、ReferenceEvaluator 与 ONNX Runtime 在全零边界不一致，因此报告不得将这一选择描述成 opset 17 schema 强制语义。

## 输入域边界

全零专用分支只适用于：

- 输入元素数大于 0；
- 每个输入值均为 finite；
- 每个输入值与浮点零相等，因此 `+0.0` 与 `-0.0` 都属于全零。

判定必须检查输入值，不能使用 `min == max`、`range == 0` 或计算后的 `scale == 0` 代替。这样可避免把下列输入错误归入全零分支：

- 非零 float32 subnormal 除以 255 后 scale 下溢为 0；
- 含 NaN 或 infinity 的输入；
- 空输入。

非零 scale 下溢继续走 normal 路径，并暂时保留既有 `scale=1.0f` 的除零保护，不能为了全零选择而静默改成 `1/255`。空输入与非 finite 输入不属于本轮新增支持声明。

## 实现审阅合同

### C runtime

- C 端从真实输入值判定 finite nonempty all-zero。
- 全零分支物化单一 float32 常量 `1.0f / 255.0f`；公开 scale 与量化使用值一致。
- `y` 每个元素写 uint8 0，scalar zero point 写 uint8 0。
- normal 非零路径保持 REAUD-001 已修正的逐阶段 float32 extrema/range/scale/quotient 顺序和 ties-to-even。

### CUDA reference

- CUDA verifier 使用与 C 同样的输入域边界与 float32 常量。
- 全零 `+0/-0` packed wire 为 integral uint8 语义字段、finite scalar scale 和 scalar zero point。
- 非零输入继续经 min/max、normal scale 与 kernel 路径，不能因 `scale == 0` 误命中全零选择。

### Numerical harness

- 新增确定性的全 `+0` 与混合 `+0/-0` DQL plans，确保 NPS/C 输出和 CUDA packed reference 都被完整三输出比较。
- scale 比较必须能区分 `1.0f`、`1.0f/255.0f` 和 `0.0f`，不能用宽松绝对容差掩盖 profile 错误；y/zp 仍执行 dtype、shape、finite/integral/range contract。
- 保留现有正常 DQL plan 和 REAUD-001 rounding boundary fixture，防止只修零输入而回归普通路径。
- 若新增 plans 改变默认总数，以运行时实际 inventory 和日志为准，不预填计划数量。

## 验证顺序

实现 agents ready 后先做最终 diff 交叉审阅，不直接修改其 source。随后按根 agent 调度执行：

1. numeric agent 作为唯一执行者运行 `make`，并使用 `python tools/cli.py compile-cuda --op dynamic_quantize_linear --force` fresh 单目标构建，不重编无关 177 个 verifier；
2. 在新 `.so` 与新 DQL verifier 上运行 C/DQL、CUDA/DQL、numerical profile 与既有 DQL precision 的三组 targeted tests，并完成独立 review；
3. freeze source snapshot；
4. 在 frozen source 和 fresh binary 上只运行一次 `python -u -m pytest -q -ra tests`；
5. 只运行一次完整 native `python -u tools/cli.py numerical --iterations 3 --skip-plots`，以实际 plan/sample/pass/fail/exception 数记录；
6. 图 importer/shape 路径未变化时不重复图 CLI；若最终 diff 触及相关路径，再由根 agent 决定是否补图 gate。

每步保存 command、stdout、stderr、真实 RC、source SHA 与相关 binary hash。禁止 `verify_all`、clean、目录删除、提交或 push；当前阶段不运行测试、构建或主目录同步。

## 报告职责

- `PLAN.md`：范围、依据、边界和门禁设计；
- `REVIEW.md`：实现 ready 后的最终只读交叉审阅；
- `VALIDATION.md`：只在实际命令完成后记录结果，不预写通过；
- `SUMMARY.md`：最终选择、实现和证据摘要，不修改旧复审归档。
