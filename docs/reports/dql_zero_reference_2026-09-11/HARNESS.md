# DynamicQuantizeLinear 全零 ReferenceEvaluator 数值门禁

## 范围

本门禁只覆盖有限、非空且所有元素数值为零的 float32 输入，包括混合 `+0.0` 与 `-0.0`。预期输出固定为：

- `y`：与输入同形状的全零 `uint8`；
- `y_zero_point`：标量 `uint8(0)`；
- `y_scale`：标量 float32 `1 / 255`，按位精确比较。

该行为用于对齐当前选定的 ONNX ReferenceEvaluator profile。ONNX schema FunctionBody、ReferenceEvaluator 与 ONNX Runtime 在零范围上的历史分歧仍作为规范缺口保留，不能把此门禁描述为 FunctionBody 已明确规定的唯一结果。

## 设计

默认 numerical inventory 增加普通全零与 signed-zero 两个固定输入计划。输入生成器严格拒绝空、非有限或含非零值的 profile 输入。DQL 特殊输出处理器根据每轮实际输入识别有限、非空全零边界，即使计划未带 profile 标记也会同时将 C 输出和 CUDA packed 输出分别与独立的 ReferenceEvaluator profile 常量比较，避免二者同时返回旧值 `1.0` 时互相印证而假通过。标记只表达固定计划的生成意图，不会传给算子构造函数。常规 DQL scale 容差不变；profile 的 scale 使用 float32 位模式精确比较。

非零 range 经 float32 `/255` 下溢为零时继续采用既有 `scale=1` 策略，不属于本次对齐范围。

## 实际结果

唯一 fresh 构建由并行 numeric 任务完成：`make` RC 0，`tensor_ops.so` SHA-256 为 `4f94df5ccbe2df3b1df054733265d1a57d7bae993a854d92745b3973d248a0e6`；DQL CUDA 单目标编译 `compiled=1, skipped=0`、RC 0，二进制 SHA-256 为 `2c39a72d9b79686a184405f641a4345850bbfebbf4f7a761ab15383ffb911d2e`。

本任务随后执行不含真实 GPU 的定向集合，覆盖新增 DQL zero profile、mocked special-output contracts、runner architecture 与 numerical harness regressions：

- 结果：`80 passed in 6.25s`；
- RC：`0`；
- stderr：空；
- 命令：[harness_targeted.command.txt](harness_targeted.command.txt)；
- 范围：[harness_targeted.scope.txt](harness_targeted.scope.txt)；
- stdout：[harness_targeted.stdout.txt](harness_targeted.stdout.txt)；
- stderr：[harness_targeted.stderr.txt](harness_targeted.stderr.txt)；
- RC：[harness_targeted.rc.txt](harness_targeted.rc.txt)。

第一次尝试通过 PowerShell 内联传递 Bash 变量时，变量被宿主 shell 提前展开，目录创建即失败；pytest 未启动，也未产生测试结果。随后改用固定脚本 [run_harness_targeted.sh](run_harness_targeted.sh) 执行上述成功门禁。

在 numeric 任务确认真实 GPU 测试进程结束后，本任务复用同一 fresh `.so` 与 DQL CUDA 二进制运行精确 CLI 过滤：3 个已注册 DQL 计划各执行 3 轮，其中包括 1 个常规计划与 2 个 zero-reference 计划。

- 结果：三个计划分别 `3/3`，汇总 `Samples 9`、绝对误差和相对误差均为 `0`；
- RC：`0`；
- stderr：空；
- 命令：[dql_cli.command.txt](dql_cli.command.txt)；
- 范围：[dql_cli.scope.txt](dql_cli.scope.txt)；
- 元数据及复用二进制哈希：[dql_cli.meta.txt](dql_cli.meta.txt)；
- stdout：[dql_cli.stdout.txt](dql_cli.stdout.txt)；
- stderr：[dql_cli.stderr.txt](dql_cli.stderr.txt)；
- RC：[dql_cli.rc.txt](dql_cli.rc.txt)；
- 执行脚本：[run_dql_cli.sh](run_dql_cli.sh)。
