# DynamicQuantizeLinear 全零兼容策略验收

## 当前结论

**最终验收 PASS。** 冻结源码、fresh C/CUDA 构建、三组定向 pytest、numerical harness 定向测试、真实 DQL CLI、full pytest 与完整 numerical 均有一致的成功证据。没有失败、异常或未闭合门禁。

## 冻结源码身份

- 冻结提交：`a1d4940cc694ac65fbb77a993b7b800fcafd6ee5`（`fix: align dynamic quantize zero reference`）。
- `git diff-tree --no-commit-id --name-only -r HEAD` 恰好列出 11 个预定文件：C、CUDA、4 个 numerical harness 文件、4 个测试文件及稳定策略文档。
- tracked worktree 与 index 均为空；当前 `git status --short` 仅列出未跟踪的报告证据目录。
- 静态最终审阅 [REVIEW.md](REVIEW.md) verdict 为 **PASS**，没有遗留阻断项。

## 策略合同

[稳定策略](../../dynamic_quantize_zero_policy.md)准确限定为 ONNX 1.21 `ReferenceEvaluator` compatibility，不将 `float32(1/255)` 描述为 opset 17 schema 的强制唯一结果。全零分支要求输入同时满足 nonempty、finite、逐值 `value == 0.0f`，因此覆盖 `+0/-0`，排除 empty、NaN、infinity 与非零常量。非零 range 的 float32 `/255` 下溢仍走既有 `scale=1` 保护，未被全零分支吸收。

C 与 CUDA 使用相同的显式输入谓词和 `1.0f / 255.0f` 表达式。普通 DQL 的 anchored min/max、float32 rounding、zero point 舍入及容差没有放宽。

## 构建与二进制身份

唯一 fresh 构建记录于 [BUILD.md](BUILD.md)：

| 项目 | 结果 | SHA-256 |
|---|---|---|
| `make` / `tensor_ops.so` | RC 0 | `4f94df5ccbe2df3b1df054733265d1a57d7bae993a854d92745b3973d248a0e6` |
| DQL 单目标 CUDA，`compiled=1 skipped=0` | RC 0 | `2c39a72d9b79686a184405f641a4345850bbfebbf4f7a761ab15383ffb911d2e` |

当前磁盘上的两个二进制哈希与构建记录一致。构建发生于提交冻结前；随后提交的生产 C/CUDA 与直接回归源保持构建快照内容，后续变化位于 Python harness 与测试，不改变这两个二进制。

## 定向证据

| 门禁 | 结果 | Skips | 覆盖重点 |
|---|---:|---:|---|
| `test_dynamic_quantize_zero_reference.py` | 11 passed | 0 | 真实 ONNX 1.21 ReferenceEvaluator、C wrapper、GPU verifier；多 shape signed zero、常量、普通输入、REAUD-001 fixture、underflow 分支 |
| `test_reaudit_quantization_precision.py` | 11 passed | 0 | 既有量化精度和 bit-exact 边界 |
| misc semantics 的 DQL 选择 | 3 passed | 0 | 既有 reference、公开 scale 与 ties-to-even 行为 |
| harness 定向集合 | 80 passed | 0 | zero plans、输入生成、mocked contract、runner 架构与回归 |
| 真实 DQL CLI | 3 plans × 3/3 | 0 | 1 个常规计划、2 个 zero-reference 计划；9 samples，abs/rel 均为 0 |

前三项合计为请求中的 25 个实现定向通过项。80 项工具测试证据见 [HARNESS.md](HARNESS.md)。真实 CLI 的命令、范围、stdout、stderr、RC 和运行时二进制哈希分别保存在 `dql_cli.*`；RC 为 0 且 stderr 为空。

## 独立 oracle 与同错防护

numerical 特殊输出处理器根据**每轮实际输入**计算 `size > 0 && all finite && all == 0.0`，不依赖 plan marker 才启用严格判断。对该输入，C/NPS 与 CUDA packed 输出分别与固定 ReferenceEvaluator profile 比较：`y` 与 `zero_point` 精确比较，scale 以 float32 `uint32` 位模式比较。

`test_numerical_dql_zero_reference.py` 明确覆盖双方同时返回错误值的场景：scale 为 `1`、`0`、正确值相邻 1 ULP，以及双方相同的错误 `y` 或 `zero_point` 均必须失败。另有无 marker 的全零输入测试，证明严格 oracle 来自实际输入而非 marker 路径；marker 仅控制固定样本生成，并在算子构造前剥离。这排除了 C 与 CUDA 同错却互相印证的假通过。

## 全量门禁状态

control 的 full pytest 证据已结束：`561 passed, 1 skipped in 21.47s`，RC 0；唯一 skip 是既有 ONNX 17 Celu float16 不支持项。

完整 numerical 于 `2026-09-11T05:37:47Z` 结束，证据核对结果为：

- preflight RC、Python RC 与 wrapper RC 均为 `0`；stderr 为空；
- metadata 的 expected/actual plan count 均为 `725`，`actual_pass_count=725`；
- 每个 plan 执行 `3` 轮，expected/actual passing iterations 均为 `2175`；
- raw stdout 含 `725` 个 `Testing` 条目和 `725` 个 `Pass (3/3)`，`FAILED`、`Traceback`、`Exception`、`Error` 与失败符号计数均为 `0`；
- 三个 DQL plan 均为 `3/3`：一个普通计划和两个新增 zero-reference 计划；DQL 汇总为 `9` samples，绝对误差和相对误差均为 `0`；
- source HEAD 为冻结提交 `a1d4940cc694ac65fbb77a993b7b800fcafd6ee5`；`tensor_ops.so` 与 DQL CUDA verifier 哈希分别保持 `4f94df5c...` 和 `2c39a72d...`；CUDA verifier inventory 为 `178`，均与预期一致。

原始证据为 `full_numerical.stdout.txt`、空的 `full_numerical.stderr.txt`、`full_numerical.python.rc.txt`、`full_numerical.wrapper.rc.txt`、`full_numerical.preflight.rc.txt`、`full_numerical.meta.txt` 和 `full_numerical.cuda_inventory.txt`。Full pytest 的命令、stdout、空 stderr 与 RC 同样保存在本目录的 `full_pytest.*` 文件中。

本次通过结论仅确认项目选择的 ONNX 1.21 `ReferenceEvaluator` compatibility profile 及其回归门禁；它不把该全零结果表述为 opset 17 的规范唯一行为。
