# DynamicQuantizeLinear 全零兼容修复验证

## 验证对象

- 源提交：`a1d4940cc694ac65fbb77a993b7b800fcafd6ee5`
- `tensor_ops.so` SHA-256：`4f94df5ccbe2df3b1df054733265d1a57d7bae993a854d92745b3973d248a0e6`
- `cache/verify_dynamic_quantize_linear` SHA-256：`2c39a72d9b79686a184405f641a4345850bbfebbf4f7a761ab15383ffb911d2e`
- WSL：`ubuntu2004`
- Python：`/home/sakauma/data/miniconda3/envs/egor/bin/python`

## 构建与定向门禁

| 门禁 | 结果 |
|---|---|
| `make` | RC 0，fresh `tensor_ops.so` 构建成功 |
| DQL CUDA 单目标强制编译 | RC 0，`compiled=1 skipped=0` |
| ReferenceEvaluator + C + 真实 GPU 新回归 | 11 passed，0 skipped |
| 既有量化精度回归 | 11 passed，0 skipped |
| misc DQL 语义选择 | 3 passed，0 skipped，21 deselected |
| numerical harness 非 GPU 门禁 | 80 passed |
| DQL 真实 CLI | 3 个计划各 3/3；9 samples；abs/rel 误差均为 0；RC 0；stderr 空 |

新回归使用 ONNX 1.21 `ReferenceEvaluator`，并检查 C 与 CUDA 的输出值、dtype、shape 和 float32 scale 位模式。harness 对实际有限、非空、全零输入自动启用独立 oracle；C 与 CUDA 即使一致返回错误的 `1`、`0` 或相差 1 ULP，也会失败。

## 完整 pytest

命令见 [full_pytest.command.txt](full_pytest.command.txt)，执行结果为 `561 passed, 1 skipped in 21.47s`，RC 0，stderr 空。唯一跳过项为：

`tests/test_operator_activation_semantics.py:115: Celu does not support float16 in ONNX17`

该跳过项为既有 ONNX17 Celu float16 能力限制，与本修复无关。原始证据见 `full_pytest.stdout.txt`、`full_pytest.stderr.txt`、`full_pytest.rc.txt` 及起止时间文件。

## 完整数值验证

唯一一次完整执行使用 `python -u tools/cli.py numerical --iterations 3 --skip-plots`，未重建二进制。预检确认源提交、两个二进制哈希和 178 个逐文件 CUDA verifier 库存均与预期一致。

| 指标 | 实际结果 |
|---|---:|
| 注册计划 | 725 |
| 每计划迭代 | 3 |
| 通过计划 | 725 |
| 通过迭代 | 2175 |
| Python RC | 0 |
| Wrapper RC | 0 |
| stderr | 空 |
| UTC 时间 | 2026-09-11T05:24:09Z 至 2026-09-11T05:37:47Z |

命令、范围、预检、真实 Python RC、wrapper RC、CUDA 库存、stdout、stderr 和元数据分别保存在 `full_numerical.*.txt`。DynamicQuantizeLinear 汇总为 9 samples，99 分位绝对误差和相对误差均为 0。

## 结论

修复通过 fresh C/CUDA 构建、独立 ReferenceEvaluator 回归、harness 误判防护、完整 pytest 和完整 CPU/CUDA 数值门禁。验证范围仍限定为有限、非空、值级全零输入；empty、NaN、infinity 未扩展，非零 subnormal range 的 scale 下溢继续走既有 `scale=1` fallback。

## 主工作区复验

`main` 已从 `45fc62b926324375d5247e3a59dc665a8673887c` fast-forward 到本验证源提交。主目录 fresh C 构建哈希与隔离构建一致；DQL CUDA 二进制仅因 `.strtab` 中 nvcc 临时文件进程编号产生 3-byte 差异，大小、build ID、section headers 与源文件均一致。主目录新 Reference/C/GPU 回归为 11 passed，DQL CLI 的 3 个默认计划各 3/3、9 samples、abs/rel 误差均为 0，两者 RC 0 且 stderr 空。详见 [MAIN_SYNC.md](MAIN_SYNC.md)。
