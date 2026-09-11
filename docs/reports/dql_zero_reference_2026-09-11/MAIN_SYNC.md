# 主工作区同步与定向验证

## 同步预检

主工作区 `D:\workspace\onnx_translator` 位于 `main`，HEAD 为 `45fc62b926324375d5247e3a59dc665a8673887c`，tracked worktree 与 index 均干净。现有未跟踪内容仅位于既有报告路径，与本修复新增文件无碰撞。Ancestry 检查确认该 HEAD 是 `a1d4940cc694ac65fbb77a993b7b800fcafd6ee5` 的祖先。

`git merge --ff-only a1d4940cc694ac65fbb77a993b7b800fcafd6ee5` 成功。未覆盖、删除或复制任何既有未跟踪报告。

## 主工作区构建

在 WSL `ubuntu2004` 中使用固定 egor Python/PATH：

| 命令 | 结果 |
|---|---|
| `make` | RC 0，`Build successful: tensor_ops.so` |
| `python -u tools/cli.py compile-cuda --op dynamic_quantize_linear --force` | RC 0，`compiled=1 skipped=0` |

主工作区 `tensor_ops.so` SHA-256 为 `4f94df5ccbe2df3b1df054733265d1a57d7bae993a854d92745b3973d248a0e6`，与隔离工作区完全一致。

主工作区 DQL CUDA verifier SHA-256 为 `2d9b851d46f87dd9322ca2e0e4f2eeabdd559cfbad38e18a8f8b36542c2326ab`，隔离工作区为 `2c39a72d9b79686a184405f641a4345850bbfebbf4f7a761ab15383ffb911d2e`。有界比较确认：大小同为 1,012,520 bytes，GNU build ID 同为 `06fa795de052a26c69e4ad5612373836474c64ec`，section headers 无差异，源文件 SHA-256 同为 `546c568b31a80b56b98429e876eefe32fef8ee480120e311122793ecef385203`。整个二进制仅 3 bytes 不同，位于 `.strtab` 内 nvcc 临时文件名的进程编号（`00000433` 与 `000003ff`）。

## 主工作区真实验证

| 门禁 | 结果 |
|---|---|
| `tests/test_dynamic_quantize_zero_reference.py` | 11 passed in 4.79s，RC 0，stderr 空 |
| DQL numerical CLI，3 plans × 3 iterations | 三项均 3/3；9 samples；abs/rel 误差均为 0；RC 0，stderr 空 |

两项验证均从主工作区加载 fast-forward 后的源码、fresh `tensor_ops.so` 和 fresh DQL CUDA verifier，确认新 C/GPU 行为及默认新增 numerical 计划实际生效。按授权未重复运行全套 pytest、完整 numerical、图验证或全量 CUDA 编译。

原始命令、stdout、stderr、RC、起止时间、二进制哈希/大小和有界比较分别保存为本目录下的 `main_*` 文件。
