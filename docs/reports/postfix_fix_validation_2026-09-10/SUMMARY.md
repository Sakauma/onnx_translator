# 第二轮缺陷修复验收摘要（2026-09-10）

八项审计缺陷已修复，独立审查无剩余阻断，隔离 worktree 的最终 CPU、CUDA 和数值门禁全部通过。生产源码与测试基线为 `c5882ba`；门禁 HEAD 为 `f084e92`，后一个提交只增加独立审查报告。

## 修复范围

1. DynamicQuantizeLinear 按公开 float32 scale 计算量化值与 zero point。
2. omitted Reduce axes 保留 `noop_with_empty_axes` identity 语义。
3. 零长度 Scan 返回初始 state 和全部空 scan outputs。
4. If 按 ONNX `TypeProto` 保留 sequence/optional 容器。
5. If、Loop、Scan、SequenceMap 子图继承完整父模型 domain/opset map。
6. numerical/verify_all 拒绝非正 iterations，关闭 `0/0` 假通过。
7. 有符号/无符号整数在普通、TopK、Split、Unique 路径均精确比较，并检查 wire 范围与 NPS dtype。
8. CUDA runner 使用逐调用隔离目录，严格检查主输出/sidecar/退出诊断并可靠清理，实际 GPU 并发已验证。

## 最终结果

| 门禁 | 结果 |
|---|---:|
| CPU `verify_all --skip-cuda --keep-artifacts` | RC 0；482 passed，11 skipped；10/10 步骤通过 |
| CUDA 全量重编 | RC 0；178 compiled，0 skipped，inventory 178 |
| fresh CUDA protocol | RC 0；14 passed，0 skipped |
| 实际 GPU 并发 Add + Unique 四输出 | RC 0；1 passed，0 skipped |
| 原生 numerical 三轮 | RC 0；723/723 plans，2169/2169 iterations，失败/异常 0 |

CPU 的 11 个 skip 包含 10 个因 `--skip-cuda` 清空 cache 而跳过的 GPU case，以及 1 个既有 Celu float16 兼容性 skip；重编后 10 个 GPU case 均执行通过。旧“34 passed”属于旧版 protocol+architecture 合并口径，当前真实命令为 `14 + 1`；architecture 已在 CPU 482 项中执行。

环境为 WSL `ubuntu2004`、Python 3.12.12、ONNX 1.21.0、NumPy 2.4.6、GCC 9.4.0、CUDA 12.4、RTX 4060 driver 610.88。全部动态命令只在 `/mnt/d/workspace/onnx_translator_bugfix_worktree` 执行，主目录生成物和未跟踪审计材料未被清理或覆盖。

完整命令、SHA、RC、原始日志链接和限制见 [VALIDATION.md](VALIDATION.md)。主目录未同步且没有 push；同步后执行前必须至少 `make` 重建 C 库，需要 CUDA 验证时再运行 `python tools/cli.py compile-cuda`，或直接使用本次已验收的隔离 worktree。
