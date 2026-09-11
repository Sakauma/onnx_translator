# 主目录 source 同步与重建结果

## Fast-forward

执行前主目录 `D:\workspace\onnx_translator` 位于 `main`，HEAD 为 `f253f634dd251f348f432e62e318faf25198fd30`，tracked worktree 与 index 均干净。`f253f634...` 是目标 `b8d9009...` 的 ancestor；14 个 source commit 写入路径与 109 个既存未跟踪路径没有碰撞。

执行：

```text
git merge --ff-only b8d90098176f0bec35c1ddea334781697f9881b7
```

结果为 fast-forward 成功。主 HEAD 现为 `b8d90098176f0bec35c1ddea334781697f9881b7`；tracked worktree 与 index 仍干净，109 个既存未跟踪路径保留。

## 主 C shared library 重建

- 固定环境命令见 `main_make.command.txt`。
- 开始：`2026-09-11T12:43:00.7761960+08:00`
- 结束：`2026-09-11T12:43:35.2705404+08:00`
- RC：0；stderr 为空；stdout footer：`Build successful: tensor_ops.so`
- 主产物大小：1016144 bytes
- 主产物 SHA256：`602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66`
- 隔离产物 SHA256：`602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66`

两处 hash 完全一致。

## 主 QuantizeLinear verifier 单目标重建

- 固定环境命令见 `main_compile_ql.command.txt`。
- 开始：`2026-09-11T12:43:51.5048547+08:00`
- 结束：`2026-09-11T12:43:58.9801177+08:00`
- RC：0；stderr 为空；stdout footer：`CUDA verifier compilation succeeded. compiled=1 skipped=0`
- 只选择并强制重建 `cuda/verify_quantize_linear.cu -> cache/verify_quantize_linear`；没有重建其他 177 个 verifier。
- 主产物大小：1051952 bytes
- 主产物 SHA256：`27079beec490af52644f5092b933e615487eafe6bdedcc63dac65367ee69cfa7`
- 隔离产物大小：1051952 bytes
- 隔离产物 SHA256：`0c12b072e454d8e80aeefa3ac9af1ea3f7a9d03850a69e67bc0b849d2c7bf8be`

两个 binary 的源码 SHA256 都是 `6ccaff3ab0e48ce588d60b027e41f01c7b48d90ee0d5403af1d6979e5fb82c89`，ELF Build ID 都是 `0856c3069fbb815534bc98ce21ce9982c362b955`。`cmp -l` 只报告 1 个字节差异，offset 960903 位于 `.strtab` 范围；这是 nvcc/linker 生成符号字符串元数据差异的有界证据。未反复重建、复制或覆盖 binary；功能验收以下述真实 GPU fixture 为准。

## 主目录真实 C/GPU 定向回归

- 命令见 `main_quant_targeted.command.txt`。
- 测试文件：`tests/test_reaudit_quantization_precision.py`、`tests/test_reaudit_quantization_cuda.py`
- 开始：`2026-09-11T12:45:13.5212470+08:00`
- 结束：`2026-09-11T12:45:23.5206343+08:00`
- 结果：`20 passed in 3.80s`，0 skipped，0 failed，RC 0，stderr 为空。

测试从主目录运行，因而加载主目录刚重建的 `tensor_ops.so` 与 `cache/verify_quantize_linear`。C 精度边界、CUDA float16/bfloat16 division mode 和 runner 参数编码 fixtures 均通过。

## 保存证据与边界

三条命令各自的 `.command.txt`、`.stdout.txt`、`.stderr.txt`、`.rc.txt`、`.start.txt`、`.end.txt` 均保存在本目录。没有执行 clean、`verify_all`、全量 pytest、图 CLI、全 178 编译或 numerical；除 `tensor_ops.so` 与 `cache/verify_quantize_linear` 外，没有重建、删除或覆盖 cache、模型、result、旧报告。

报告同步阶段尚未执行。等待 harness 定稿并提交报告后，再经明确授权将主目录从 source commit fast-forward 到 final report commit。
