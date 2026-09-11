# 主目录同步与定向重建计划（阶段一已执行；报告同步待完成）

最终门禁成功并经根 agent 明确授权后，source fast-forward、主目录定向重建和定向加载回归已执行，实际结果见 `MAIN_SYNC_RESULT.md`。报告提交与第二次 fast-forward 尚未执行；未 push。

## 固定对象

- 主目录：`D:\workspace\onnx_translator` / WSL `/mnt/d/workspace/onnx_translator`
- 隔离修复目录：`D:\workspace\onnx_translator_bugfix_worktree`
- 当前主分支预期：`main`
- 当前主 HEAD 预期：`f253f634dd251f348f432e62e318faf25198fd30`
- 最终 source commit：`b8d90098176f0bec35c1ddea334781697f9881b7`
- 固定 WSL/Python/PATH：与 `CPU_GATE_RESULT.md` 一致。
- 隔离最终 `tensor_ops.so` SHA256：`602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66`
- 隔离最终 `cache/verify_quantize_linear` SHA256：`0c12b072e454d8e80aeefa3ac9af1ea3f7a9d03850a69e67bc0b849d2c7bf8be`

## 第一阶段：同步 source commit

1. 只读复核主目录仍位于 `main`、HEAD 仍为 `f253f634...`，`git diff --quiet` 与 `git diff --cached --quiet` 均为 RC 0。列出未跟踪文件并保存；它们属于用户，必须保留。
2. 比较 `f253f634...` 到 `b8d9009...` 的将写入路径与主目录未跟踪路径，若存在同路径碰撞则停止并报告，不移动或删除文件。
3. 确认 ancestry：`git merge-base --is-ancestor f253f634... b8d9009...` 必须 RC 0。
4. 在主目录执行且只执行 fast-forward：

   ```text
   git merge --ff-only b8d90098176f0bec35c1ddea334781697f9881b7
   ```

5. 核对主 HEAD 精确等于 `b8d9009...`，tracked worktree/index 仍干净。此阶段不把隔离目录的未跟踪修复报告复制到主目录。

## 第二阶段：主目录定向重建

在主目录固定 WSL 环境中顺序执行：

```text
cd /mnt/d/workspace/onnx_translator && PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin make
```

```text
cd /mnt/d/workspace/onnx_translator && PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin /home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda --op quantize_linear --force
```

单目标 CLI 已由 `tools/commands/compile_cuda.sh` 核实：`--op quantize_linear` 只选择 `cuda/verify_quantize_linear.cu`，`--force` 强制重建对应 `cache/verify_quantize_linear`；预期 footer 为 `compiled=1 skipped=0`。不运行全 178 编译，不修改其他 cache artifact。

两条命令分别保存 command/stdout/stderr/RC/start/end。随后记录：

- 主 HEAD；
- `tensor_ops.so` 大小、mtime、SHA256；
- `cache/verify_quantize_linear` 大小、mtime、SHA256；
- 其他 cache、模型、result 和旧报告均保持原位。

若主目录产物 hash 与隔离产物不同，先比较编译器版本、命令、源 SHA、ELF metadata/构建可重复性，不能仅因 hash 不同判失败或覆盖隔离产物。

## 第三阶段：主目录定向加载回归

只验证主目录真实加载刚构建的 C shared library 与 QuantizeLinear GPU binary，不重复全量门禁：

```text
cd /mnt/d/workspace/onnx_translator && PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin /home/sakauma/data/miniconda3/envs/egor/bin/python -u -m pytest -q -ra tests/test_reaudit_quantization_precision.py tests/test_reaudit_quantization_cuda.py
```

前一文件覆盖 REAUD-001/002 的 C runtime 精度边界；后一文件是 `b8d9009...` 新增的真实 QuantizeLinear CUDA precision-mode 与参数编码回归。记录 passed/skipped/failed、所有 skip 原因和 RC；不得为了通过放宽断言。

## 第四阶段：报告定稿与第二次 fast-forward

1. 将主目录同步、两项重建、产物 hash 与定向回归的实际证据写入隔离目录 `docs/reports/reaudit_fix_2026-09-11/MAIN_SYNC_RESULT.md`。不先手工复制到主目录。
2. harness 在隔离分支定稿并提交全部新报告，提供精确 report commit SHA；确认该 commit 是 `b8d9009...` 的后代。
3. 再次只读核对主目录 HEAD、tracked worktree/index 和未跟踪路径碰撞，然后执行：

   ```text
   git merge --ff-only <final-report-commit-sha>
   ```

4. 核对主 HEAD 等于 report commit，tracked worktree/index 干净，既有未跟踪报告仍保留。

## 禁止项

不运行 `make clean`、`verify_all.py`、全量 pytest、图 CLI、全 178 CUDA 编译或完整 numerical；不删除 cache、模型、result、旧报告或任何未跟踪文件；不 merge commit、不 rebase、不 reset、不 push。
