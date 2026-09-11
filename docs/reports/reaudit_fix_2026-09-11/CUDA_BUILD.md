# Fresh CUDA verifier 构建记录

## 正式验收运行

- HEAD：`8a6f46985ad16265c5abc4283984f62ba98be2e1`（parent `f253f634dd251f348f432e62e318faf25198fd30`）
- 分支：`codex/postfix-bugfix-20260910`
- 工作树：`/mnt/d/workspace/onnx_translator_bugfix_worktree`
- 环境：WSL distro `ubuntu2004`；`PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin`
- CUDA：`nvcc 12.4, V12.4.99`
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`，driver `610.88`
- 原生命令：`python -u tools/cli.py compile-cuda --force`
- 日志包装：绝对路径 `tee` 分别保存 stdout/stderr，并启用 Bash `pipefail`，因此 tool exit code 保留原生命令失败。
- 实际 tool exit code / RC：`0`
- stdout：`cuda_compile.stdout.txt`，180 行
- stderr：`cuda_compile.stderr.txt`，0 bytes
- 产物 mtime 时间窗：`2026-09-11T12:01:18.636116+08:00` 至 `2026-09-11T12:04:14.236414+08:00`
- footer：`CUDA verifier compilation succeeded. compiled=178 skipped=0`
- 构建后库存：178 个唯一 `cuda/verify_*.cu`；178 个同名 executable；missing `[]`；orphan `[]`
- 构建后 tracked worktree/index：clean

## 首次运行的日志封装错误

首次 `--force` 运行本身也输出 `compiled=178 skipped=0`，但外围 Bash 日志变量被 Windows shell 提前展开为空，导致编译完成后的日志/RC 文件写入失败，外围 tool exit code 为 `1`。该 RC 不代表 nvcc 或 CLI 编译失败。没有清理 cache；随后仅重跑一次，以本页所列第二次运行作为正式证据。没有第三次重编。
