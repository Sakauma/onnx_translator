<!--
/**
  ******************************************************************************
  * @file        release.md
  * @author      Egor Izmaylov
  * @brief       记录发布级门禁、性能基线、内存安全和 CUDA CI 要求。
  * @details     2026.06.27  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
-->

# 发布级工程化门禁

本项目的 9/10 工业级目标不只要求测试通过，还要求发布元数据、官方 ONNX 覆盖、C 后端主路径、性能基线、内存安全和 CUDA 数值验证都能被自动证明。

## 发布预检总入口

发布前最后一轮建议运行聚合预检：

```bash
make PYTHON=$PYTHON release-preflight
```

该命令会依次运行 `onnx-semantic-matrix`、`release-check`、全量 pytest、`model-smoke`、`benchmark-smoke-report`、`benchmark-baseline-check`、`package-smoke`、`release-artifacts` 和 `sanitize`，并写入 `result/release_preflight.json` 记录每个 gate 的命令、状态和耗时，同时生成 `result/release_dashboard.md` 和 `result/release_dashboard.json` 汇总 sanitizer、manylinux、性能和 CUDA 证据。仪表盘会列出每个 gate 的本地命令、返回码、耗时、关键 artifact 状态以及 CI/workflow 配置 token，方便区分“本次已跑”“本次计划但未跑”和“仅 CI 已配置”。

带 CUDA 的机器可以额外运行：

```bash
$PYTHON tools/release_preflight.py --include-cuda-smoke
```

固定 CUDA runner 或发布验收机上可以把完整 CUDA 数值 gate 也纳入预检：

```bash
$PYTHON tools/release_preflight.py --include-cuda-full
```

固定性能 runner 上可以把严格性能基线 gate 纳入同一份仪表盘：

```bash
$PYTHON tools/release_preflight.py --include-fixed-runner-perf
```

带 Docker 的 Linux/WSL 环境可以额外把 manylinux wheel 构建纳入预检：

```bash
$PYTHON tools/release_preflight.py --include-manylinux
```

发布验收或固定构建机上应运行完整 manylinux 矩阵：

```bash
$PYTHON tools/release_preflight.py --include-manylinux-full
```

调试预检计划但不执行命令：

```bash
$PYTHON tools/release_preflight.py --dry-run --json /tmp/release_preflight_plan.json
```

已有 preflight JSON 时，可以单独刷新仪表盘：

```bash
make PYTHON=$PYTHON release-dashboard
$PYTHON tools/release_dashboard.py --preflight result/release_preflight.json
```

CI 的 `release-readiness` job 可通过 PR、main/master push 或 `workflow_dispatch` 运行，并会生成一份远端证据入口：`result/release_preflight_plan.json` 使用 `--dry-run` 记录本地、CUDA、manylinux 和固定性能 runner 的完整发布门禁计划；`result/release_dashboard.md` 与 `result/release_dashboard.json` 把该计划、workflow 配置、关键 artifact 路径和长期趋势证据窗口汇总成表；三者连同 checklist、`docs/release_trend_manifest.json` 与 `docs/release_trend_history.json` 作为 `release-evidence-${{ github.run_id }}` artifact 上传并保留 90 天。发布验收时应把这份远端 artifact 与 CUDA、Wheels、Performance workflow 的实际 run 链接一起归档。

发布验收时使用 [`docs/release_evidence_checklist.md`](release_evidence_checklist.md) 汇总本地命令、dashboard 产物、manylinux/CUDA/performance 外部 runner 证据和 CI artifact 链接。任何没有在本地执行的重门禁都必须在 checklist 中记录替代 CI run 或明确标为发布阻断项。

[`docs/release_trend_manifest.json`](release_trend_manifest.json) 是长期证据契约，声明 release evidence、full CUDA、fixed-runner performance 和 manylinux full wheel 四类趋势窗口的 workflow、artifact 名称、保留周期和必备 payload。[`docs/release_trend_history.json`](release_trend_history.json) 是当前历史样本快照，记录每个窗口已经归档的成功 run、artifact digest、样本数和下一步缺口。`release-check` 会校验 manifest、history 与 workflow 配置一致；顶级发布判定至少需要 3 次同一窗口的历史 run 作为趋势样本，失败项必须记录 failed gate、run URL、commit SHA、owner、root cause、resolution 和 follow-up。

外部 CUDA、Performance 或 manylinux full workflow 跑完后，可用下面的命令从 GitHub Actions API 刷新历史快照；未设置 `GITHUB_TOKEN` 时会使用 GitHub 公共 API，但更容易触发匿名 rate limit。

```bash
make PYTHON=$PYTHON release-trend-history-refresh
```

该命令会按 manifest 中的 artifact pattern 自动筛选成功 run。manylinux full 只有在同一次 workflow run 同时保留 `cp310`、`cp311`、`cp312` 三个 manylinux artifact 时才计入 full-matrix 趋势样本。

## 发布就绪检查

```bash
export PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python
make PYTHON=$PYTHON release-check
```

`release-check` 会检查：

- `pyproject.toml` 中的包名、版本、依赖和 CLI 入口。
- 当前安装 ONNX 最新默认域名称级覆盖必须完整。
- `docs/onnx_semantic_matrix.json` 必须与当前代码重新生成的官方 ONNX 语义矩阵一致，且所有 latest 默认域官方算子都必须达到 `verified` 状态。
- 普通数值/张量算子不能存在 Python-only 主运行路径。
- `tensor_ops.h` 与 `tensor_ops_*.c` 的公共 C ABI 声明/实现必须一致。
- `docs/abi_manifest.json` 必须与当前 `tensor_ops/tensor_ops.h` 的公开 ABI 一致。
- CUDA verifier 数量必须与 active numerical 唯一算子覆盖一致。
- `tensor_ops/tensor_ops_*.c` 后端 shard 单文件行数不能超过 400 行，避免重新退化成难维护的大文件。
- `cibuildwheel` manylinux 配置、wheelhouse 检查入口和发布构建目标必须存在。
- 代表性模型 smoke、性能 smoke、包安装 smoke、sanitizer 和 CUDA smoke 均有可执行入口。

## ONNX 语义矩阵

```bash
make PYTHON=$PYTHON onnx-semantic-matrix
```

该门禁生成 `docs/onnx_semantic_matrix.json` 和 `docs/onnx_semantic_matrix.md`，按当前安装 ONNX 的最新默认 domain schema 逐项记录 import 支持、operator class、运行路径、semantic evidence 和最终状态。矩阵会把官方 deprecated alias 显式映射到项目 canonical class，例如 `Scatter` 由 `ScatterElements` 承载，`Upsample` 由 `Resize` 承载；这些映射必须有 numerical、CUDA、ONNX reference pytest 或 Python orchestration 等强证据，不能只靠 C symbol 存在通过。

## ABI manifest

公开 C ABI 快照保存在 `docs/abi_manifest.json`，记录 `DataType` 枚举、公开 struct 字段和函数签名。检查命令：

```bash
make PYTHON=$PYTHON abi-check
```

只有在明确接受 ABI 变化时才刷新：

```bash
$PYTHON tools/abi_manifest.py --write
```

## 包构建

`setup.py` 的 build hook 会先调用 `make tensor_ops.so`，再把生成的 C runtime 复制到 wheel 中的 `nn/tensor_ops.so`。运行时加载顺序为：

1. `TENSOR_OPS_LIB` 环境变量。
2. 已安装包内的 `nn/tensor_ops.so`。
3. 开发仓库根目录的 `tensor_ops.so`。

因为 wheel 内包含平台相关的 Linux 共享库，发布 wheel 必须是平台 wheel，而不能标记为 `py3-none-any`。`package-smoke` 和 `release-artifacts` 会打开 wheel 检查 `nn/tensor_ops.so`、`Root-Is-Purelib: false`、非 `any` 平台 tag，并拒绝把 `.so` 放进 `.data/purelib` 的布局，避免安装器或 `auditwheel` 把二进制运行时误当成跨平台纯 Python 内容。

发布物安装 smoke test：

```bash
make PYTHON=$PYTHON package-smoke
make PYTHON=$PYTHON release-artifacts
```

该命令会在临时目录构建 wheel、用 `pip --target --no-deps` 安装 wheel，然后从非仓库工作目录导入 `nn`，确认 `nn.TENSOR_OPS_LIB_PATH` 指向安装包内的 `nn/tensor_ops.so`，并执行一个 C 后端 `ADD` 算子。

`release-artifacts` 会构建 sdist 和 wheel，运行 `twine check`，检查 sdist 中包含 `Makefile`、`pyproject.toml`、`setup.py`、`tensor_ops/` C 源、`cuda/` verifier、关键工具脚本和发布文档，然后从 sdist 重新构建 wheel 并执行安装 smoke。CI 会上传 `result/release_artifacts/dist/*`，供发布归档或人工检查。

## Manylinux Wheels

Linux 发布 wheel 由 `cibuildwheel` 构建并交给 `auditwheel` repair，配置集中在 `pyproject.toml` 的 `[tool.cibuildwheel]` 和 `[tool.cibuildwheel.linux]`。当前发布矩阵覆盖 CPython 3.10、3.11、3.12 的 `manylinux2014_x86_64` wheel。PR/push 的 `Wheels` workflow 只构建 `cp312` smoke wheel；定时任务和手工触发运行完整 `cp310/cp311/cp312` 矩阵。`cibuildwheel` 阶段不安装全量 runtime 依赖运行测试，避免 wheel smoke 被 `torch`、`onnx` 等重依赖污染；结构检查由 `manylinux-wheelhouse-check` 负责，真实安装执行由普通 `package-smoke` 覆盖。

本地 Docker 环境可运行：

```bash
make PYTHON=$PYTHON manylinux-wheels
make PYTHON=$PYTHON manylinux-wheelhouse-check
```

默认 Make target 为快速 smoke 构建 `cp312-manylinux_x86_64`。完整矩阵使用专用目标：

```bash
make PYTHON=$PYTHON manylinux-wheels-full
make PYTHON=$PYTHON manylinux-wheelhouse-check-full
```

`manylinux-wheelhouse-check` 会检查 wheelhouse 中的 wheel 是否为平台 wheel，是否包含 `nn/tensor_ops.so`，`Root-Is-Purelib` 是否为 `false`，是否没有 `.data/purelib` 中的共享库，以及 platform tag 是否包含 `manylinux`。`manylinux-wheelhouse-check-full` 还会强制要求 `cp310`、`cp311` 和 `cp312` 三个 Python tag 都存在。

`cibuildwheel.before-build` 必须清理 `build/`、`onnx_translator.egg-info/`、`tensor_ops.so` 和 `tensor_ops_asan.so` 等宿主生成物，确保容器内 wheel 使用 manylinux 工具链重新编译 C runtime，而不是复用开发机上已有的共享库。

## 模型 smoke

```bash
make PYTHON=$PYTHON model-smoke
```

该门禁生成并严格导入三类 deterministic ONNX 模型：

- `vision_cnn`：Conv / BatchNormalization / Pool / GlobalAveragePool / Gemm / Softmax。
- `transformer_block`：MatMul / Transpose / Softmax / residual Add / LayerNormalization。
- `embedding_mlp`：Gather / ReduceMean / Concat / Gemm / Sigmoid。

每个模型都会拒绝 `GenericNode` fallback，用 `Graph.forward_` 验证输出 shape，并用 ONNX reference evaluator 对齐 `Graph.forward` 的真实数值输出。

## 性能基线

```bash
make PYTHON=$PYTHON benchmark-smoke
make PYTHON=$PYTHON benchmark-smoke-report
make PYTHON=$PYTHON benchmark-baseline-check
make PYTHON=$PYTHON benchmark-fixed-runner-check
make PYTHON=$PYTHON benchmark
```

默认基准覆盖 `add`、`matmul`、`conv2d`、`reduce_sum` 和 `softmax`，用于捕捉 C 后端路径退化或明显性能回退。`benchmark-smoke` 使用保守吞吐阈值，适合作为普通 CI runner 上的 PR 门禁；`benchmark-smoke-report` 还会写入 `result/benchmark_smoke.json`，CI 会上传该 JSON 作为性能证据；`benchmark-baseline-check` 使用版本化的 `docs/performance_baseline.json` 做回归检查，并写入 `result/benchmark_baseline_check.json`。该 baseline 是便携发布下限，不等同于固定高性能 runner 的调优基线；`benchmark-fixed-runner-check` 使用 `docs/performance_fixed_runner_baseline.json`，要求 `baseline_kind=fixed_runner` 和稳定 `runner_id` 匹配，并写入 `result/benchmark_fixed_runner_check.json`。`benchmark` 用于开发者或固定 runner 记录完整性能数据。需要保存机器可读结果时：

```bash
$PYTHON tools/benchmark_runtime.py --repeat 20 --json result/benchmark_runtime.json
```

固定硬件或固定 self-hosted runner 上建议保存 baseline：

```bash
PERF_RUNNER_ID=fixed-linux-x64-perf $PYTHON tools/benchmark_runtime.py --warmup 5 --repeat 20 --baseline-kind fixed_runner --runner-id fixed-linux-x64-perf --write-baseline result/perf_baseline.json
```

后续回归对比：

```bash
PERF_RUNNER_ID=fixed-linux-x64-perf $PYTHON tools/benchmark_runtime.py --warmup 5 --repeat 20 --baseline result/perf_baseline.json --baseline-kind fixed_runner --require-runner-id fixed-linux-x64-perf --require-baseline-kind fixed_runner --max-regression 0.10
```

CI smoke 的等价命令：

```bash
$PYTHON tools/benchmark_runtime.py --smoke --warmup 1 --repeat 3
$PYTHON tools/benchmark_runtime.py --smoke --warmup 1 --repeat 3 --json result/benchmark_smoke.json
$PYTHON tools/benchmark_runtime.py --warmup 1 --repeat 3 --baseline docs/performance_baseline.json --max-regression 0 --json result/benchmark_baseline_check.json
PERF_RUNNER_ID=fixed-linux-x64-perf $PYTHON tools/benchmark_runtime.py --warmup 5 --repeat 20 --baseline docs/performance_fixed_runner_baseline.json --baseline-kind fixed_runner --require-runner-id fixed-linux-x64-perf --require-baseline-kind fixed_runner --max-regression 0.10 --json result/benchmark_fixed_runner_check.json
```

固定硬件上可以添加更严格的自定义阈值：

```bash
$PYTHON tools/benchmark_runtime.py --min-throughput add=100000000
```

## 内存安全门禁

```bash
make PYTHON=$PYTHON sanitize
```

该目标构建 `tensor_ops_asan.so`，启用 AddressSanitizer 和 UndefinedBehaviorSanitizer，并通过 `TENSOR_OPS_LIB` 指向该动态库运行 C 后端回归子集。默认还会在同一个 sanitizer 进程环境中运行 `tools/model_suite.py`，让 CNN、Transformer block 和 Embedding MLP 的组合路径也经过 ASan/UBSan 与 ONNX reference 数值对齐。`detect_leaks=0` 用于避免 Python 解释器和第三方扩展带来的非项目泄漏噪声；越界、use-after-free、未定义行为仍会使门禁失败。

## CUDA CI

`.github/workflows/cuda.yml` 面向带 CUDA 的 self-hosted Linux runner，默认标签为 `self-hosted`, `linux`, `x64`, `cuda`。该 workflow 通过定时任务和手工触发运行，避免 PR 在没有可用 self-hosted runner 时长期排队。需要 CUDA smoke 证据时运行：

```bash
make PYTHON=python verify-cuda-smoke
```

该 smoke gate 会把 `--op` 过滤条件同时传给 CUDA 编译和 numerical 验证，默认只编译并运行 `add`、`matmul`、`conv2d`、`softmax`、`quantize_linear` 和 `dequantize_linear` 等关键算子。`tools/cli.py compile-cuda` 会跳过比源文件、公共 `.cuh` 头和编译脚本更新的 cached executable；需要强制重编时使用 `--force`。
CUDA smoke 和 full Make target 会向 `tools/verify_all.py` 传入 `--keep-artifacts`，保证成功运行后 `cache/verify_*` 仍可被 workflow 上传为 CUDA evidence artifact。

定时任务和手工触发会运行完整 CUDA gate：

```bash
make PYTHON=python verify-cuda-full
```

它不传 `--op`，因此会编译全部 CUDA verifier，并执行 CUDA-backed numerical 正确性验证。没有 GPU 的普通 GitHub-hosted runner 继续运行 CPU、release 和 sanitizer 门禁。
