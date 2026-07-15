<!--
/**
  ******************************************************************************
  * @file        README.md
  * @author      Egor Izmaylov
  * @brief       记录 ONNX Translator 相关说明。
  * @details     2026.06.02  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
-->

# ONNX Translator

本项目实现 ONNX 模型导入、图构建、C 后端算子执行，以及 CUDA 参考程序数值验证。

## 文档入口

- [工程架构说明](docs/architecture.md)：说明 ONNXImport、算子层、C 后端、CUDA verifier 和测试框架的关系。
- [工程接手指南](docs/onboarding.md)：说明环境检查、一键门禁、当前关键覆盖数字和常见失败排查。
- [新增算子流程](docs/add_operator.md)：说明新增算子的 Python、ONNX factory、C ABI、CUDA verifier、数值计划和 pytest 覆盖步骤。
- [开发流程清单](docs/development.md)：记录算子开发时需要同步修改和验证的位置。
- [发布级工程化门禁](docs/release.md)：说明发布检查、性能基线、sanitizer 和 CUDA CI。
- [验证总结](docs/reports/verify_summary.md)：记录当前数值验证和图结构验证覆盖范围。
- [算子覆盖报告](docs/reports/operator_coverage.md)：记录 ONNXImport、C 后端、CUDA verifier 和数值计划覆盖情况。

## 后端化边界

普通数值/张量算子的核心计算应优先落在 `tensor_ops/tensor_ops_*.c`，Python 侧主要负责 ONNX 属性解析、shape 推导、ctypes 调度和少量动态语义 fallback。新增或补齐算子时，不应把主体计算长期停留在 `nn/operators/`。

当前 ONNXImport 覆盖的普通数值/张量算子均应具备 C runtime path；审计报告中不应出现 Python-only 待后端化项。复杂算子仍可在 Python 保留等价 fallback，但不能作为唯一运行路径。

## 环境

推荐在 WSL/Linux 中运行。当前验证使用已有环境：

```bash
PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python
```

安装依赖：

```bash
$PYTHON -m pip install -r requirements.txt
```

`requirements.txt` 用于仓库完整开发和验证环境。作为包安装时，核心运行时只需要 `numpy`、`onnx` 和 `ml_dtypes`；按用途安装可选依赖：

```bash
$PYTHON -m pip install .
$PYTHON -m pip install '.[verify]'      # PyTorch 模型生成和 numerical 绘图
$PYTHON -m pip install '.[viz]'         # Graphviz 图可视化
$PYTHON -m pip install '.[dev]'         # pytest、构建和发布工具
```

如果只做导入和图构建验证，可以暂时不安装 CUDA；如果要跑完整数值正确性验证，需要 WSL/Linux 中可用的 `nvcc`。

## 构建与静态检查

环境健康检查：

```bash
$PYTHON tools/health_check.py
```

```bash
make clean
make PYTHON=$PYTHON check
```

`make check` 会编译 `tensor_ops.so`，并对主要 Python 入口做 `py_compile` 检查。当前目标是保持 `-Wall -Wextra` 下无 C 编译警告。

## 常用验证命令

```bash
make PYTHON=$PYTHON check
$PYTHON -m pytest -q tests
$PYTHON tools/cli.py compile-cuda
$PYTHON tools/cli.py numerical --iterations 20 --skip-plots
$PYTHON tools/verify_all.py --skip-cuda
$PYTHON tools/audit_ops.py --strict --output /tmp/onnx_translator_audit.md
make PYTHON=$PYTHON release-check
make PYTHON=$PYTHON onnx-semantic-matrix
make PYTHON=$PYTHON release-preflight
make PYTHON=$PYTHON release-artifacts
make PYTHON=$PYTHON package-smoke
make PYTHON=$PYTHON manylinux-wheels
make PYTHON=$PYTHON manylinux-wheelhouse-check
make PYTHON=$PYTHON manylinux-wheels-full
make PYTHON=$PYTHON manylinux-wheelhouse-check-full
make PYTHON=$PYTHON model-smoke
make PYTHON=$PYTHON benchmark-smoke
make PYTHON=$PYTHON benchmark-smoke-report
make PYTHON=$PYTHON benchmark-baseline-check
make PYTHON=$PYTHON benchmark-fixed-runner-check
make PYTHON=$PYTHON benchmark
make PYTHON=$PYTHON sanitize
```

如果只改 Python 导入或 shape 逻辑，优先运行 `make PYTHON=$PYTHON check` 和 `$PYTHON -m pytest -q tests`。如果改 C 后端或 CUDA verifier，应补充 `$PYTHON tools/cli.py compile-cuda` 和对应 `$PYTHON tools/cli.py numerical --op <op>`。

## 推荐开发流程

1. 修改前运行 `tools/audit_ops.py --strict` 或目标测试，确认当前覆盖基线。
2. Python 算子改动放在 `nn/operators/`，旧入口 `nn/Operators.py` 只做兼容 re-export。
3. ONNX 属性解析放在 `nn/importer/node_factories_*.py`，通过注册表接入。
4. 普通数值算子的核心计算放在 `tensor_ops/tensor_ops_*.c`，共享辅助逻辑放在 `tensor_ops/tensor_ops_internal.h`。
5. 数值验证计划放在 `tools/numerical/cli.py`，测试按领域放进 `tests/test_operator_*.py`。
6. 提交前至少运行 `make PYTHON=$PYTHON check`、`$PYTHON -m pytest -q tests` 和 `tools/audit_ops.py --strict`。

## 一键验证门禁

完整本地门禁会清理旧产物、检查环境、编译 CUDA verifier、构建 C 后端、运行单元测试、执行 strict 算子覆盖审计、生成/验证两个 ONNX 模型，并执行 CUDA 数值正确性验证：

```bash
$PYTHON tools/verify_all.py
```

无 CUDA 的环境可以跑 CPU 门禁；该命令仍会执行 strict 算子覆盖审计，用于证明普通数值/张量算子没有 Python-only 主路径、已实现算子没有遗漏默认 numerical 计划、公共 C ABI 声明/实现一致、当前 ONNX 最新默认域名称级导入无缺口：

```bash
$PYTHON tools/verify_all.py --skip-cuda
```

也可以通过 Makefile 调用：

```bash
make PYTHON=$PYTHON verify-cpu
make PYTHON=$PYTHON verify
```

调试时保留生成物：

```bash
$PYTHON tools/verify_all.py --keep-artifacts
```

## 覆盖审计

刷新算子实现统计报告：

```bash
make PYTHON=$PYTHON audit
```

报告写入 `docs/reports/operator_coverage.md`，会区分实际 C runtime path、合理 Python 调度/元数据算子、当前暂缓后端化算子、当前暂缓深度语义/数值验证算子，以及 CUDA/数值验证覆盖。提交或交接前建议使用 `tools/audit_ops.py --strict`，让覆盖回退直接变成非零退出码。

## 发布、性能和内存安全

发布前运行：

```bash
make PYTHON=$PYTHON release-check
make PYTHON=$PYTHON release-preflight
```

该门禁会确认包元数据存在、当前 ONNX 最新默认域名称级导入无缺口、官方 ONNX 语义矩阵无 weak row、普通数值/张量算子没有 Python-only 主路径、C ABI 声明/实现一致、CUDA verifier 与 active numerical 覆盖一致，并检查 manylinux wheel 构建入口和 `cibuildwheel` 配置。
`onnx-semantic-matrix` 会刷新 `docs/onnx_semantic_matrix.json` 和 `docs/onnx_semantic_matrix.md`，逐项记录最新默认域官方算子的 import、operator class、semantic evidence、deprecated alias 和最终状态。
`release-preflight` 会进一步聚合语义矩阵、全量 pytest、模型 smoke、性能 smoke 报告、性能 baseline 回归检查、包安装 smoke、release artifacts 和 sanitizer，并写入 `result/release_preflight.json` 作为发布前证据。

C ABI 快照检查：

```bash
make PYTHON=$PYTHON abi-check
```

公开 ABI 变化需要显式刷新 `docs/abi_manifest.json`。

发布物安装 smoke test：

```bash
make PYTHON=$PYTHON package-smoke
make PYTHON=$PYTHON release-artifacts
```

`package-smoke` 会验证源码树构建的 wheel 安装后能加载包内 `nn/tensor_ops.so` 并执行 C 后端算子。`release-artifacts` 会构建 sdist/wheel、运行 metadata 检查、确认 sdist 包含 C/CUDA 源码，再从 sdist 重建 wheel 并执行同样的安装 smoke。
由于 wheel 内包含平台相关的 `tensor_ops.so`，发布 smoke 也会拒绝错误的 `py3-none-any` wheel 和 `.data/purelib` 共享库布局，确保发布物按平台 wheel/platlib 语义分发。

Linux manylinux wheel smoke test 需要 Docker：

```bash
make PYTHON=$PYTHON manylinux-wheels
make PYTHON=$PYTHON manylinux-wheelhouse-check
make PYTHON=$PYTHON manylinux-wheels-full
make PYTHON=$PYTHON manylinux-wheelhouse-check-full
```

默认 `manylinux-wheels` 只构建 `cp312-manylinux_x86_64` 快速 wheel；完整发布矩阵使用 `manylinux-wheels-full` 构建 `cp310/cp311/cp312`，并由 `manylinux-wheelhouse-check-full` 强制检查三个 Python tag。`cibuildwheel` 只负责 build/repair，后置 wheelhouse check 负责检查 manylinux tag、`Root-Is-Purelib: false` 和 `.so` 不在 `.data/purelib`。

代表性模型 smoke test：

```bash
make PYTHON=$PYTHON model-smoke
```

该命令生成并严格验证 CNN、Transformer block 和 Embedding MLP 三类 ONNX 模型，覆盖常见发布路径中的导入、initializer、图连接、shape 推导，并用 ONNX reference evaluator 对齐 C 后端真实前向数值。

性能基线：

```bash
make PYTHON=$PYTHON benchmark-smoke
make PYTHON=$PYTHON benchmark-smoke-report
make PYTHON=$PYTHON benchmark-baseline-check
make PYTHON=$PYTHON benchmark-fixed-runner-check
make PYTHON=$PYTHON benchmark
```

`benchmark-smoke` 使用保守吞吐阈值，适合 CI 和 PR 上捕捉 C 后端灾难性回退；`benchmark-smoke-report` 会额外写入 `result/benchmark_smoke.json` 供 CI 上传；`benchmark-baseline-check` 使用 `docs/performance_baseline.json` 做版本化回归检查并写入 `result/benchmark_baseline_check.json`；`benchmark-fixed-runner-check` 使用 `docs/performance_fixed_runner_baseline.json`、`baseline_kind=fixed_runner` 和 `PERF_RUNNER_ID` 做固定机器回归检查；`benchmark` 用于在固定机器上记录更稳定的性能数据。

固定机器上可保存并对比性能 baseline：

```bash
$PYTHON tools/benchmark_runtime.py --write-baseline result/perf_baseline.json
$PYTHON tools/benchmark_runtime.py --baseline result/perf_baseline.json --max-regression 0.15
```

内存安全门禁：

```bash
make PYTHON=$PYTHON sanitize
```

该门禁在 ASan/UBSan 下运行 C 后端回归子集，并额外跑代表性模型 smoke，让真实模型组合路径也进入内存安全检查。

CUDA CI 使用 `.github/workflows/cuda.yml`，面向带 `cuda` 标签的 self-hosted Linux runner。该 workflow 通过定时任务和手工触发运行，避免 PR 在没有可用 self-hosted runner 时长期排队；需要 CUDA 证据时先运行 `make verify-cuda-smoke`，再按需触发完整 CUDA verifier 编译和 numerical 正确性验证。

## 生成 ONNX 模型

使用 PyTorch 导出综合算子模型：

```bash
$PYTHON tools/cli.py create-model
```

生成只覆盖图结构/shape 类算子的模型：

```bash
$PYTHON tools/cli.py create-graph-model
```

模型默认写入 `onnx_model/model.onnx`。

## 图导入和构建验证

严格验证模型导入、图连接、shape 推导和可视化生成：

```bash
$PYTHON tools/cli.py verify-graph --model ./onnx_model/model.onnx --task-name nps_verification
```

默认 strict 模式会在不支持或解析失败的节点上直接失败。只有在明确接受占位节点时才使用：

```bash
$PYTHON tools/cli.py verify-graph --model ./onnx_model/model.onnx --no-strict --allow-generic
```

## CUDA 数值验证

先编译 CUDA 参考程序：

```bash
$PYTHON tools/cli.py compile-cuda
```

可通过环境变量覆盖路径：

```bash
CUDA_DIR=cuda CACHE_DIR=cache NVCC=/path/to/nvcc $PYTHON tools/cli.py compile-cuda
```

运行数值验证：

```bash
$PYTHON tools/cli.py numerical --iterations 20 --skip-plots
```

只跑某个算子：

```bash
$PYTHON tools/cli.py numerical --op add --iterations 5 --skip-plots
```

如果 `cache/verify_<op>` 缺失，数值验证会返回非零退出码。

## 单元测试

```bash
$PYTHON -m pytest -q
```

当前测试覆盖图执行重复调用、图输出返回、ONNXImport strict 失败、GenericNode 降级记录，以及重点算子的导入、shape 推导、Python fallback 和 C 后端路径一致性。
