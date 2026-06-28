<!--
/**
  ******************************************************************************
  * @file        release_evidence_checklist.md
  * @author      Egor Izmaylov
  * @brief       记录发布验收证据清单。
  * @details     2026.06.28  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/
-->

# Release Evidence Checklist

发布候选版本只有在每一类证据都能定位到具体命令、产物或 CI run 时，才视为完成发布级验收。无法在本地执行的重门禁必须记录对应 CI workflow、runner 要求和未本地执行原因，不能只写“环境不支持”。

## 基本信息

- [ ] 记录 release candidate commit SHA、branch、tag 或 artifact 版本号。
- [ ] 记录 Python、GCC、CUDA、Docker、ONNX、NumPy、Torch 和 ml_dtypes 版本。
- [ ] 确认发布验收开始前工作树无非预期脏文件；生成物仅允许出现在 `result/`、`wheelhouse/`、`cache/`、`onnx_model/` 或 `dist/` 等已知目录。

## 核心本地门禁

- [ ] `make PYTHON=$PYTHON check` 通过。
- [ ] `python -m pytest -q tests` 通过，并记录总数、skip 数和失败数。
- [ ] `make PYTHON=$PYTHON release-check` 通过，记录 ONNX latest 覆盖、C runtime 覆盖、CUDA verifier / numerical 覆盖和默认计划数量。
- [ ] `python tools/verify_all.py --skip-cuda` 通过，并确认 `docs/reports/operator_coverage.md` 未被 gate 自动改写。
- [ ] `make PYTHON=$PYTHON model-smoke` 通过，证明代表性 CNN、Transformer block 和 embedding MLP 模型可导入、推形和数值对齐。

## 发布预检与仪表盘

- [ ] `python tools/release_preflight.py --json result/release_preflight.json` 通过，且 `dry_run=false`。
- [ ] `result/release_dashboard.md` 和 `result/release_dashboard.json` 已生成。
- [ ] dashboard 中 sanitizer、performance smoke/baseline、package smoke、release artifacts 的状态为 `passed`。
- [ ] 对 dashboard 中状态为 `planned`、`configured` 或 `missing` 的 gate，记录对应 CI run、runner 要求或发布阻断原因。

## 包与 manylinux 证据

- [ ] `make PYTHON=$PYTHON package-smoke` 通过，证明 wheel 内含 `nn/tensor_ops.so` 且不是 pure Python wheel。
- [ ] `make PYTHON=$PYTHON release-artifacts` 通过，证明 sdist/wheel metadata、sdist 源码完整性、twine check 和从 sdist 重建 wheel smoke。
- [ ] `make PYTHON=$PYTHON manylinux-wheels` 与 `make PYTHON=$PYTHON manylinux-wheelhouse-check` 通过，或记录 Wheels workflow 的 `manylinux-smoke` run。
- [ ] 发布验收运行 `make PYTHON=$PYTHON manylinux-wheels-full` 与 `make PYTHON=$PYTHON manylinux-wheelhouse-check-full`，或记录 Wheels workflow 的 `manylinux-full` run，并确认 `cp310`、`cp311`、`cp312` 均有 manylinux wheel。

## 性能与安全证据

- [ ] `make PYTHON=$PYTHON benchmark-smoke-report` 通过并保留 `result/benchmark_smoke.json`。
- [ ] `make PYTHON=$PYTHON benchmark-baseline-check` 通过并保留 `result/benchmark_baseline_check.json`。
- [ ] 固定 runner 上运行 `make PYTHON=$PYTHON benchmark-fixed-runner-check`，或记录 Performance workflow 的 `fixed-runner-baseline` run。
- [ ] `make PYTHON=$PYTHON sanitize` 通过，证明 ASan/UBSan C 后端回归子集和代表性模型路径均通过。

## CUDA 证据

- [ ] `make PYTHON=$PYTHON verify-cuda-smoke` 通过，或记录 CUDA workflow 的 `cuda-smoke` run。
- [ ] 发布验收运行 `make PYTHON=$PYTHON verify-cuda-full`，或记录 CUDA workflow 的 `cuda-full` run。
- [ ] 如果本地只跑 smoke，必须在 dashboard 或发布记录中说明 full CUDA 由 self-hosted CUDA runner 覆盖。

## 归档

- [ ] 归档 `result/release_preflight.json`、`result/release_dashboard.md`、`result/release_dashboard.json`。
- [ ] 归档 `result/benchmark_smoke.json`、`result/benchmark_baseline_check.json`，以及固定 runner 性能结果。
- [ ] 归档 release artifact 目录、manylinux wheelhouse 或对应 CI artifacts。
- [ ] 在 PR 或 release note 中粘贴本 checklist 的完成状态和所有外部 CI run 链接。
