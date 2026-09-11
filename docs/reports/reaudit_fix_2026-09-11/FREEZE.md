# Source freeze

## Commit

- Branch: `codex/postfix-bugfix-20260910`
- Baseline parent: `f253f634dd251f348f432e62e318faf25198fd30`
- Frozen source/test commit: `8a6f46985ad16265c5abc4283984f62ba98be2e1`
- Subject: `fix: enforce ONNX boundary semantics`

Immediately after the commit, the isolated worktree had no tracked worktree diff and an empty index. The main repository remained at baseline `f253f634dd251f348f432e62e318faf25198fd30` with no tracked worktree or index diff. Untracked audit and validation reports were deliberately retained and excluded from the source commit.

## Exact committed files

Production code:

1. `nn/operators/common.py`
2. `nn/operators/sequence_optional_control.py`
3. `tensor_ops/tensor_ops_dtype.h`
4. `tensor_ops/tensor_ops_dynamic_quant.c`
5. `tensor_ops/tensor_ops_quantize_linear.c`
6. `tools/numerical/runner.py`
7. `tools/numerical/runner_special_outputs.py`
8. `tools/numerical/output_contracts.py`

Persistent regression tests:

9. `tests/test_reaudit_control_boundaries.py`
10. `tests/test_reaudit_quantization_precision.py`
11. `tests/test_reaudit_special_output_contracts.py`

`git diff --cached --check` passed before the commit. No generated binary, cache file, prior audit evidence, or report file was staged.

## Final CUDA oracle snapshot

After the first complete numerical run exposed two stale QuantizeLinear CUDA-reference comparisons, a separately reviewed scoped follow-up was committed:

- Final source HEAD: `b8d90098176f0bec35c1ddea334781697f9881b7`
- Parent source/test fix: `8a6f46985ad16265c5abc4283984f62ba98be2e1`
- Subject: `fix: align CUDA quantization precision modes`
- Exact files:
  1. `cuda/verify_quantize_linear.cu`
  2. `tools/numerical/runner_cuda_params.py`
  3. `tests/test_reaudit_quantization_cuda.py`

`git diff --cached --check` passed before this commit. Immediately afterward the tracked worktree and index were clean. `tensor_ops.so` remained unchanged at SHA-256 `602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66`. The singly rebuilt `cache/verify_quantize_linear` SHA-256 is `0c12b072e454d8e80aeefa3ac9af1ea3f7a9d03850a69e67bc0b849d2c7bf8be`; comparison with the first-run 178-file inventory shows it is the only changed verifier binary, leaving the other 177 unchanged.

The source history is intentionally two commits so the original seven-finding product/test fix and the subsequently exposed CUDA oracle protocol correction remain independently reviewable.
