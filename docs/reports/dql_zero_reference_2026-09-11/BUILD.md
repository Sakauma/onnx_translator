# DynamicQuantizeLinear build and targeted validation

## Source identity

- Worktree: `/mnt/d/workspace/onnx_translator_bugfix_worktree`
- Base commit: `45fc62b926324375d5247e3a59dc665a8673887c`
- Tracked dirty diff SHA256: `dea322fba36f5f3d772721f3b5cf495cca674cd83f7ce13b3d4b0d0b9dccb3ab`
- Source-file manifest SHA256: `6dd5541b419647b4f6de20bea78af60df8c3a0902eb081074942cb0605d4986e`

The tracked dirty diff is saved as `build/tracked_dirty.diff`. `build/source_files.sha256` records the C, CUDA, test, numerical-harness, and policy sources, including untracked source files that an ordinary Git diff does not contain.

After validation, the C, CUDA, and direct regression files still matched this build snapshot. Two harness-owned files had changed concurrently: `tools/numerical/runner_special_outputs.py` and `tests/test_numerical_dql_zero_reference.py`. The later workspace is captured separately as `build/post_validation_tracked_dirty.diff` (SHA256 `ec797ba777915b62a0842c47a8263c4dd9e3aaf4920d502251bea6ecaf0d8693`) and `build/post_validation_source_files.sha256` (manifest SHA256 `1f54debb2274d626bee86af667962c946e317ffd039e3db4a9956d2e31d34158`). Those concurrent harness edits did not participate in the direct C/CUDA/Reference test file reported below.

## Fresh builds

Environment PATH was `/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin` in WSL distro `ubuntu2004`.

| Command | RC | Result |
|---|---:|---|
| `make` | 0 | `Build successful: tensor_ops.so` |
| `/home/sakauma/data/miniconda3/envs/egor/bin/python -u tools/cli.py compile-cuda --op dynamic_quantize_linear --force` | 0 | `compiled=1 skipped=0` |

Built artifact identity:

| Artifact | Size | SHA256 |
|---|---:|---|
| `tensor_ops.so` | 1,016,144 bytes | `4f94df5ccbe2df3b1df054733265d1a57d7bae993a854d92745b3973d248a0e6` |
| `cache/verify_dynamic_quantize_linear` | 1,012,520 bytes | `2c39a72d9b79686a184405f641a4345850bbfebbf4f7a761ab15383ffb911d2e` |

## Targeted tests

| Command | RC | Passed | Skipped | Deselected |
|---|---:|---:|---:|---:|
| `/home/sakauma/data/miniconda3/envs/egor/bin/python -u -m pytest -q tests/test_dynamic_quantize_zero_reference.py` | 0 | 11 | 0 | 0 |
| `/home/sakauma/data/miniconda3/envs/egor/bin/python -u -m pytest -q tests/test_reaudit_quantization_precision.py` | 0 | 11 | 0 | 0 |
| `/home/sakauma/data/miniconda3/envs/egor/bin/python -u -m pytest -q tests/test_operator_misc_semantics.py -k dynamic_quantize_linear` | 0 | 3 | 0 | 21 |

The new suite executed actual ONNX 1.21 `ReferenceEvaluator`, C wrapper, and CUDA verifier paths. It covers signed-zero mixtures across multiple non-empty shapes, exact dtype/shape/value and float32 scale bits, positive and negative constants, a nondegenerate input, the existing precision fixture, and the retained nonzero-range scale-underflow fallback.

An earlier combined selection command is also retained in `build/targeted_pytest.*`; because its `-k` expression filtered the new file's differently named tests, it reported 7 passed and 39 deselected. The three explicit commands above provide the complete requested coverage.

Exact command, stdout, stderr, and return-code files are stored under `build/`. No full pytest, full numerical run, clean, commit, or push was performed.
