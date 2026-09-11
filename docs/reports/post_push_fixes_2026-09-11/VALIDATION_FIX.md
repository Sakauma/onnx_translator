# Validation tooling fixes

Date: 2026-09-11

Branch: `codex/post-push-fixes-20260911`
Baseline: `96d070e6f1a26f61ee0cf727c9616148b8680629`

## Implemented changes

### PP-TOOL-002: compiler-aware CUDA verifier cache

`tools/commands/compile_cuda.sh` now stores one compiler identity per verifier below `cache/.compile-identities/`. The identity contains the resolved compiler path, `cksum` of the selected executable, and its `--version` output. A cached verifier is fresh only when its sources/headers/script remain older and the stored identity exactly matches the current compiler.

The metadata directory does not match `cache/verify_*`, so artifact upload and executable discovery retain their existing behavior. Identity files are written to a unique temporary file and atomically renamed after a successful compilation. Before any recompile starts, the prior identity is removed, so a compiler that partially overwrites its output and then fails cannot make that damaged output look fresh on the next invocation. Existing caches without identity metadata rebuild once. `--force` and `--op` behavior remain intact.

Regression coverage verifies same-compiler cache reuse, switching compiler paths, replacing a compiler binary in place while its version text remains unchanged, and retrying after a forced compiler failure partially overwrites the executable. All tests use isolated fake compilers and temporary cache directories; no shared or real CUDA cache was changed.

### PP-TOOL-003: ordinary NPS output dtype contract

`tools/numerical/runner.py` now checks every ordinary output's actual NumPy dtype before scalar normalization, decoding, quantization, or numeric comparison. It uses `nn.DTYPE_TO_NUMPY`, preserving the project's physical storage conventions such as `bfloat16 -> uint16` and float8 types using `uint8`.

Regression coverage exercises accepted storage dtypes for bool, float16, float32, float64, and bfloat16, plus a wrong dtype for each. Wrong dtypes now make the plan fail with an output-contract diagnostic. Existing special-output handlers remain responsible for their own schemas, and the excluded bool-wire tolerance candidate was not changed.

### LayerNormalization numerical protocol integration

To match the native/CUDA LayerNormalization fix owned by the numeric workstream, the CUDA parameter payload now contains seven int32 fields—`row_count`, `normalized_size`, `has_scale`, `has_bias`, `emit_stats`, `stash_type`, and the ONNX input dtype code—followed by float32 epsilon. The input dtype remains independent of stash precision so CUDA can materialize stage two in X type. Missing `stash_type` defaults to ONNX value 1.

The default numerical inventory now includes the audited large-offset, small-variance float32 case, and LayerNormalization input preparation honors an explicit `input_values` fixture. Regression tests verify the payload layout and the new plan's exact input values. A bounded real NPS-path gate takes the plan through input preparation, Tensor construction, LayerNormalization construction, and forward execution; `input_values` is removed before operator construction, and single-output LayerNormalization now supplies the explicit `outputs=["y"]` slot required by its output contract.

## Validation

Command and raw logs are under `evidence/tooling/targeted_pytest.*`.

```text
python -m pytest -q tests/test_compile_cuda.py tests/test_post_push_validation_contracts.py tests/test_numerical_runner_cuda_params.py
...................                                                      [100%]
25 passed in 2.58s
```

`git diff --check` completed without errors; PowerShell reported only the repository's line-ending conversion warning for `tests/test_compile_cuda.py`.

The tooling workstream itself did not run the full suite, mutate the shared cache, clean, install, commit, stage, or push. Coordinated integration subsequently completed the real CUDA rebuild and LayerNormalization numerical run with the matching seven-int parameter parser; those results are recorded below.

## Final integration acceptance

After the native and CUDA LayerNormalization work was rebuilt, the coordinated CUDA diff was reviewed read-only. The seven-int parser accepts only stash types 1/16 and input type codes 1/10/11/16. Stage one uses only `stash_type`; stage two casts Normalized and materializes Mul/Add according to input T, including float16 round-to-nearest, bfloat16 RNE, float32, and double. The dedicated GPU tests execute the verifier for float16 and float64 with nontrivial scale/bias and compare against both ONNX ReferenceEvaluator and an independent two-stage oracle; emitted statistics remain float32.

The actual CLI gate then ran all nine active LayerNormalization plans for three iterations each, including the large-offset fixture and float16/bfloat16 plans. All 27 iterations passed with zero reported error:

```text
python tools/cli.py numerical --op layer_normalization --iterations 3 --skip-plots
LAYER_NORMALIZATION ... Samples 27, Abs 0.00e+00, Rel 0.00e+00
```

Raw evidence is in `evidence/tooling/layernorm_numerical.*`; exit code was 0. The final tooling regression rerun completed `25 passed in 2.58s`, also with exit code 0, in `evidence/tooling/targeted_pytest.*`.
