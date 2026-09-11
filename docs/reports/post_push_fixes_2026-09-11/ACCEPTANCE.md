# Post-push fixes acceptance

## Status

**ISOLATED WORKTREE PASS** — the corrected source, replacement native build, focused gates, second full pytest, and first full numerical run all pass with consistent source and binary identities. The failed first full pytest is retained as evidence of the two issues corrected before the final gates. Main-workspace delivery validation is tracked separately below and remains pending.

## Reviewed source identity

- Isolated worktree: `/mnt/d/workspace/onnx_translator_bugfix_worktree`
- Initial implementation commit: `fa095a914ac71ef05650806f76f813b94ac8c198`
- Final source commit: `e0a03c6b3e71ba5c9562624f783c3ac6e4165322`
- Final commit subject: `fix: preserve normalization validation contracts`
- Final fixup changed four files after the initial 17-file implementation commit.
- Expected base: `96d070e6f1a26f61ee0cf727c9616148b8680629`

## Completed gates

| Gate | Result | Evidence |
| --- | --- | --- |
| Native `make` | PASS, RC 0; stderr empty | `integration_make.*` |
| Full CUDA compile | PASS, RC 0; `compiled=178 skipped=0`; stderr empty | `integration_compile_cuda.*` |
| CUDA inventory | PASS, 178 verifier binaries | `final_cuda_inventory.meta.txt`, `final_cuda_inventory.sha256` |
| LayerNormalization forced rebuild | PASS, RC 0; `compiled=1 skipped=0`; stderr empty | `layernorm_recompile.*` |
| Graph gate | PASS, RC 0; importer and `forward_` graph construction completed; stderr empty | `graph_gate.*` |
| Shape semantics | PASS, 7 tests | `shape_targeted.*` |
| Tooling contracts | PASS, 25 tests | `evidence/tooling/targeted_pytest.*` |
| LayerNormalization numerical plan | PASS, 9 plans × 3 iterations = 27 samples; zero reported error | `evidence/tooling/layernorm_numerical.*` |
| LayerNormalization independent C/GPU regression | PASS, 13 tests | `evidence/numeric/layernorm_targeted.*` |
| Replacement native `make` | PASS, RC 0; stderr empty | `fixup_make.*` |
| Post-shard normalization regression | PASS, 30 tests; stderr empty | `evidence/numeric/layernorm_post_shard_targeted.*` |
| Full pytest attempt 2 | PASS, RC 0; 601 passed, 1 skip; stderr empty | `final_pytest_attempt2.*` |
| Full numerical | PASS, RC 0; 726/726 plans, 3 iterations per plan, 2,178 samples; no failures, crashes, or skips; stderr empty | `final_numerical.*` |

The dedicated LayerNormalization regression used the fresh native library and final forced-rebuild CUDA verifier. Float32 and float64 product inputs exercise the new native float-stash symbols. Float16 and bfloat16 product inputs exercise the corrected Python fallback. Separate direct verifier tests execute real CUDA for FLOAT32, FLOAT16, and DOUBLE and compare it to ONNX ReferenceEvaluator and an independent staged-precision formula rather than treating either native path as the GPU oracle. The broader numerical output therefore represents a mix of native and Python product paths compared with CUDA; it is not an all-C result.

## Final binary identities

| Artifact | SHA-256 |
| --- | --- |
| `tensor_ops.so` | `6f35ec47feff1a5e833030548a6a6642b7160d698d789ffecae80f2b7202de89` |
| `cache/verify_layer_normalization` | `cc31ed88f8509fb01812b9a1e641ecd912b43d94c987f9df14e36ed9195dd074` |
| Final 178-entry inventory manifest | `0ee9df10a281dd557d1615a5781a980de0c79943a5b7853dc8f7471830a2ed8a` |

The LayerNormalization verifier hash remains the value from `layernorm_recompile.hashes.txt`; the replacement native hash matches `fixup_make.hashes.txt`. The final CUDA inventory contains 178 entries. Full pytest recorded final source commit `e0a03c6b3e71ba5c9562624f783c3ac6e4165322` and the replacement native hash in `final_pytest_attempt2.meta.txt`.

The single full pytest skip is expected: `tests/test_operator_activation_semantics.py:115` records that Celu does not support float16 in ONNX 17.

The failed first full pytest evidence is retained in `final_pytest.*`. It exposed the stale pre-fix float64 LayerNormalization oracle and the 381-line normalization shard. The fixup moved the new implementation to `tensor_ops_layer_norm.c`, reduced the original shard to 322 lines, and replaced the test with a strict staged-precision oracle. The post-fix 30-test gate includes the exact release shard-budget check.

## Final gate reconciliation

| Gate | Required acceptance evidence | Current result |
| --- | --- | --- |
| Corrected native rebuild | RC 0, empty stderr, replacement `tensor_ops.so` hash | PASS |
| Corrected targeted regression | Strict FLOAT-stash float64 assertion and LayerNormalization suite pass against fresh native binary | PASS, 30 tests |
| Full pytest | RC 0, empty stderr, test count, source SHA, binary hashes, explained skips | PASS, 601 passed and 1 expected skip |
| Full numerical | RC 0, empty stderr, plan count, iteration/sample count, skip reasons, source SHA, exact binary identity | PASS, 726/726 plans and 2,178 samples |

`final_numerical.meta.txt` records final commit `e0a03c6b3e71ba5c9562624f783c3ac6e4165322`, native hash `6f35ec47feff1a5e833030548a6a6642b7160d698d789ffecae80f2b7202de89`, 178 verifiers, three iterations per plan, and 726 actual passes from 726 plans. The raw output independently contains 726 plan headers and 726 `Pass (3/3)` results. It contains no failure, crash, or skip line. The preflight inventory is byte-identical to the final inventory manifest, has SHA-256 `0ee9df10a281dd557d1615a5781a980de0c79943a5b7853dc8f7471830a2ed8a`, and records LayerNormalization verifier hash `cc31ed88f8509fb01812b9a1e641ecd912b43d94c987f9df14e36ed9195dd074`.

## Final verdict

Isolated worktree: **PASS**

Main workspace delivery: **PASS**

## Main workspace delivery appendix

The main workspace at `/mnt/d/workspace/onnx_translator` was fast-forwarded to final source commit `e0a03c6b3e71ba5c9562624f783c3ac6e4165322`. Independent inspection of `evidence/main/` and the resulting workspace confirms:

| Gate | Result |
| --- | --- |
| Fast-forward integration | PASS, RC 0 |
| Main native `make` | PASS, RC 0; stderr empty |
| Main `tensor_ops.so` | `6f35ec47feff1a5e833030548a6a6642b7160d698d789ffecae80f2b7202de89`, identical to the accepted isolated artifact |
| LayerNormalization-only CUDA rebuild | PASS, RC 0; `compiled=1 skipped=0`; stderr empty |
| Main CUDA inventory | 178 verifier binaries |
| Shape and LayerNormalization focused pytest | PASS, 20 tests in 4.00s; stderr empty |
| LayerNormalization numerical | PASS, 9 plans × 3 iterations = 27/27; zero reported error; stderr empty |
| Resize numerical | PASS, 5 plans × 3 iterations = 15/15; zero reported error; stderr empty |
| Git state | HEAD matches final source; tracked working tree and index clean; pre-existing untracked reports retained |

The freshly built main LayerNormalization verifier has SHA-256 `2f0dca3994e5ed6f673778e4fa0c6caca2cb9e8d90828740384b31ecf20641ae`, which differs from the isolated verifier hash `cc31ed88f8509fb01812b9a1e641ecd912b43d94c987f9df14e36ed9195dd074`. The evidence establishes fresh compilation from the accepted source, the expected compiler environment, 178-verifier inventory continuity, and successful real GPU execution. No cause is assigned to the binary hash difference, and no rebuild was performed solely to reproduce the isolated hash.

Raw main-workspace commands, outputs, return codes, stderr files, and identity metadata are stored under `evidence/main/`; the delivery narrative is in `MAIN_SYNC.md`.
