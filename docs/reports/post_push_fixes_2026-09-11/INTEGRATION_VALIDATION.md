# Post-push fixes integration validation

## Frozen source

- Branch: `codex/post-push-fixes-20260911`
- Initial implementation commit: `fa095a914ac71ef05650806f76f813b94ac8c198`
- Final source commit: `e0a03c6b3e71ba5c9562624f783c3ac6e4165322`
- Workspace: `/mnt/d/workspace/onnx_translator_bugfix_worktree`
- Environment: WSL `ubuntu2004`, fixed egor Python and PATH

## Build

The authorized `run_integration_build.sh` recorded environment versions, ran `make`, then invoked the complete real `compile-cuda` command without deleting cache or manufacturing cache identity. Make and CUDA compilation have separate commands, stdout, stderr, and return-code files. The shared library hash, every verifier hash, artifact sizes, and verifier count are recorded. After the final LayerNormalization CUDA protocol adjustment, its single verifier was force-rebuilt once; the other 177 outputs were reused.

| Gate | Result |
|---|---|
| Environment capture | RC 0; Python 3.12.12, NumPy 2.4.6, ONNX 1.21.0, nvcc 12.4, RTX 4060 Laptop GPU |
| `make` | RC 0; stderr empty; `tensor_ops.so` SHA-256 `e6a720b448b03113da11e7310ce15bcdd326c026c2147d2aabe187b577c046a9` |
| Complete `compile-cuda` | RC 0; `compiled=178 skipped=0`; stderr empty; 178 verifier hashes recorded |
| LayerNormalization targeted rebuild | RC 0; `compiled=1 skipped=0`; stderr empty; verifier SHA-256 `cc31ed88f8509fb01812b9a1e641ecd912b43d94c987f9df14e36ed9195dd074` |
| Replacement C-only `make` after shard split | RC 0; stderr empty; final `tensor_ops.so` SHA-256 `6f35ec47feff1a5e833030548a6a6642b7160d698d789ffecae80f2b7202de89` |

The final CUDA inventory contains 178 verifier binaries and remained unchanged during the C-only replacement build. The LayerNormalization verifier remains `cc31ed88f8509fb01812b9a1e641ecd912b43d94c987f9df14e36ed9195dd074`; the inventory manifest SHA-256 is `0ee9df10a281dd557d1615a5781a980de0c79943a5b7853dc8f7471830a2ed8a`. Raw build evidence is in [the initial make log](integration_make.stdout.txt), [the full CUDA log](integration_compile_cuda.stdout.txt), [the targeted verifier rebuild](layernorm_recompile.stdout.txt), and [the replacement make log](fixup_make.stdout.txt).

## Importer graph gate

`run_graph_gate.sh` created a checker-valid Resize-17 model only under `/tmp`, with the repaired defaults omitted, and invoked `tools/cli.py verify-graph` with an explicit model path and a HEAD-scoped unique task name. It passed `--no-clean` and refused to run if its result directory already existed, so it could not delete or overwrite an existing result. This gate checks strict import, graph construction, declared-output shape validation, and visualization separately from numerical validation.

| Gate | Result |
|---|---|
| Resize-17 `verify-graph` | RC 0; strict import, graph construction, shape validation, and visualization passed; stderr empty |

## Targeted validation

| Owner/domain | Result |
|---|---|
| Resize defaults and empty Slice | 7 passed; RC 0; [stdout](shape_targeted.stdout.txt) |
| LayerNormalization independent C/GPU gate | 13 passed; RC 0; live FLOAT32/FLOAT16/DOUBLE CUDA checks; [stdout](evidence/numeric/layernorm_targeted.stdout.txt) |
| Post-shard normalization and release budget | 30 passed; RC 0; original normalization shard 322 lines and new shard within budget; [stdout](evidence/numeric/layernorm_post_shard_targeted.stdout.txt) |
| CUDA cache, dtype validation, and numerical protocol | 25-owner gate plus the corrected pre-existing diagnostic regression, 26 checks total; all passed; [owner gate](evidence/tooling/targeted_pytest.stdout.txt) |
| LayerNormalization numerical CLI | 9 plans × 3 iterations = 27 samples; zero reported error; RC 0; stderr empty; [stdout](evidence/tooling/layernorm_numerical.stdout.txt) |

## Full validation

The first frozen implementation commit exposed three integration failures. The owners corrected the existing integer diagnostic compatibility, moved the new native implementation into `tensor_ops_layer_norm.c` so both C shards meet the 325-line release limit, and replaced the stale float64 LayerNormalization expectation with the formal FLOAT32-stage-one/FLOAT64-stage-two oracle without relaxing tolerance. These corrections were committed separately as final source commit `e0a03c6b3e71ba5c9562624f783c3ac6e4165322`.

| Gate | Result |
|---|---|
| Full pytest attempt 1 | RC 1; 598 passed, 3 failed, 1 skipped in 22.56s; stderr empty; [stdout](final_pytest.stdout.txt) |
| Full pytest attempt 2 | RC 0; 601 passed, 1 skipped in 20.72s; stderr empty; [stdout](final_pytest_attempt2.stdout.txt), [metadata](final_pytest_attempt2.meta.txt) |
| Full numerical, three iterations | RC 0; 726/726 plans, 2,178/2,178 iterations; no failure, crash, or skip; stderr empty; [stdout](final_numerical.stdout.txt), [metadata](final_numerical.meta.txt) |

The three attempt-1 failures remain preserved in [final_pytest.stdout.txt](final_pytest.stdout.txt): one stale numerical-harness error-message assertion, one pre-fix LayerNormalization mixed-precision expectation, and the C shard line-budget gate when the original normalization file reached 381 lines. No numerical command ran after that failure. After the reviewed fixup and replacement C-only build, attempt 2 passed; its only skip is the established `tests/test_operator_activation_semantics.py:115` Celu float16 limitation in ONNX17.

The successful full numerical run used final source `e0a03c6b3e71ba5c9562624f783c3ac6e4165322`, native hash `6f35ec47feff1a5e833030548a6a6642b7160d698d789ffecae80f2b7202de89`, the unchanged 178-verifier inventory, and three iterations per plan. Its raw output contains 726 plan headers and 726 `Pass (3/3)` results.

## Main workspace delivery

The main workspace passed its separately owned delivery gate. It was cleanly fast-forwarded from baseline `96d070e6f1a26f61ee0cf727c9616148b8680629` to final source `e0a03c6b3e71ba5c9562624f783c3ac6e4165322`. A fresh native build reproduced `tensor_ops.so` SHA-256 `6f35ec47feff1a5e833030548a6a6642b7160d698d789ffecae80f2b7202de89`; one forced LayerNormalization CUDA rebuild succeeded with `compiled=1 skipped=0` while the verifier inventory remained at 178.

The main-workspace focused pytest passed 20 tests. Its LayerNormalization and Resize numerical gates passed 27/27 and 15/15 iterations respectively, with every command returning RC 0 and empty stderr. The main verifier hash differs from the isolated build; the evidence does not assign a cause, and the verifier was not repeatedly rebuilt to chase a byte-identical hash. Full commands, identities, and raw evidence are recorded in [MAIN_SYNC.md](MAIN_SYNC.md).

## Constraints

No cache cleanup, `verify_all.py`, environment installation, or push was performed. The main-workspace synchronization and focused validation are documented separately in [MAIN_SYNC.md](MAIN_SYNC.md); the original isolated build evidence above remains unchanged.
