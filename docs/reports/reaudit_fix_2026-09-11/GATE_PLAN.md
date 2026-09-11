# Full gate plan

## Fixed environment

All Python commands use WSL `ubuntu2004`, interpreter `/home/sakauma/data/miniconda3/envs/egor/bin/python`, and:

```text
PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin
```

Every gate records its exact command, stdout, stderr, return code, source commit, and relevant binary hashes. Results remain pending until their commands actually finish.

## Ordered execution

1. Snapshot frozen commit `8a6f46985ad16265c5abc4283984f62ba98be2e1`, confirm tracked/index cleanliness, and record the existing `tensor_ops.so` SHA-256.
2. Numeric owner runs a forced CUDA rebuild from the frozen commit. The command is:

   ```text
   /home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py compile-cuda --force
   ```

   Success requires all 178 verifier targets to compile. Pre/post executable hashes and timestamps distinguish this fresh forced build from the earlier cache.
3. After forced compilation completes, the control owner runs the full pytest suite once so GPU tests consume the freshly compiled verifiers:

   ```text
   /home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests
   ```

4. The control owner runs graph entry points against the existing `onnx_model/model.onnx` without invoking `verify_all.py`:

   ```text
   /home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py graph-logic --model ./onnx_model/model.onnx --task-name reaudit_fix_graph_logic_20260911
   /home/sakauma/data/miniconda3/envs/egor/bin/python tools/cli.py verify-graph --model ./onnx_model/model.onnx --task-name reaudit_fix_verify_graph_20260911 --no-clean
   ```

5. After pytest and graph gates finish, the integration owner runs the complete native C-versus-CUDA numerical inventory with three samples per plan:

   ```text
   /home/sakauma/data/miniconda3/envs/egor/bin/python -u tools/cli.py numerical --iterations 3 --skip-plots
   ```

GPU-bearing gates run serially to avoid shared-device and sidecar interference. No `make`, `make clean`, `verify_all.py`, cache deletion, or duplicate full pytest run is planned.

## Evidence already available before the full gate

The validator-targeted run passed `83` tests with no skip. It included cached-verifier GPU execution rather than only mocks:

- `test_dynamic_quantize_gpu_exact_ties_to_even_fixture`;
- five parameter cases of `test_reduce_log_sum_exp_gpu_nonfinite_and_stable_semantics`;
- `test_unique_gpu_writes_four_exact_outputs`;
- `test_unique_sidecar_write_failure_is_nonzero_and_names_path`;
- `test_real_gpu_add_and_unique_multioutput_cleanup`, including concurrent Add and Unique sidecars.

It also executed `test_add_without_visible_gpu_fails_with_diagnostic`, which deliberately hides the device and requires a nonzero diagnostic. Because no test skipped, the nine GPU-capable success/failure protocol cases above reached their expected outcomes using the previously compiled cache. This is valid targeted evidence, but it does not replace the planned `compile-cuda --force` plus full pytest against freshly rebuilt verifiers, and it is not a native full numerical inventory.

## Acceptance

Final acceptance requires zero return codes for forced 178/178 CUDA compilation, full pytest, both graph CLI commands, and the full three-iteration numerical inventory. Counts and any skips will be copied from raw logs after execution; this plan does not predeclare them as passing.

## Execution disposition

The planned gates completed. Final source `b8d90098176f0bec35c1ddea334781697f9881b7` passed 537 pytest cases with one documented Celu/ONNX17 compatibility skip, both graph commands, and the final 723-plan numerical inventory at 3/3 iterations per plan. The first numerical run's two stale QuantizeLinear CUDA-oracle failures and RC 1 remain preserved; after the scoped oracle correction and review, attempt2 passed 723/723 with RC 0. See `VALIDATION.md` and `ACCEPTANCE.md` for the evidence-backed final disposition.
