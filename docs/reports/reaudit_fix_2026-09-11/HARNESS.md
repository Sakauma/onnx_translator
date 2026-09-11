# REAUD-007 harness remediation

## Change

Special-output verification now checks raw NPS/C outputs and CUDA outputs against their schema or wire contracts before any lossy cast, rounding, clipping, boolean conversion, or reshape. DynamicQuantizeLinear retains its float32 packed CUDA wire protocol and validates the uint8-bearing fields before decoding. TopK and Unique typed sidecars remain int64. Dropout retains its uint8 CUDA mask wire and validates binary values before converting it to bool.

## Persistent test design

`tests/test_reaudit_special_output_contracts.py` contains the eight audit fault injections through the real `verify_op` decision path, legal controls for DynamicQuantizeLinear, TopK, Unique, and Dropout, and a CLI aggregation check that requires exit code 1 for a failed plan.

## Validation status

The coordinated C build completed before this targeted run. The loaded `tensor_ops.so` SHA-256 was `602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66`, matching the build owner's supplied digest.

The first targeted run exposed two compatibility defects in the new checks: a legacy direct TopK fixture omitted the explicit K input, and the DQL missing-CUDA-output test was intercepted by NPS contract validation before infrastructure error propagation. The owned runner changes now use schema-derived TopK shape whenever K is present while retaining the legacy fixture fallback, and defer DQL schema validation until after the CUDA no-output check while still keeping it before all coercion. The complete targeted set was then rerun.

Final targeted command (WSL ubuntu2004):

```text
PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin /home/sakauma/data/miniconda3/envs/egor/bin/python -m pytest -q tests/test_reaudit_special_output_contracts.py tests/test_numerical_harness_regression.py tests/test_numerical_runner_architecture.py tests/test_numerical_runner_nps.py tests/test_cuda_verifier_protocol.py
```

Final result: `83 passed in 10.80s`, exit code `0`, no skipped tests, and empty stderr. The exact command, stdout, stderr, and return code are stored beside this report as `validator_targeted.command.txt`, `validator_targeted.stdout.txt`, `validator_targeted.stderr.txt`, and `validator_targeted.rc.txt`.

No CUDA executable, rebuild, full numerical inventory, or full repository gate was run.
