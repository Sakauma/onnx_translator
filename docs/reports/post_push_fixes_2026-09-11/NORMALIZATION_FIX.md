# LayerNormalization stash precision and optional-output fix

## Contract

ONNX LayerNormalization-17 uses `stash_type` for the stage-one reduction and normalization precision. The schema permits FLOAT (`1`) and BFLOAT16 (`16`) for this temporary type. With `stash_type=1`, conversion to float32 and every stage-one arithmetic result must materialize in float32. The normalized value is then cast back to the input type before scale and bias are applied.

An omitted optional output retains its positional slot. For outputs `["y", "", "inv_std"]`, the operator-level result is `(y, None, inv_std)`, allowing `Graph.forward` and `Graph.forward_` to bind output index 2 correctly.

## Root cause

The existing C and CUDA paths accumulated LayerNormalization statistics in double precision. For `[[1e8, 1e8, 1e8 + 8]]`, this produced `[-0.7071065, -0.7071065, 1.4142131]` instead of the float32-stash result `[0, 0, 1.7320505]`. Even `[1, 2, 3]` differed from ONNX ReferenceEvaluator by one ULP.

The Python implementation used NumPy reductions, which did not express the schema's required per-operation stash materialization. It also removed empty output names from the returned tuple, shifting later optional outputs to the wrong indices.

## Implementation

The old native entry points remain unchanged for ABI compatibility. Two new symbols were declared in `tensor_ops/tensor_ops.h` and implemented in the dedicated `tensor_ops/tensor_ops_layer_norm.c` shard:

- `layer_norm_float_stash_forward`
- `layer_norm_float_stash_multi_output_forward`

`nn/operators/normalization_ops.py` configures and calls these symbols only for FLOAT stash with float32 or float64 input. The native implementation casts input values to float32 for stage one, materializes sum, mean, difference, square, variance, square root, reciprocal, and normalization in float32, and writes auxiliary statistics as FLOAT. Float64 input casts the normalized result back to double before the double scale/bias stage. Float16 and bfloat16 inputs use the corrected Python fallback.

The Python fallback performs explicit sequential accumulation and materializes every stage-one operation in the selected stash type. BFLOAT16 uses round-to-nearest-even conversion after each operation. It casts the normalized result to the input type before scale and bias and materializes low-precision affine results. Unsupported direct `stash_type` values now raise `ValueError`.

`cuda/verify_layer_normalization.cu` consumes the coordinated numerical-runner payload of seven int32 values:

```text
row_count, normalized_size, has_scale, has_bias, emit_stats, stash_type, input_dtype
```

`input_dtype` uses ONNX type codes FLOAT=`1`, FLOAT16=`10`, DOUBLE=`11`, and BFLOAT16=`16`; the payload ends with float32 epsilon. Although verifier files transport numeric arrays as doubles, this field preserves the product type `T`: stage one materializes according to `stash_type`, while Normalized, Mul, and Add materialize according to `input_dtype`. The verifier rejects unsupported stash and input types, so CUDA no longer acts as a double-precision oracle for this operator.

Both numerical and shape-only operator paths preserve empty optional-output positions.

## Regression coverage

`tests/test_post_push_layernorm_semantics.py` covers:

- strict opset-17 import and full ONNX checker validation;
- ReferenceEvaluator and independent float32-stage formulas;
- the large-offset regression and ordinary `[1, 2, 3]` bit patterns;
- calls through both new native symbols after a fresh build;
- FLOAT stash statistic dtypes and reduction shapes;
- direct and graph-level optional output slot preservation;
- float16 and bfloat16 fallback storage behavior;
- rejection of unsupported stash types.
- direct execution of the freshly compiled CUDA verifier for ordinary and large-offset FLOAT32 fixtures, comparing independently with both ReferenceEvaluator and the explicit float32 formula;
- direct CUDA statistic-output comparison with the same two independent oracles.
- direct FLOAT16 and DOUBLE CUDA checks with nontrivial scale/bias, validating that stage two follows input type `T` while statistics remain FLOAT.

QA's numerical plans cover the CUDA parameter layout, FLOAT32 normal cases, emitted statistics, and the large-offset fixture. The current default numerical plan scope for this protocol is FLOAT32; the explicit `input_dtype` field prevents the double wire representation from being confused with product dtype and supports separately validated low-precision and DOUBLE plans.

## Validation status

The root coordinator's shared native build succeeded, and the updated LayerNormalization CUDA verifier was then rebuilt as the sole forced target. The dedicated regression ran against the fresh native library and GPU executable:

```text
/home/sakauma/data/miniconda3/envs/egor/bin/python -u -m pytest -q tests/test_post_push_layernorm_semantics.py
13 passed in 3.68s
```

The CUDA FLOAT32 results are bit-exact against ReferenceEvaluator and the independent sequential formula. FLOAT16 and DOUBLE stage-two results are exact against the independent two-stage formula and within `1e-3` and `2e-7`, respectively, of ONNX 1.21 ReferenceEvaluator. That evaluator vectorizes stage one and returns auxiliary outputs in `T` rather than schema `U`; the tests therefore use it as a tolerance-based secondary value oracle for those cases while requiring exact FLOAT statistics from the independent formula.

Evidence is stored in `evidence/numeric/layernorm_targeted.command.txt`, `layernorm_targeted.stdout.txt`, `layernorm_targeted.stderr.txt`, and `layernorm_targeted.rc.txt`.
