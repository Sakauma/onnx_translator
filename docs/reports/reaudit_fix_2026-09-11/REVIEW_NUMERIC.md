# REAUD-001/002 independent numeric review

## Scope and status

This is a static cross-review of the current uncommitted C changes in `tensor_ops_dynamic_quant.c`, `tensor_ops_quantize_linear.c`, and the narrowly required FLOAT16 codec correction in `tensor_ops_dtype.h`, together with `tests/test_reaudit_quantization_precision.py`. No build or test command was run.

## Finding and disposition

### FLOAT16 materialization inherits an incorrect subnormal rounding boundary

The new QuantizeLinear precision path materializes FLOAT16 operands and quotients through `float16_to_float(float_to_float16(value))`. The existing `float_to_float16` codec returns signed zero whenever its computed `shift >= 24`. For float32 inputs in the exponent bin around the binary16 minimum-subnormal midpoint, that early return discards the guard and sticky information. Values strictly above `2^-25` must round to binary16 bits `0x0001`, while the current codec returns `0x0000`.

This affects supported QuantizeLinear paths whenever the default scale dtype or explicit `precision` selects FLOAT16 and an operand or quotient reaches that boundary.

The numeric implementation owner corrected the early return from `shift >= 24` to `shift > 24`. Consequently, `shift == 24` now reaches the existing guard/sticky/LSB round-to-nearest-even logic: exact `2^-25` ties to zero, while the next float32 value away from zero rounds to binary16 minimum-subnormal bits `0x0001` (and the corresponding signed forms). A through-QuantizeLinear regression checks converted bits `[0x0000, 0x0001, 0x8000, 0x8001]` and final int8 output `[0, 1, 0, -1]`. Static re-review confirms that this resolves the reported boundary without changing larger FLOAT16 encodings.

## Test adequacy

The DQL bit-exact fixture directly locks the original one-ULP scale error and changed uint8 output. The QuantizeLinear default FLOAT16 fixture directly locks the original result (`29` versus required `30`). The normal DQL cases preserve the existing zero-range choice (`scale=1`) without claiming that the unresolved external specification disagreement has been settled.

The initial explicit FLOAT16 operand-conversion fixture did not distinguish the final C output from the old float32 path. It was replaced with `x=3435.3032` and `scale=981.6252`: float32 division rounds to `3`, while converted FLOAT16 operands produce a FLOAT16 quotient of `3.5` and the required ties-to-even result `4`. Static re-review confirms that the replacement locks the intended operation ordering.

## Semantics reviewed

The current precision selector correctly uses the scale dtype when `precision=0` and recognizes ONNX FLOAT16, FLOAT, DOUBLE, and BFLOAT16 attribute values. The explicit DOUBLE path reads operands as double; FLOAT uses float; the intended FLOAT16/BFLOAT16 paths materialize operands and quotient before nearest-even integer rounding. Subject to correction of the FLOAT16 codec boundary, this structure matches the reviewed opset-24 division-precision requirement.

The DQL change uses float extrema, float `fminf`/`fmaxf`, a separately materialized float range and scale, and the same float scale for zero point and output quotient. It preserves the pre-existing zero-range fallback of `scale=1` and does not silently adopt either disputed external zero-range interpretation.

## Validation status

The static review has no remaining finding. The coordinated shared-library build completed with SHA-256 `602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66`. The later `83 passed` run covered the special-output verifier and existing non-GPU verifier protocol suites; it did not execute `tests/test_reaudit_quantization_precision.py`, so this review does not record a runtime pass for REAUD-001/002. Numeric runtime validation remains assigned to the coordinated numeric test run.
