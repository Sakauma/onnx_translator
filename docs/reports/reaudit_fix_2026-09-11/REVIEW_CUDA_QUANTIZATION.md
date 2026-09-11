# QuantizeLinear CUDA oracle independent review

## Scope

Static review covered the scoped changes in `cuda/verify_quantize_linear.cu`, the QuantizeLinear branch of `tools/numerical/runner_cuda_params.py`, and `tests/test_reaudit_quantization_cuda.py`. The review did not modify source or run tests, compilation, or numerical validation.

## Verdict

PASS with no blocking finding.

The patch corrects the CUDA reference protocol exposed by the first full numerical run. It does not change the expected C result or widen the product fix.

## Precision protocol

The second parameter word now has four explicit meanings:

- mode 0: double division, preserving the prior `use_float_math=0` path;
- mode 1: float32 division, preserving the prior `use_float_math=1` path;
- mode 2: FLOAT16 round-to-nearest conversion of both operands and the quotient;
- mode 3: BFLOAT16 round-to-nearest conversion of both operands and the quotient.

Modes 0 and 1 retain their prior arithmetic order. Modes 2 and 3 use CUDA `__float2half_rn` and `__float2bfloat16_rn`, decode the materialized values back to float for the physical division, materialize the quotient again, and only then apply `rintf` for integer outputs. This matches the independently reviewed opset-24 division-precision stages.

When `precision=0`, parameter encoding selects the mode from `dtypes[1]`, the scale dtype. Explicit ONNX precision values DOUBLE (11), FLOAT (1), FLOAT16 (10), and BFLOAT16 (16) override that default. This fixes the previous boolean encoding that collapsed FLOAT, FLOAT16, and BFLOAT16 into one float32 path. The DequantizeLinear branch returns before this encoding and retains its existing parameter layout.

The CUDA file wire remains double. Inputs entering this verifier have already been numerically decoded; all float32, FLOAT16, and BFLOAT16 values are exactly representable in double, so the transport does not erase the low-precision payload value before the kernel performs the selected materialization.

## Regression strength

The FLOAT16 GPU fixture uses exact half payloads from the original REAUD-002 case. It requires mode 2 to produce 30 while legacy float32 mode 1 and double mode 0 produce 29, so the old CUDA kernel fails the new expectation.

The BFLOAT16 GPU fixture constructs exact BFLOAT16 payloads independently: `49664 / 616` is about `80.623`, whose BFLOAT16-materialized quotient is `80.5`. Ties-to-even therefore produces 80 in mode 3, while float32 mode 1 produces 81. This distinguishes quotient materialization rather than merely asserting agreement with the C backend.

The parameter tests separately cover default FLOAT16 and BFLOAT16 selection and explicit FLOAT, FLOAT16, and DOUBLE overrides. Together with the direct mode fixtures, they cover the runner-to-wire decision and the kernel arithmetic without using the product output as the oracle.

## Runtime status

Runtime compilation and targeted GPU results are owned by the numeric executor and must be taken from that executor's raw evidence. This static review does not predeclare them successful.
