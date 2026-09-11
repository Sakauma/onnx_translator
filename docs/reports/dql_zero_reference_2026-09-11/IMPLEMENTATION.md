# DynamicQuantizeLinear all-zero ReferenceEvaluator compatibility

## Compatibility choice

For a non-empty FLOAT input whose elements are all finite `+0.0` or `-0.0`, the C backend and CUDA verifier now publish `y_scale = float32(1 / 255)`, `y = uint8(0)`, and scalar `y_zero_point = uint8(0)`. This intentionally matches ONNX 1.21 `ReferenceEvaluator`; it is a compatibility choice rather than a normative opset-17 requirement.

## Guarded scope

The special case uses an explicit all-zero predicate. Any nonzero or nonfinite element clears it, and an empty input does not satisfy it. Positive and negative nonzero constants continue through the normal min/max formula. A nonzero range whose float32 division underflows to zero continues to use the existing `scale = 1` fallback. This change does not claim new support for empty, NaN, or infinite inputs.

## Changes and validation

- `tensor_ops/tensor_ops_dynamic_quant.c` and `cuda/verify_dynamic_quantize_linear.cu` implement the same guarded behavior.
- Existing all-zero expectations now use `float32(1 / 255)`.
- `tests/test_dynamic_quantize_zero_reference.py` derives zero, constant, nondegenerate, and precision-fixture expectations independently from an actual ONNX 1.21 `ReferenceEvaluator`. It exercises the real C wrapper and CUDA verifier protocols, checks output dtype, shape, and values exactly, and compares the scale's float32 bit pattern. A separate regression preserves the scale-underflow fallback.

Fresh C and targeted CUDA builds succeeded. The CUDA command compiled one verifier and skipped none. The new Reference/C/GPU suite passed 11 tests with zero skips, the existing quantization precision suite passed 11 tests with zero skips, and the three DynamicQuantizeLinear tests selected from the miscellaneous semantics suite passed with zero skips. Exact commands, output streams, return codes, source identity, and binary hashes are recorded in `build/` and summarized in `BUILD.md`.
