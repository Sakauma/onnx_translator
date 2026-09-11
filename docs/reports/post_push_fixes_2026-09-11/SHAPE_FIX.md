# Resize defaults and empty Slice fix

## Scope

This change fixes PP-NATIVE-001's ONNX Resize-17 defaults and PP-GRAPH-002's empty-dimension Slice result. It changes only:

- `nn/importer/node_factories_02.py`
- `nn/operators/shape_extra_ops.py`
- `nn/operators/shape_transform_ops.py`
- `tests/test_post_push_shape_semantics.py`

No shared build, CUDA compilation, full pytest, full numerical run, cleanup, staging, commit, or push was performed.

## Implementation

The Resize importer and direct wrapper now default to the ONNX Resize-17 schema values `coordinate_transformation_mode="half_pixel"` and `nearest_mode="round_prefer_floor"`. Explicit attributes continue to override these defaults.

Slice retains the ONNX Slice-13/17 schema's explicit clipping rules for nonempty dimensions: negative-step starts clamp to `[0, dim-1]` and ends clamp to `[-1, dim-1]`. A shared helper now handles `dim_len == 0` explicitly as a zero-length result for either step direction. `forward` and `forward_` use the same normalized starts, steps, and output shape. The helper also rejects mismatched parameter lengths, repeated/out-of-range axes, and zero steps instead of relying on incidental indexing behavior.

For the nonempty extreme case `x=[0,1,2]`, `starts=ends=INT64_MIN`, `steps=-1`, the formal schema clipping produces `[0]`; both the C path and forced Python fallback return `[0]`, and `forward_` reports shape `(1,)`. ONNX 1.21 ReferenceEvaluator returns `[]` because its Python slicing behavior differs from the schema's stated negative-start clamp. This change deliberately preserves the repository's prior schema-aligned nonempty behavior and limits the functional fix to empty dimensions. The schema text and observed ReferenceEvaluator result are saved in `slice_schema.txt` and `slice_extreme_reference.stdout.txt`.

## Targeted validation

The dedicated test file covers:

- checker-valid, publicly imported opset-17 Resize with omitted coordinate and nearest attributes, compared to ONNX 1.21 `ReferenceEvaluator` through `Graph.forward`;
- the direct Resize wrapper with both defaults omitted;
- checker-valid, publicly imported empty float32 Slice for positive and negative steps;
- C, forced Python fallback, and `Slice.forward_` agreement on both empty cases;
- nonempty positive-step and negative-step controls through the real C backend.
- the formal-schema extreme negative-start control through C, forced Python fallback, and `forward_`.

Final command: `python -u -m pytest -q tests/test_post_push_shape_semantics.py`

Final result: `7 passed in 1.45s`, RC 0.

Evidence: `shape_targeted.command.txt`, `shape_targeted.stdout.txt`, and `shape_targeted.rc.txt`. The first run exposed a local missing `ndim` assignment after refactoring and is preserved as `shape_targeted_attempt1.*`; subsequent intermediate passes are preserved as `shape_targeted_attempt2.*` and `shape_targeted_attempt3.*`. The final run includes public importer/Graph paths for both empty Slice directions and explicit C/fallback/metadata agreement.
