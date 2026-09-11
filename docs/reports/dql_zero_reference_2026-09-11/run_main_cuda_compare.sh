#!/usr/bin/env bash
set -u
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
main=/mnt/d/workspace/onnx_translator/cache/verify_dynamic_quantize_linear
iso=/mnt/d/workspace/onnx_translator_bugfix_worktree/cache/verify_dynamic_quantize_linear
report=/mnt/d/workspace/onnx_translator_bugfix_worktree/docs/reports/dql_zero_reference_2026-09-11/main_cuda_bounded_compare.txt
{
  echo 'sha256:'; sha256sum "$main" "$iso"
  echo 'size:'; stat -c '%n %s bytes' "$main" "$iso"
  echo 'cmp:'; cmp -l "$main" "$iso" | head -20 || true
  echo 'main notes:'; readelf -n "$main" || true
  echo 'isolated notes:'; readelf -n "$iso" || true
  echo 'section headers diff:'; diff -u <(readelf -SW "$iso") <(readelf -SW "$main") || true
  echo 'source token hashes:'; strings "$main" | grep -E 'dynamic_quantize|y_scale|zero_point' | sort | sha256sum; strings "$iso" | grep -E 'dynamic_quantize|y_scale|zero_point' | sort | sha256sum
} > "$report" 2>&1