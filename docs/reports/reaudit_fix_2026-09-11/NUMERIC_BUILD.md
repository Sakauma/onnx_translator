# REAUD-001/002 C 构建记录

- 最终构建时间：2026-09-11 11:49:31 +0800
- 工作树：`/mnt/d/workspace/onnx_translator_bugfix_worktree`
- 分支：`codex/postfix-bugfix-20260910`
- 基线 HEAD：`f253f634dd251f348f432e62e318faf25198fd30`
- 环境：WSL distro `ubuntu2004`；`PATH=/home/sakauma/data/miniconda3/envs/egor/bin:/usr/local/cuda/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin`
- GCC：`gcc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0`
- Python：`Python 3.12.12`
- 命令：`make`
- 实际 RC：`0`
- 产物：`tensor_ops.so`，size `1016144` bytes，mtime `2026-09-11 11:49:31.986032000 +0800`
- 最终 SHA-256：`602e9c24c5327ca9d466c56a7ee515d903b9615aff4e8404b86f2d91940b0a66`
- 最晚相关源文件 `tensor_ops_dtype.h` mtime 为 `2026-09-11 11:48:50.742988700 +0800`。产物更新时间晚于该文件以及两处量化 `.c` 文件，确认最终源码完成重编译。
- 构建次数：首次量化 `.c` 修改后构建 RC `0`；独立审阅发现 half codec midpoint 后，完成最小 helper 修复并再次运行相同 `make`，最终构建 RC `0`。本页哈希与 mtime 均指第二次最终产物。

## stdout/stderr

```text
Compiling C extension...
gcc -O3 -fPIC -Wall -Wextra -fopenmp -o tensor_ops.so tensor_ops/tensor_ops_reduce_formula.c tensor_ops/tensor_ops_softmax_family.c tensor_ops/tensor_ops_deform_conv.c tensor_ops/tensor_ops_activation_extra.c tensor_ops/tensor_ops_spectral_recurrent.c tensor_ops/tensor_ops_elementwise_misc.c tensor_ops/tensor_ops_shape_grid.c tensor_ops/tensor_ops_embedding.c tensor_ops/tensor_ops_matrix_quant.c tensor_ops/tensor_ops_global_pool.c tensor_ops/tensor_ops_matmul.c tensor_ops/tensor_ops_layout_sequence.c tensor_ops/tensor_ops_conv_pool_roi.c tensor_ops/tensor_ops.c tensor_ops/tensor_ops_pad_crop.c tensor_ops/tensor_ops_trig.c tensor_ops/tensor_ops_group_norm.c tensor_ops/tensor_ops_sort_scan.c tensor_ops/tensor_ops_resize.c tensor_ops/tensor_ops_conv_quant.c tensor_ops/tensor_ops_index_scatter.c tensor_ops/tensor_ops_reduce_arg_misc.c tensor_ops/tensor_ops_spectral_transform.c tensor_ops/tensor_ops_random.c tensor_ops/tensor_ops_rms_norm.c tensor_ops/tensor_ops_recurrent.c tensor_ops/tensor_ops_loss.c tensor_ops/tensor_ops_quantize_linear.c tensor_ops/tensor_ops_core.c tensor_ops/tensor_ops_attention.c tensor_ops/tensor_ops_gather.c tensor_ops/tensor_ops_normalization_loss_random.c tensor_ops/tensor_ops_detection_sampling.c tensor_ops/tensor_ops_window.c tensor_ops/tensor_ops_pool_roi.c tensor_ops/tensor_ops_roi.c tensor_ops/tensor_ops_reduce_arg.c tensor_ops/tensor_ops_nonzero.c tensor_ops/tensor_ops_compare_logic.c tensor_ops/tensor_ops_unary_basic.c tensor_ops/tensor_ops_reduce_logsumexp.c tensor_ops/tensor_ops_dynamic_quant.c tensor_ops/tensor_ops_elementwise_activation.c tensor_ops/tensor_ops_lrn.c tensor_ops/tensor_ops_shape_index.c tensor_ops/tensor_ops_elementwise.c -shared -lm
Build successful: tensor_ops.so
```
