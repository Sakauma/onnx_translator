# /**
#   ******************************************************************************
#   * @file        runner_cuda_params.py
#   * @author      Egor Izmaylov
#   * @brief       Builds CUDA verifier parameter payloads for numerical plans.
#   * @details     2026.06.28  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import numpy as np

import nn

from .dtype import to_float32
from .runner_params import (
    normalize_slice_parameters,
    recurrent_params_binary,
    slice_io_values,
)


def build_cuda_params(op_name, inputs_np, init_args, shapes, dtypes, out_dtype, nps_out):
    # 3. CUDA 参数打包
    params_bin = None
    if op_name == "conv2d":
        x, w = inputs_np[0], inputs_np[1]
        pads, s, d, g = init_args['pads'], init_args['strides'], init_args['dilations'], init_args['group']
        oh = (x.shape[2] + pads[0] + pads[2] - d[0]*(w.shape[2]-1) - 1)//s[0] + 1
        ow = (x.shape[3] + pads[1] + pads[3] - d[1]*(w.shape[3]-1) - 1)//s[1] + 1
        p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], w.shape[0], w.shape[2], w.shape[3],
                  oh, ow, pads[0], pads[1], s[0], s[1], d[0], d[1], g]
        params_bin = np.array(p_list, dtype=np.int32).tobytes()
    elif op_name == "conv_integer":
        x, w, x_zp, w_zp = inputs_np[0], inputs_np[1], inputs_np[2], inputs_np[3]
        pads, s, d, g = init_args['pads'], init_args['strides'], init_args['dilations'], init_args['group']
        oh = (x.shape[2] + pads[0] + pads[2] - d[0]*(w.shape[2]-1) - 1)//s[0] + 1
        ow = (x.shape[3] + pads[1] + pads[3] - d[1]*(w.shape[3]-1) - 1)//s[1] + 1
        p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], w.shape[0], w.shape[2], w.shape[3],
                  oh, ow, pads[0], pads[1], s[0], s[1], d[0], d[1], g, x_zp.size, w_zp.size]
        params_bin = np.array(p_list, dtype=np.int32).tobytes()
    elif op_name == "qlinear_conv":
        x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp = inputs_np[:8]
        pads, s, d, g = init_args['pads'], init_args['strides'], init_args['dilations'], init_args['group']
        oh = (x.shape[2] + pads[0] + pads[2] - d[0]*(w.shape[2]-1) - 1)//s[0] + 1
        ow = (x.shape[3] + pads[1] + pads[3] - d[1]*(w.shape[3]-1) - 1)//s[1] + 1
        p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], w.shape[0], w.shape[2], w.shape[3],
                  oh, ow, pads[0], pads[1], s[0], s[1], d[0], d[1], g,
                  x_scale.size, x_zp.size, w_scale.size, w_zp.size, y_scale.size, y_zp.size]
        params_bin = np.array(p_list, dtype=np.int32).tobytes()
    elif op_name == "conv_transpose":
        x, w = inputs_np[0], inputs_np[1]
        pads, s, d, g = init_args['pads'], init_args['strides'], init_args['dilations'], init_args['group']
        output_padding = init_args.get('output_padding', [0, 0])
        effective_h = d[0] * (w.shape[2] - 1) + 1
        effective_w = d[1] * (w.shape[3] - 1) + 1
        oh = s[0] * (x.shape[2] - 1) + output_padding[0] + effective_h - pads[0] - pads[2]
        ow = s[1] * (x.shape[3] - 1) + output_padding[1] + effective_w - pads[1] - pads[3]
        out_c = w.shape[1] * g
        p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], w.shape[1], w.shape[2], w.shape[3],
                  out_c, oh, ow, pads[0], pads[1], s[0], s[1], d[0], d[1], g]
        params_bin = np.array(p_list, dtype=np.int32).tobytes()
    elif op_name == "deform_conv":
        x, w, offset = inputs_np[0], inputs_np[1], inputs_np[2]
        pads = list(map(int, init_args.get("pads", [0, 0, 0, 0])))
        s = list(map(int, init_args.get("strides", [1, 1])))
        d = list(map(int, init_args.get("dilations", [1, 1])))
        g = int(init_args.get("group", 1))
        offset_group = int(init_args.get("offset_group", 1))
        has_bias = 1 if len(inputs_np) > 3 and inputs_np[3] is not None else 0
        has_mask = 1 if len(inputs_np) > 4 and inputs_np[4] is not None else 0
        p_list = [
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            w.shape[0],
            w.shape[2],
            w.shape[3],
            offset.shape[2],
            offset.shape[3],
            pads[0],
            pads[1],
            pads[2],
            pads[3],
            s[0],
            s[1],
            d[0],
            d[1],
            g,
            offset_group,
            has_bias,
            has_mask,
        ]
        params_bin = np.array(p_list, dtype=np.int32).tobytes()
    elif op_name == "attention":
        q, k, v = inputs_np[0], inputs_np[1], inputs_np[2]
        mask = inputs_np[3] if len(inputs_np) > 3 else None
        mask_shape = [1, 1, 1, 1]
        mask_rank = 0
        if mask is not None:
            mask_rank = mask.ndim
            mask_shape[:mask_rank] = list(mask.shape)
        int_params = np.array(
            [
                q.shape[0],
                q.shape[1],
                k.shape[1],
                q.shape[2],
                k.shape[2],
                q.shape[3],
                v.shape[3],
                1 if mask is not None else 0,
                1 if len(dtypes) > 3 and dtypes[3] == "bool" else 0,
                mask_rank,
                *mask_shape,
                int(init_args.get("is_causal", 0)),
            ],
            dtype=np.int32,
        )
        float_params = np.array(
            [
                float(init_args["scale"]) if init_args.get("scale") is not None else -1.0,
                float(init_args.get("softcap", 0.0) or 0.0),
            ],
            dtype=np.float32,
        )
        params_bin = int_params.tobytes() + float_params.tobytes()
    elif op_name == "matmul_integer":
        a, b, a_zp, b_zp = inputs_np[0], inputs_np[1], inputs_np[2], inputs_np[3]
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        params_bin = np.array([M, K, N, a_zp.size, b_zp.size], dtype=np.int32).tobytes()
    elif op_name == "qlinear_matmul":
        a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp = inputs_np[:8]
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        params_bin = np.array(
            [M, K, N, a_scale.size, a_zp.size, b_scale.size, b_zp.size, y_scale.size, y_zp.size],
            dtype=np.int32,
        ).tobytes()
    elif op_name == "max_pool":
        x = inputs_np[0]
        k, pads, s = init_args['kernel_shape'], init_args['pads'], init_args['strides']
        oh = (x.shape[2] + pads[0] + pads[2] - k[0])//s[0] + 1
        ow = (x.shape[3] + pads[1] + pads[3] - k[1])//s[1] + 1
        p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], oh, ow, k[0], k[1], pads[0], pads[1], s[0], s[1]]
        params_bin = np.array(p_list, dtype=np.int32).tobytes()
    elif op_name == "average_pool":
        x = inputs_np[0]
        k, pads, s = init_args['kernel_shape'], init_args['pads'], init_args['strides']
        d = init_args.get('dilations', [1, 1])
        count_include_pad = init_args.get('count_include_pad', 0)
        kernel_extent_h = d[0] * (k[0] - 1) + 1
        kernel_extent_w = d[1] * (k[1] - 1) + 1
        oh = (x.shape[2] + pads[0] + pads[2] - kernel_extent_h)//s[0] + 1
        ow = (x.shape[3] + pads[1] + pads[3] - kernel_extent_w)//s[1] + 1
        p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], oh, ow,
                  k[0], k[1], pads[0], pads[1], s[0], s[1], d[0], d[1], count_include_pad]
        params_bin = np.array(p_list, dtype=np.int32).tobytes()
    elif op_name == "lp_pool":
        x = inputs_np[0]
        k, pads, s = init_args['kernel_shape'], init_args['pads'], init_args['strides']
        d = init_args.get('dilations', [1, 1])
        p_norm = init_args.get('p', 2)
        kernel_extent_h = d[0] * (k[0] - 1) + 1
        kernel_extent_w = d[1] * (k[1] - 1) + 1
        oh = (x.shape[2] + pads[0] + pads[2] - kernel_extent_h)//s[0] + 1
        ow = (x.shape[3] + pads[1] + pads[3] - kernel_extent_w)//s[1] + 1
        p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], oh, ow,
                  k[0], k[1], pads[0], pads[1], s[0], s[1], d[0], d[1], p_norm]
        params_bin = np.array(p_list, dtype=np.int32).tobytes()
    elif op_name in {"global_average_pool", "global_max_pool", "global_lp_pool"}:
        x = inputs_np[0]
        spatial_size = int(np.prod(x.shape[2:])) if x.ndim > 2 else 1
        if op_name == "global_lp_pool":
            params_bin = np.array([x.shape[0], x.shape[1], spatial_size, init_args.get('p', 2)], dtype=np.int32).tobytes()
        else:
            params_bin = np.array([x.shape[0], x.shape[1], spatial_size], dtype=np.int32).tobytes()
    elif op_name == "lrn":
        x = inputs_np[0]
        spatial_size = int(np.prod(x.shape[2:]))
        int_params = np.array(
            [x.shape[0], x.shape[1], spatial_size, int(init_args.get("size", 1))],
            dtype=np.int32,
        )
        float_params = np.array(
            [
                float(init_args.get("alpha", 0.0001)),
                float(init_args.get("beta", 0.75)),
                float(init_args.get("bias", 1.0)),
            ],
            dtype=np.float32,
        )
        params_bin = int_params.tobytes() + float_params.tobytes()
    elif op_name == "max_unpool":
        x = inputs_np[0]
        k, pads, s = init_args['kernel_shape'], init_args['pads'], init_args['strides']
        oh = (x.shape[2] - 1) * s[0] - pads[0] - pads[2] + k[0]
        ow = (x.shape[3] - 1) * s[1] - pads[1] - pads[3] + k[1]
        p_list = [x.shape[0], x.shape[1], x.shape[2], x.shape[3], oh, ow,
                  k[0], k[1], pads[0], pads[1], pads[2], pads[3], s[0], s[1]]
        params_bin = np.array(p_list, dtype=np.int32).tobytes()
    elif op_name == "grid_sample":
        x, grid = inputs_np[0], inputs_np[1]
        mode = {"linear": "bilinear", "cubic": "bicubic"}.get(init_args.get("mode", "bilinear"), init_args.get("mode", "bilinear"))
        mode_code = {"bilinear": 0, "nearest": 1, "bicubic": 2}[mode]
        padding_code = {"zeros": 0, "border": 1, "reflection": 2}[init_args.get("padding_mode", "zeros")]
        p_list = [
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            grid.shape[1],
            grid.shape[2],
            mode_code,
            padding_code,
            int(init_args.get("align_corners", 0)),
        ]
        params_bin = np.array(p_list, dtype=np.int32).tobytes()
    elif op_name == "max_roi_pool":
        x, rois = inputs_np[0], inputs_np[1]
        pooled_shape = init_args["pooled_shape"]
        ints = np.array(
            [x.shape[0], x.shape[1], x.shape[2], x.shape[3], rois.shape[0], pooled_shape[0], pooled_shape[1]],
            dtype=np.int32,
        ).tobytes()
        params_bin = ints + np.array([init_args.get("spatial_scale", 1.0)], dtype=np.float32).tobytes()
    elif op_name == "roi_align":
        x, rois = inputs_np[0], inputs_np[1]
        mode_code = 0 if init_args.get("mode", "avg").lower() == "avg" else 1
        coord_mode = init_args.get("coordinate_transformation_mode", "half_pixel").lower()
        coord_code = 0 if coord_mode == "half_pixel" else 1
        ints = np.array(
            [
                x.shape[0],
                x.shape[1],
                x.shape[2],
                x.shape[3],
                rois.shape[0],
                init_args["output_height"],
                init_args["output_width"],
                init_args.get("sampling_ratio", 0),
                mode_code,
                coord_code,
            ],
            dtype=np.int32,
        ).tobytes()
        params_bin = ints + np.array([init_args.get("spatial_scale", 1.0)], dtype=np.float32).tobytes()
    elif op_name == "dft":
        x = inputs_np[0]
        dft_length = int(np.asarray(inputs_np[1]).item())
        axis = int(init_args.get("axis", 1))
        if axis < 0:
            axis += x.ndim
        inverse = init_args.get("inverse", 0)
        onesided = init_args.get("onesided", 0)
        output_shape = list(map(int, np.asarray(nps_out).shape))
        input_shape = list(map(int, x.shape))
        params_bin = np.array(
            [
                len(input_shape),
                axis,
                inverse,
                onesided,
                dft_length,
                input_shape[-1],
                output_shape[-1],
                *input_shape,
                *output_shape,
            ],
            dtype=np.int32,
        ).tobytes()
    elif op_name == "stft":
        signal = inputs_np[0]
        frame_step = int(np.asarray(inputs_np[1]).item())
        frame_length = int(np.asarray(inputs_np[3]).item())
        signal_len = int(signal.shape[-2])
        signal_complex_dim = int(signal.shape[-1])
        prefix_total = int(np.prod(signal.shape[:-2])) if signal.ndim > 2 else 1
        n_frames = 1 + (signal_len - frame_length) // frame_step
        bins = frame_length // 2 + 1 if init_args.get("onesided", 1) else frame_length
        has_window = 1 if inputs_np[2] is not None else 0
        params_bin = np.array(
            [
                prefix_total,
                signal_len,
                signal_complex_dim,
                n_frames,
                bins,
                frame_step,
                frame_length,
                init_args.get("onesided", 1),
                has_window,
            ],
            dtype=np.int32,
        ).tobytes()
    elif op_name in {"rnn", "gru", "lstm"}:
        x = inputs_np[0]
        w = inputs_np[1]
        hidden = int(init_args.get("hidden_size", inputs_np[2].shape[-1]))
        direction_code = {"forward": 0, "reverse": 1, "bidirectional": 2}[init_args.get("direction", "forward")]
        layout = int(init_args.get("layout", 0))
        seq_len = x.shape[1] if layout == 1 else x.shape[0]
        batch = x.shape[0] if layout == 1 else x.shape[1]
        input_size = x.shape[2]
        num_dirs = w.shape[0]
        op_specific = 0
        if op_name == "gru":
            op_specific = int(init_args.get("linear_before_reset", 0))
        elif op_name == "lstm":
            op_specific = int(init_args.get("input_forget", 0))
        params_bin = recurrent_params_binary(
            [seq_len, batch, input_size, num_dirs, hidden, direction_code, layout, op_specific],
            init_args,
        )
    elif op_name == "gemm":
        a, b, c = inputs_np[0], inputs_np[1], inputs_np[2]
        tA, tB = init_args['transA'], init_args['transB']
        M = a.shape[0] if tA==0 else a.shape[1]
        K = a.shape[1] if tA==0 else a.shape[0]
        N = b.shape[1] if tB==0 else b.shape[0]
        has_c = 1 if c is not None else 0
        c_type = 0
        if has_c:
            if c.size == 1: c_type=1
            elif c.ndim==1 or (c.ndim==2 and c.shape[0]==1): c_type=2
            elif c.ndim==2 and c.shape[1]==1: c_type=3
            else: c_type=4
        ints = np.array([M, N, K, tA, tB, c_type, has_c], dtype=np.int32).tobytes()
        floats = np.array([init_args['alpha'], init_args['beta']], dtype=np.float32).tobytes()
        params_bin = ints + floats
    elif op_name in {"softmax", "hardmax", "log_softmax"}:
        x, axis = inputs_np[0], init_args['axis']
        if axis < 0: axis += x.ndim
        inner, outer = x.shape[axis], int(np.prod(x.shape[:axis]))
        rem = int(np.prod(x.shape[axis+1:]))
        params_bin = np.array([outer, inner, rem], dtype=np.int32).tobytes()
    elif op_name in {"quantize_linear", "dequantize_linear"}:
        input_shape = list(inputs_np[0].shape)
        rank = len(input_shape)
        axis = int(init_args.get("axis", 1))
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ValueError(f"{op_name} axis {init_args.get('axis', 1)} is out of bounds for rank {rank}")
        scale_size = int(np.prod(inputs_np[1].shape, dtype=np.int64))
        zp_size = int(np.prod(inputs_np[2].shape, dtype=np.int64)) if inputs_np[2] is not None else 1
        scale_shape = list(inputs_np[1].shape)
        scale_rank = len(scale_shape)
        block_size = int(init_args.get("block_size", 0))
        shape_params = [rank, axis, scale_size, zp_size, block_size, scale_rank, *input_shape]
        if scale_rank == rank:
            shape_params.extend(scale_shape)
        if op_name == "dequantize_linear":
            params_bin = np.array(shape_params, dtype=np.int32).tobytes()
        else:
            target_dtype_code = {
                "uint8": 0,
                "int8": 1,
                "uint16": 2,
                "int16": 3,
                "float8_e4m3": 4,
                "float8_e5m2": 5,
                "float8_e4m3fnuz": 6,
                "float8_e5m2fnuz": 7,
                "uint4": 8,
                "int4": 9,
                "uint2": 10,
                "int2": 11,
                "float4_e2m1": 12,
                "float8_e8m0": 13,
            }.get(out_dtype, 0)
            precision = int(init_args.get("precision", 0))
            if precision == 11:  # ONNX TensorProto.DOUBLE
                use_float_math = 0
            elif precision in {1, 10, 16}:  # FLOAT/FLOAT16/BFLOAT16 均走 float 参考路径
                use_float_math = 1
            else:
                use_float_math = 0 if "float64" in {dtypes[0], dtypes[1]} else 1
            saturate = int(init_args.get("saturate", 1))
            params_bin = np.array([target_dtype_code, use_float_math, saturate, *shape_params], dtype=np.int32).tobytes()
    elif op_name == "matmul":
        M, K = shapes[0]
        K2, N = shapes[1]
        assert K2 == K
        params_bin = np.array([M, K, N], dtype=np.int32).tobytes()
    elif op_name == "reduce_mean":
        M, N = shapes[0]
        params_bin = np.array([M, N], dtype=np.int32).tobytes()

    elif op_name == "gather":
        # 主路径：data=(M,N), indices=(I,), axis=0
        M, N = shapes[0]
        (I,) = shapes[1]
        params_bin = np.array([M, N, I], dtype=np.int32).tobytes()

    elif op_name == "scatternd":
        M, N = shapes[0]         # data: (M,N)
        I, K = shapes[1]         # indices: (I,2)
        assert K == 2
        (I2,) = shapes[2]        # updates: (I,)
        assert I2 == I
        params_bin = np.array([M, N, I], dtype=np.int32).tobytes()
    elif op_name == "tensor_scatter":
        cache_shape = list(shapes[0])
        update_shape = list(shapes[1])
        rank = len(cache_shape)
        axis = int(init_args.get("axis", -2))
        if axis < 0:
            axis += rank
        mode_code = 1 if init_args.get("mode", "linear") == "circular" else 0
        params_bin = np.array([rank, axis, mode_code, *cache_shape, *update_shape], dtype=np.int32).tobytes()
    elif op_name in [
        "reduce_sum", "reduce_max", "reduce_min", "reduce_prod",
        "reduce_l1", "reduce_l2", "reduce_log_sum", "reduce_log_sum_exp", "reduce_sum_square",
    ]:
        in_len = int(inputs_np[0].size)
        params_bin = np.array([in_len], dtype=np.int64).tobytes()

    elif op_name == "gather_elements":
        M, N = shapes[0]
        axis = init_args.get("axis", 1)
        params_bin = np.array([M, N, axis], dtype=np.int32).tobytes()

    elif op_name == "gathernd":
        A, B = shapes[0]
        I, K = shapes[1]
        params_bin = np.array([A, B, I, K], dtype=np.int32).tobytes()

    elif op_name in {"cumsum", "cumprod"}:
        N = int(np.prod(shapes[0]))
        exclusive = int(init_args.get("exclusive", 0))
        reverse = int(init_args.get("reverse", 0))
        params_bin = np.array([N, exclusive, reverse], dtype=np.int32).tobytes()

    elif op_name == "nonzero":
        x = inputs_np[0]
        rank = x.ndim
        dims = np.array(list(x.shape), dtype=np.int32)
        params_bin = np.array([rank], dtype=np.int32).tobytes() + dims.tobytes()

    elif op_name == "argmin" or op_name == "argmax":
        M, N = shapes[0]
        axis = init_args.get("axis", 1)
        keepdims = init_args.get("keepdims", 0)
        select_last_index = init_args.get("select_last_index", 0)
        params_bin = np.array([M, N, axis, keepdims, select_last_index], dtype=np.int32).tobytes()

    elif op_name == "resize":
        N, C, IH, IW = shapes[0]
        OH, OW = init_args["sizes_value"][2], init_args["sizes_value"][3]
        params_bin = np.array([N, C, IH, IW, OH, OW], dtype=np.int32).tobytes()

    elif op_name == "affine_grid":
        size_value = list(map(int, init_args.get("size_value", list(inputs_np[1]))))
        spatial_rank = len(size_value) - 2
        if spatial_rank == 2:
            n, _, h, w = size_value
            d = 1
        elif spatial_rank == 3:
            n, _, d, h, w = size_value
        else:
            raise ValueError(f"AffineGrid size_value must have rank 4 or 5, got {size_value}")
        params_bin = np.array(
            [spatial_rank, n, d, h, w, int(init_args.get("align_corners", 0))],
            dtype=np.int32,
        ).tobytes()

    elif op_name == "expand":
        input_shape = list(shapes[0])
        target_shape = list(map(int, init_args.get("target_shape", input_shape)))
        aligned_input = [1] * (len(target_shape) - len(input_shape)) + input_shape
        output_shape = [
            in_dim if target_dim == -1 else target_dim
            for in_dim, target_dim in zip(aligned_input, target_shape)
        ]
        params_bin = np.array([len(input_shape), len(output_shape), *input_shape, *output_shape], dtype=np.int32).tobytes()

    elif op_name == "reshape":
        params_bin = None

    elif op_name == "squeeze":
        params_bin = None

    elif op_name == "unsqueeze":
        params_bin = None

    elif op_name == "transpose":
        input_shape = list(shapes[0])
        rank = len(input_shape)
        perm = init_args.get("perm")
        if perm is None:
            perm = list(reversed(range(rank)))
        perm = [axis + rank if axis < 0 else axis for axis in perm]
        params_bin = np.array([rank, *input_shape, *perm], dtype=np.int32).tobytes()

    elif op_name == "tile":
        input_shape = list(shapes[0])
        repeats = list(map(int, init_args.get("repeats_value", [1] * len(input_shape))))
        params_bin = np.array([len(input_shape), *input_shape, *repeats], dtype=np.int32).tobytes()

    elif op_name == "concat":
        rank = len(shapes[0])
        axis = init_args.get("axis", 0)
        if axis < 0:
            axis += rank
        shape_values = []
        for shape in shapes:
            shape_values.extend(shape)
        params_bin = np.array([len(shapes), rank, axis, *shape_values], dtype=np.int32).tobytes()

    elif op_name == "pad":
        input_shape = list(shapes[0])
        rank = len(input_shape)
        pads = list(map(int, init_args.get("pads_value", [0] * (2 * rank))))
        output_shape = [input_shape[i] + pads[i] + pads[i + rank] for i in range(rank)]
        mode_code = {"constant": 0, "reflect": 1, "edge": 2, "wrap": 3}.get(init_args.get("mode", "constant"), 0)
        params_bin = np.array([rank, mode_code, *input_shape, *output_shape], dtype=np.int32).tobytes()

    elif op_name == "center_crop_pad":
        input_shape = list(shapes[0])
        output_shape = list(np.asarray(nps_out).shape)
        params_bin = np.array([len(input_shape), *input_shape, *output_shape], dtype=np.int32).tobytes()

    elif op_name == "depth_to_space":
        n, c, h, w = shapes[0]
        mode_code = 0 if init_args.get("mode", "DCR") == "DCR" else 1
        params_bin = np.array(
            [n, c, h, w, int(init_args["blocksize"]), mode_code],
            dtype=np.int32,
        ).tobytes()

    elif op_name == "space_to_depth":
        n, c, h, w = shapes[0]
        params_bin = np.array(
            [n, c, h, w, int(init_args["blocksize"])],
            dtype=np.int32,
        ).tobytes()

    elif op_name == "slice":
        input_shape = list(map(int, shapes[0]))
        output_shape = list(map(int, np.asarray(nps_out).shape))
        starts, ends, axes, steps = slice_io_values(init_args, input_shape)
        full_starts, _full_ends, full_steps = normalize_slice_parameters(input_shape, starts, ends, axes, steps)
        params_bin = np.array(
            [len(input_shape), *input_shape, *output_shape, *full_starts, *full_steps],
            dtype=np.int32,
        ).tobytes()

    elif op_name in {"tril", "triu", "trilu"}:
        input_shape = list(map(int, shapes[0]))
        if op_name == "tril":
            upper = 0
        elif op_name == "triu":
            upper = 1
        else:
            upper = int(init_args.get("upper", 1))
        k_val = int(np.asarray(inputs_np[1]).item()) if len(inputs_np) > 1 else int(init_args.get("k_value", 0))
        params_bin = np.array([len(input_shape), upper, k_val, *input_shape], dtype=np.int32).tobytes()

    elif op_name == "one_hot":
        indices_shape = list(map(int, shapes[0]))
        output_shape = list(map(int, np.asarray(nps_out).shape))
        axis = int(init_args.get("axis", -1))
        if axis < 0:
            axis += len(output_shape)
        depth = int(np.asarray(inputs_np[1]).item())
        params_bin = np.array(
            [len(indices_shape), len(output_shape), axis, depth, *indices_shape, *output_shape],
            dtype=np.int32,
        ).tobytes()

    elif op_name == "reverse_sequence":
        input_shape = list(map(int, shapes[0]))
        rank = len(input_shape)
        time_axis = int(init_args.get("time_axis", 0))
        batch_axis = int(init_args.get("batch_axis", 1))
        if time_axis < 0:
            time_axis += rank
        if batch_axis < 0:
            batch_axis += rank
        params_bin = np.array([rank, time_axis, batch_axis, *input_shape], dtype=np.int32).tobytes()

    elif op_name == "det":
        input_shape = list(map(int, shapes[0]))
        n = input_shape[-1]
        batch = int(np.prod(input_shape[:-2])) if len(input_shape) > 2 else 1
        params_bin = np.array([batch, n], dtype=np.int32).tobytes()

    elif op_name == "mel_weight_matrix":
        bins = int(np.asarray(inputs_np[0]).item())
        dft_len = int(np.asarray(inputs_np[1]).item())
        sample_rate = int(np.asarray(inputs_np[2]).item())
        lower = float(np.asarray(inputs_np[3]).item())
        upper = float(np.asarray(inputs_np[4]).item())
        spectrogram_bins = dft_len // 2 + 1
        params_bin = (
            np.array([bins, dft_len, sample_rate, spectrogram_bins], dtype=np.int32).tobytes()
            + np.array([lower, upper], dtype=np.float32).tobytes()
        )

    elif op_name in {"hann_window", "hamming_window", "blackman_window"}:
        size_value = int(np.asarray(inputs_np[0]).item())
        params_bin = np.array([size_value, int(init_args.get("periodic", 1))], dtype=np.int32).tobytes()

    elif op_name == "compress":
        input_shape = list(map(int, shapes[0]))
        output_shape = list(map(int, np.asarray(nps_out).shape))
        axis_value = init_args.get("axis", None)
        axis = -1
        if axis_value is not None:
            axis = int(axis_value)
            if axis < 0:
                axis += len(input_shape)
        params_bin = np.array(
            [len(input_shape), len(output_shape), axis, int(np.prod(shapes[1])), *input_shape, *output_shape],
            dtype=np.int32,
        ).tobytes()

    elif op_name == "scatter_elements":
        data_shape = list(map(int, shapes[0]))
        update_shape = list(map(int, shapes[2]))
        axis = int(init_args.get("axis", 0))
        if axis < 0:
            axis += len(data_shape)
        reduction = {"none": 0, "add": 1, "mul": 2}.get(init_args.get("reduction", "none"), 0)
        params_bin = np.array(
            [len(data_shape), axis, reduction, *data_shape, *update_shape],
            dtype=np.int32,
        ).tobytes()

    elif op_name == "constant_of_shape":
        target_shape = list(map(int, init_args.get("shape_value", list(shapes[0]))))
        fill_value = float(init_args.get("fill_value", 0.0))
        params_bin = (
            np.array([len(target_shape), *target_shape], dtype=np.int32).tobytes()
            + np.array([fill_value], dtype=np.float32).tobytes()
        )

    elif op_name == "eye_like":
        rows, cols = shapes[0]
        params_bin = np.array([rows, cols, int(init_args.get("k", 0))], dtype=np.int32).tobytes()

    elif op_name in {"cast", "cast_like"}:
        output_kind = {"float": 0, "int32": 1, "int64": 2, "bool": 3}.get(out_dtype, 0)
        params_bin = np.array([output_kind], dtype=np.int32).tobytes()

    elif op_name == "bitcast":
        params_bin = np.array([np.dtype(nn.DTYPE_TO_NUMPY[out_dtype]).itemsize], dtype=np.int32).tobytes()

    elif op_name == "bit_shift":
        direction = 0 if init_args.get("direction", "LEFT").upper() == "LEFT" else 1
        params_bin = np.array([direction], dtype=np.int32).tobytes()

    elif op_name == "isinf":
        params_bin = np.array(
            [int(init_args.get("detect_positive", 1)), int(init_args.get("detect_negative", 1))],
            dtype=np.int32,
        ).tobytes()

    elif op_name == "size":
        params_bin = np.array([int(np.prod(inputs_np[0].shape, dtype=np.int64))], dtype=np.int64).tobytes()

    elif op_name == "hard_sigmoid":
        params_bin = np.array(
            [float(init_args.get("alpha", 0.2)), float(init_args.get("beta", 0.5))],
            dtype=np.float32,
        ).tobytes()

    elif op_name == "rms_normalization":
        input_shape = list(shapes[0])
        axis = int(init_args.get("axis", -1))
        if axis < 0:
            axis += len(input_shape)
        normalized_size = int(np.prod(input_shape[axis:]))
        row_count = int(np.prod(input_shape[:axis])) if axis > 0 else 1
        params_bin = (
            np.array([row_count, normalized_size], dtype=np.int32).tobytes()
            + np.array([float(init_args.get("epsilon", 1e-5))], dtype=np.float32).tobytes()
        )

    elif op_name == "mean_variance_normalization":
        x = inputs_np[0]
        rank = x.ndim
        axes = init_args.get("axes", [0, 2, 3])
        resolved_axes = []
        for ax in axes:
            axis = int(ax)
            if axis < 0:
                axis += rank
            resolved_axes.append(axis)
        resolved_axes = sorted(set(resolved_axes))
        params_bin = np.array(
            [rank, len(resolved_axes), *x.shape, *resolved_axes],
            dtype=np.int32,
        ).tobytes()

    elif op_name == "batch_normalization":
        x = inputs_np[0]
        spatial_size = int(np.prod(x.shape[2:])) if x.ndim > 2 else 1
        params_bin = (
            np.array([x.shape[0], x.shape[1], spatial_size, int(init_args.get("training_mode", 0))], dtype=np.int32).tobytes()
            + np.array([float(init_args.get("epsilon", 1e-5)), float(init_args.get("momentum", 0.9))], dtype=np.float32).tobytes()
        )

    elif op_name == "instance_normalization":
        x = inputs_np[0]
        spatial_size = int(np.prod(x.shape[2:])) if x.ndim > 2 else 1
        params_bin = (
            np.array([x.shape[0], x.shape[1], spatial_size], dtype=np.int32).tobytes()
            + np.array([float(init_args.get("epsilon", 1e-5))], dtype=np.float32).tobytes()
        )

    elif op_name == "layer_normalization":
        x = inputs_np[0]
        input_shape = list(x.shape)
        axis = int(init_args.get("axis", -1))
        if axis < 0:
            axis += len(input_shape)
        normalized_size = int(np.prod(input_shape[axis:]))
        row_count = int(np.prod(input_shape[:axis])) if axis > 0 else 1
        has_scale = 1 if inputs_np[1] is not None else 0
        has_bias = 1 if inputs_np[2] is not None else 0
        params_bin = (
            np.array([row_count, normalized_size, has_scale, has_bias, int(init_args.get("emit_stats", 0))], dtype=np.int32).tobytes()
            + np.array([float(init_args.get("epsilon", 1e-5))], dtype=np.float32).tobytes()
        )

    elif op_name == "lp_normalization":
        x = inputs_np[0]
        axis = int(init_args.get("axis", -1))
        if axis < 0:
            axis += x.ndim
        outer = int(np.prod(x.shape[:axis])) if axis > 0 else 1
        inner = int(x.shape[axis])
        remaining = int(np.prod(x.shape[axis + 1:])) if axis + 1 < x.ndim else 1
        params_bin = np.array([outer, inner, remaining, int(init_args.get("p", 2))], dtype=np.int32).tobytes()

    elif op_name == "group_normalization":
        x = inputs_np[0]
        spatial_size = int(np.prod(x.shape[2:])) if x.ndim > 2 else 1
        params_bin = (
            np.array([x.shape[0], x.shape[1], spatial_size, int(init_args["num_groups"])], dtype=np.int32).tobytes()
            + np.array([float(init_args.get("epsilon", 1e-5))], dtype=np.float32).tobytes()
        )

    elif op_name == "rotary_embedding":
        x = inputs_np[0]
        rank = x.ndim
        if rank == 4:
            batch, heads, sequence, head_size = x.shape
        else:
            batch, sequence, hidden = x.shape
            heads = int(init_args["num_heads"])
            head_size = hidden // heads
        rotary_dim = int(init_args.get("rotary_embedding_dim", 0) or head_size)
        has_position_ids = 1 if inputs_np[3] is not None else 0
        cos_rank = inputs_np[1].ndim
        params_bin = np.array(
            [
                rank,
                batch,
                heads,
                sequence,
                head_size,
                rotary_dim,
                int(init_args.get("interleaved", 0)),
                has_position_ids,
                cos_rank,
            ],
            dtype=np.int32,
        ).tobytes()

    elif op_name == "col2im":
        x = inputs_np[0]
        image_shape = [int(v) for v in np.asarray(inputs_np[1], dtype=np.int64).reshape(-1)]
        block_shape = [int(v) for v in np.asarray(inputs_np[2], dtype=np.int64).reshape(-1)]
        spatial_rank = len(image_shape)
        pads = list(map(int, init_args.get("pads", [0] * (2 * spatial_rank))))
        strides = list(map(int, init_args.get("strides", [1] * spatial_rank)))
        dilations = list(map(int, init_args.get("dilations", [1] * spatial_rank)))
        n_blocks = []
        for axis in range(spatial_rank):
            count = (
                image_shape[axis]
                + pads[axis]
                + pads[axis + spatial_rank]
                - dilations[axis] * (block_shape[axis] - 1)
                - 1
            ) // strides[axis] + 1
            n_blocks.append(int(count))
        kernel_size = int(np.prod(block_shape))
        block_count = int(np.prod(n_blocks))
        channels = int(x.shape[1] // kernel_size)
        params_bin = np.array(
            [
                x.shape[0],
                channels,
                spatial_rank,
                x.shape[2],
                kernel_size,
                block_count,
                *image_shape,
                *block_shape,
                *pads,
                *strides,
                *dilations,
                *n_blocks,
            ],
            dtype=np.int32,
        ).tobytes()

    elif op_name in {"elu", "leaky_relu", "celu", "thresholded_relu"}:
        params_bin = np.array([float(init_args.get("alpha", 1.0))], dtype=np.float32).tobytes()

    elif op_name == "binarizer":
        params_bin = np.array([float(init_args.get("threshold", 0.0))], dtype=np.float32).tobytes()

    elif op_name == "swish":
        params_bin = np.array([float(init_args.get("alpha", 1.0))], dtype=np.float32).tobytes()

    elif op_name == "selu":
        params_bin = np.array(
            [float(init_args.get("alpha", 1.67326)), float(init_args.get("gamma", 1.0507))],
            dtype=np.float32,
        ).tobytes()

    elif op_name == "gelu" and init_args.get("approximate", "none") == "tanh":
        params_bin = np.array([1], dtype=np.int32).tobytes()

    elif op_name == "shrink":
        params_bin = np.array(
            [float(init_args.get("bias", 0.0)), float(init_args.get("lambd", 0.5))],
            dtype=np.float32,
        ).tobytes()

    elif op_name == "einsum":
        M, K = shapes[0]
        K2, N = shapes[1]
        assert K == K2
        params_bin = np.array([M, K, N], dtype=np.int32).tobytes()

    elif op_name == "topk":
        M, N = shapes[0]
        k_val = int(inputs_np[1].reshape(-1)[0])
        axis = init_args.get("axis", 1)
        largest = init_args.get("largest", 1)
        sorted_flag = init_args.get("sorted", 1)
        params_bin = np.array([M, N, axis, k_val, largest, sorted_flag], dtype=np.int32).tobytes()

    elif op_name == "random_uniform_like":
        numel = int(np.prod(shapes[0]))
        low = float(init_args.get("low", 0.0))
        high = float(init_args.get("high", 1.0))
        seed = np.uint32(int(init_args.get("seed", 123)))
        params_bin = (np.array([numel], dtype=np.int32).tobytes() + np.array([low, high], dtype=np.float32).tobytes() + np.array([seed], dtype=np.uint32).tobytes())

    elif op_name == "random_uniform":
        numel = int(np.prod(np.asarray(nps_out).shape))
        low = float(init_args.get("low", 0.0))
        high = float(init_args.get("high", 1.0))
        seed = np.uint32(int(init_args.get("seed", 123)))
        params_bin = (
            np.array([numel], dtype=np.int32).tobytes()
            + np.array([low, high], dtype=np.float32).tobytes()
            + np.array([seed], dtype=np.uint32).tobytes()
        )

    elif op_name in {"random_normal", "random_normal_like"}:
        numel = int(np.prod(np.asarray(nps_out).shape))
        mean = float(init_args.get("mean", 0.0))
        scale = float(init_args.get("scale", 1.0))
        seed = np.uint32(int(init_args.get("seed", 123)))
        params_bin = (
            np.array([numel], dtype=np.int32).tobytes()
            + np.array([mean, scale], dtype=np.float32).tobytes()
            + np.array([seed], dtype=np.uint32).tobytes()
        )

    elif op_name == "bernoulli":
        numel = int(np.prod(shapes[0]))
        seed = np.uint32(int(init_args.get("seed", 123)))
        params_bin = np.array([numel], dtype=np.int32).tobytes() + np.array([seed], dtype=np.uint32).tobytes()

    elif op_name == "multinomial":
        batch, classes = shapes[0]
        sample_size = int(init_args.get("sample_size", np.asarray(nps_out).shape[1]))
        output_dtype_code = 1 if out_dtype == "int64" else 0
        seed = np.uint32(int(init_args.get("seed", 0) or 0))
        params_bin = (
            np.array([batch, classes, sample_size, output_dtype_code], dtype=np.int32).tobytes()
            + np.array([seed], dtype=np.uint32).tobytes()
        )

    elif op_name in {"negative_log_likelihood_loss", "softmax_cross_entropy_loss"}:
        data_shape = list(map(int, shapes[0]))
        batch = int(data_shape[0])
        classes = int(data_shape[1])
        spatial = int(np.prod(data_shape[2:])) if len(data_shape) > 2 else 1
        reduction_code = {"none": 0, "mean": 1, "sum": 2}[init_args.get("reduction", "mean")]
        has_weight = 1 if len(inputs_np) > 2 and inputs_np[2] is not None else 0
        has_ignore = 1 if init_args.get("ignore_index", None) is not None else 0
        emit_log = 1 if op_name == "softmax_cross_entropy_loss" and int(init_args.get("emit_log_prob", 0)) else 0
        int_params = np.array(
            [batch, classes, spatial, reduction_code, has_weight, has_ignore, emit_log],
            dtype=np.int32,
        )
        ignore_value = np.array([int(init_args.get("ignore_index", 0) or 0)], dtype=np.int64)
        params_bin = int_params.tobytes() + ignore_value.tobytes()

    elif op_name == "non_max_suppression":
        boxes = inputs_np[0]
        scores = inputs_np[1]
        max_output = int(np.asarray(inputs_np[2]).item())
        iou_threshold = np.float32(to_float32(inputs_np[3], dtypes[3]).reshape(-1)[0])
        score_threshold = np.float32(to_float32(inputs_np[4], dtypes[4]).reshape(-1)[0])
        params_bin = (
            np.array(
                [boxes.shape[0], boxes.shape[1], scores.shape[1], max_output, int(init_args.get("center_point_box", 0))],
                dtype=np.int32,
            ).tobytes()
            + np.array([iou_threshold, score_threshold], dtype=np.float32).tobytes()
        )

    elif op_name == "dropout":
        input_len = int(np.prod(shapes[0]))
        ratio = float(np.asarray(inputs_np[1]).item()) if len(inputs_np) > 1 else float(init_args.get("ratio_value", 0.5))
        training_mode = int(bool(np.asarray(inputs_np[2]).item())) if len(inputs_np) > 2 else int(bool(init_args.get("training_mode_value", 0)))
        seed_value = init_args.get("seed", 0)
        if seed_value is None:
            seed_value = 0
        params_bin = (
            np.array([input_len, training_mode], dtype=np.int32).tobytes()
            + np.array([np.uint32(int(seed_value))], dtype=np.uint32).tobytes()
            + np.array([ratio], dtype=np.float32).tobytes()
        )

    elif op_name == "split":
        input_shape = list(map(int, shapes[0]))
        axis = int(init_args.get("axis", 0))
        if axis < 0:
            axis += len(input_shape)
        if len(inputs_np) > 1:
            split_sizes = [int(v) for v in np.asarray(inputs_np[1], dtype=np.int64).reshape(-1)]
        else:
            count = int(init_args.get("num_outputs", len(nps_out)))
            dim_len = input_shape[axis]
            div, remainder = divmod(dim_len, count)
            split_sizes = [div + (1 if idx < remainder else 0) for idx in range(count)]
        params_bin = np.array(
            [len(input_shape), axis, len(split_sizes), *input_shape, *split_sizes],
            dtype=np.int32,
        ).tobytes()

    elif op_name == "unique":
        type_code = 1 if dtypes[0] == "int64" else 0
        params_bin = np.array(
            [type_code, int(init_args.get("sorted", 1)), int(np.prod(shapes[0]))],
            dtype=np.int32,
        ).tobytes()

    return params_bin
