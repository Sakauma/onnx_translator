# /**
#   ******************************************************************************
#   * @file        conv_ops.py
#   * @author      Egor Izmaylov
#   * @brief       保存 `conv_ops` 分组中的 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *

class Conv(Ops):
    # 初始化 `Conv` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, pads, strides, dilations, group, dtype, kernel_shape=None, auto_pad="NOTSET", version="17"):
        super(Conv, self).__init__(inputs, outputs)
        # 必须完整保存所有参数
        self.pads = list(pads) if pads is not None else None
        self.strides = list(strides) if strides is not None else None
        self.dilations = list(dilations) if dilations is not None else None
        self.group = group
        self.kernel_shape = list(kernel_shape) if kernel_shape is not None else None
        self.auto_pad = auto_pad
        self.dtype = dtype
        self.version = version

        # 注册 C 函数参数类型
        if self.lib:
            self.lib.conv2d_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), 
                ctypes.POINTER(CTensor), ctypes.POINTER(CConvParams)
            ]

    # 封装 `_params` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _params(self, x_shape, w_shape):
        spatial_rank = len(x_shape) - 2
        kernel_shape = self.kernel_shape if self.kernel_shape is not None else list(w_shape[2:])
        if list(kernel_shape) != list(w_shape[2:]):
            raise ValueError(f"Conv kernel_shape {kernel_shape} does not match weight spatial shape {w_shape[2:]}")
        strides = _conv_attr(self.strides, spatial_rank, 1)
        dilations = _conv_attr(self.dilations, spatial_rank, 1)
        pads = _conv_resolve_pads(list(x_shape[2:]), kernel_shape, self.pads, strides, dilations, self.auto_pad)
        out_spatial = _conv_output_spatial(list(x_shape[2:]), kernel_shape, pads, strides, dilations)
        return kernel_shape, strides, dilations, pads, tuple(x_shape[:1]) + tuple(w_shape[:1]) + out_spatial

    # 执行 `Conv` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x: Tensor, w: Tensor, b: Tensor = None) -> dict:
        _, strides, dilations, pads, out_shape = self._params(x.size, w.size)

        if self.lib is None or len(x.size) != 4:
            out_data = _conv_nd_numpy(
                _tensor_data_as_numeric(x),
                _tensor_data_as_numeric(w),
                None if b is None else _tensor_data_as_numeric(b),
                pads=pads,
                strides=strides,
                dilations=dilations,
                group=self.group,
                auto_pad="NOTSET",
                acc_dtype=np.float64,
            )
            out_data = _cast_numeric_to_dtype(out_data, self.dtype)
            out_tensor = Tensor(*out_data.shape, dtype=self.dtype, data=out_data)
            values = {"tensor": out_tensor, "parameters": None, "graph": None}
            self.parameters = {"values": values}
            return values
        
        # 2. 准备 C 参数
        pads_arr = (ctypes.c_int * 4)(*pads)
        strides_arr = (ctypes.c_int * 2)(*strides)
        dilations_arr = (ctypes.c_int * 2)(*dilations)
        
        c_params = CConvParams()
        c_params.pads = ctypes.cast(pads_arr, ctypes.POINTER(ctypes.c_int))
        c_params.strides = ctypes.cast(strides_arr, ctypes.POINTER(ctypes.c_int))
        c_params.dilations = ctypes.cast(dilations_arr, ctypes.POINTER(ctypes.c_int))
        c_params.group = self.group

        # 3. 准备 Tensor
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        w_c = self._numpy_to_ctensor(w.data, w.dtype)
        b_c = self._numpy_to_ctensor(b.data, b.dtype) if b is not None else ctypes.POINTER(CTensor)()
        
        # 创建输出 Tensor
        output_shape_c = (ctypes.c_int * 4)(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, 4, DTYPE_MAP[self.dtype])
        
        # 4. 执行计算
        self.lib.conv2d_forward(x_c, w_c, b_c, output_c, ctypes.byref(c_params))
        
        # 5. 回收与返回
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(x_c)
        self.lib.free_tensor(w_c)
        self.lib.free_tensor(output_c)
        if b is not None: self.lib.free_tensor(b_c)

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # 执行 `Conv` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x: Tensor_, w: Tensor_, b: Tensor_ = None) -> dict:
        # 仅做形状推断
        _, _, _, _, out_shape = self._params(x.size, w.size)
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values


class ConvTranspose(Ops):
    # 初始化 `ConvTranspose` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(
        self,
        inputs,
        outputs,
        pads=None,
        strides=None,
        dilations=None,
        group=1,
        kernel_shape=None,
        output_padding=None,
        output_shape=None,
        auto_pad="NOTSET",
        dtype="float32",
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.pads = list(pads) if pads is not None else None
        self.strides = list(strides) if strides is not None else None
        self.dilations = list(dilations) if dilations is not None else None
        self.group = group
        self.kernel_shape = list(kernel_shape) if kernel_shape is not None else None
        self.output_padding = list(output_padding) if output_padding is not None else None
        self.output_shape = tuple(output_shape) if output_shape is not None else None
        self.auto_pad = auto_pad
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.conv_transpose2d_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CConvParams)
            ]

    # 封装 `_params` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _params(self, x_shape, w_shape):
        spatial_rank = len(x_shape) - 2
        kernel_shape = self.kernel_shape if self.kernel_shape is not None else list(w_shape[2:])
        strides = _conv_attr(self.strides, spatial_rank, 1)
        dilations = _conv_attr(self.dilations, spatial_rank, 1)
        output_padding = _conv_attr(self.output_padding, spatial_rank, 0)
        pads = [0] * (2 * spatial_rank) if self.pads is None else list(self.pads)
        if len(pads) != 2 * spatial_rank:
            raise ValueError(f"ConvTranspose pads must contain {2 * spatial_rank} values")
        effective = _conv_effective_kernel(kernel_shape, dilations)
        if self.output_shape is not None:
            out_spatial = tuple(self.output_shape)
            if len(out_spatial) != spatial_rank:
                raise ValueError(f"ConvTranspose output_shape rank {len(out_spatial)} does not match spatial rank {spatial_rank}")
            if self.pads is None or self.auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
                begin_pads, end_pads = [], []
                for dim in range(spatial_rank):
                    total = strides[dim] * (x_shape[dim + 2] - 1) + output_padding[dim] + effective[dim] - out_spatial[dim]
                    if self.auto_pad == "SAME_LOWER":
                        begin = total - total // 2
                    else:
                        begin = total // 2
                    begin_pads.append(begin)
                    end_pads.append(total - begin)
                pads = begin_pads + end_pads
        elif self.auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
            out_spatial = tuple(x_shape[dim + 2] * strides[dim] for dim in range(spatial_rank))
            begin_pads, end_pads = [], []
            for dim in range(spatial_rank):
                total = strides[dim] * (x_shape[dim + 2] - 1) + output_padding[dim] + effective[dim] - out_spatial[dim]
                if self.auto_pad == "SAME_LOWER":
                    begin = total - total // 2
                else:
                    begin = total // 2
                begin_pads.append(begin)
                end_pads.append(total - begin)
            pads = begin_pads + end_pads
        elif self.auto_pad == "VALID":
            pads = [0] * (2 * spatial_rank)
            out_spatial = tuple(
                strides[dim] * (x_shape[dim + 2] - 1)
                + output_padding[dim]
                + effective[dim]
                for dim in range(spatial_rank)
            )
        else:
            out_spatial = tuple(
                strides[dim] * (x_shape[dim + 2] - 1)
                + output_padding[dim]
                + effective[dim]
                - pads[dim]
                - pads[spatial_rank + dim]
                for dim in range(spatial_rank)
            )
        return kernel_shape, pads, strides, dilations, output_padding, out_spatial

    # 执行 `ConvTranspose` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, w, b=None):
        x_data = np.asarray(_tensor_data_as_numeric(x), dtype=np.float64)
        w_data = np.asarray(_tensor_data_as_numeric(w), dtype=np.float64)
        n_batches, in_channels = x_data.shape[:2]
        if w_data.shape[0] != in_channels:
            raise ValueError(f"ConvTranspose weight input channels {w_data.shape[0]} != input channels {in_channels}")
        if self.group <= 0 or in_channels % self.group != 0:
            raise ValueError(f"Invalid ConvTranspose group={self.group} for input channels={in_channels}")
        m_per_group = w_data.shape[1]
        out_channels = m_per_group * self.group
        in_per_group = in_channels // self.group
        kernel_shape, pads, strides, dilations, _output_padding, out_spatial = self._params(x_data.shape, w_data.shape)
        out_shape = (n_batches, out_channels) + out_spatial
        if (
            self.lib is not None
            and x.data.ndim == 4
            and w.data.ndim == 4
            and x.dtype in nn.DTYPE_MAP
            and w.dtype in nn.DTYPE_MAP
            and (b is None or b.dtype in nn.DTYPE_MAP)
        ):
            pads_c = (ctypes.c_int * len(pads))(*pads)
            strides_c = (ctypes.c_int * len(strides))(*strides)
            dilations_c = (ctypes.c_int * len(dilations))(*dilations)
            c_params = CConvParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.group = self.group

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data.astype(nn.DTYPE_TO_NUMPY[x.dtype], copy=False)), x.dtype)
            w_c = self._numpy_to_ctensor(np.ascontiguousarray(w.data.astype(nn.DTYPE_TO_NUMPY[w.dtype], copy=False)), w.dtype)
            b_c = self._numpy_to_ctensor(np.ascontiguousarray(b.data), b.dtype) if b is not None else ctypes.POINTER(CTensor)()
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            self.lib.conv_transpose2d_forward(x_c, w_c, b_c, out_c, ctypes.byref(c_params))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(w_c)
            self.lib.free_tensor(out_c)
            if b is not None:
                self.lib.free_tensor(b_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        out = np.zeros((n_batches, out_channels) + out_spatial, dtype=np.float64)
        spatial_rank = len(out_spatial)

        for n in range(n_batches):
            for ic in range(in_channels):
                group_idx = ic // in_per_group
                for in_index in np.ndindex(*x_data.shape[2:]):
                    x_value = x_data[(n, ic) + in_index]
                    for oc_local in range(m_per_group):
                        oc = group_idx * m_per_group + oc_local
                        for kernel_index in np.ndindex(*kernel_shape):
                            out_index = tuple(
                                in_index[dim] * strides[dim] + kernel_index[dim] * dilations[dim] - pads[dim]
                                for dim in range(spatial_rank)
                            )
                            if all(0 <= out_index[dim] < out_spatial[dim] for dim in range(spatial_rank)):
                                out[(n, oc) + out_index] += x_value * w_data[(ic, oc_local) + kernel_index]

        if b is not None:
            out += np.asarray(_tensor_data_as_numeric(b), dtype=np.float64).reshape((1, out_channels) + (1,) * spatial_rank)
        out_data = _cast_numeric_to_dtype(out, self.dtype)
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `ConvTranspose` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, w, b=None):
        _kernel_shape, _pads, _strides, _dilations, _output_padding, out_spatial = self._params(x.size, w.size)
        out_channels = w.size[1] * self.group
        return {"tensor": Tensor_(x.size[0], out_channels, *out_spatial, dtype=self.dtype), "parameters": None}


class Col2Im(Ops):
    # 初始化 `Col2Im` 的构造参数，保存 pads、strides、dilations 和输出 dtype。
    def __init__(self, inputs, outputs, pads=None, strides=None, dilations=None, dtype="float32", version="18"):
        super().__init__(inputs, outputs)
        self.pads = list(pads) if pads is not None else None
        self.strides = list(strides) if strides is not None else None
        self.dilations = list(dilations) if dilations is not None else None
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.col2im_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CConvParams),
            ]

    # 解析 image_shape、block_shape 和滑动块数量，统一 forward 与 forward_ 的校验逻辑。
    def _params(self, input_shape, image_shape_value, block_shape_value):
        if len(input_shape) != 3:
            raise ValueError(f"Col2Im input must be rank 3 [N, C*prod(block_shape), L], got {input_shape}")
        image_shape = [int(v) for v in np.asarray(image_shape_value, dtype=np.int64).reshape(-1).tolist()]
        block_shape = [int(v) for v in np.asarray(block_shape_value, dtype=np.int64).reshape(-1).tolist()]
        spatial_rank = len(image_shape)
        if spatial_rank < 2 or len(block_shape) != spatial_rank:
            raise ValueError(f"Col2Im image_shape and block_shape must have the same rank >= 2, got {image_shape} and {block_shape}")
        strides = _conv_attr(self.strides, spatial_rank, 1)
        dilations = _conv_attr(self.dilations, spatial_rank, 1)
        pads = [0] * (2 * spatial_rank) if self.pads is None else list(self.pads)
        if len(pads) != 2 * spatial_rank:
            raise ValueError(f"Col2Im pads must contain {2 * spatial_rank} values")
        block_size = int(np.prod(block_shape, dtype=np.int64))
        if block_size <= 0 or input_shape[1] % block_size != 0:
            raise ValueError(f"Col2Im input channel dimension {input_shape[1]} is not divisible by block size {block_size}")
        channels = input_shape[1] // block_size
        n_blocks = []
        for axis in range(spatial_rank):
            block_count = (
                image_shape[axis]
                + pads[axis]
                + pads[axis + spatial_rank]
                - dilations[axis] * (block_shape[axis] - 1)
                - 1
            ) // strides[axis] + 1
            if block_count <= 0:
                raise ValueError(f"Col2Im calculated non-positive block count {block_count} on axis {axis}")
            n_blocks.append(int(block_count))
        expected_l = int(np.prod(n_blocks, dtype=np.int64))
        if input_shape[2] != expected_l:
            raise ValueError(f"Col2Im input L={input_shape[2]} does not match sliding block count {expected_l}")
        return image_shape, block_shape, pads, strides, dilations, channels, tuple(n_blocks)

    # 执行 `Col2Im` 的真实张量计算路径，将列块按官方 fold 语义累加回图像张量。
    def forward(self, input, image_shape, block_shape):
        image_values = np.asarray(image_shape.data, dtype=np.int64)
        block_values = np.asarray(block_shape.data, dtype=np.int64)
        image_dims, block_dims, pads, strides, dilations, channels, _n_blocks = self._params(input.size, image_values, block_values)
        out_shape = (input.size[0], channels, *image_dims)

        if self.lib is not None and input.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            pads_c = (ctypes.c_int * len(pads))(*pads)
            strides_c = (ctypes.c_int * len(strides))(*strides)
            dilations_c = (ctypes.c_int * len(dilations))(*dilations)
            c_params = CConvParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.group = 1
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            image_c = self._numpy_to_ctensor(np.ascontiguousarray(image_shape.data), image_shape.dtype)
            block_c = self._numpy_to_ctensor(np.ascontiguousarray(block_shape.data), block_shape.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.col2im_forward(input_c, image_c, block_c, out_c, ctypes.byref(c_params))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(image_c)
            self.lib.free_tensor(block_c)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

        from onnx.reference.ops.op_col2im import col2im_naive_implementation

        data = _tensor_data_as_numeric(input)
        block_size = int(np.prod(block_dims, dtype=np.int64))
        reshaped = data.reshape(input.size[0], channels, block_size, input.size[2])
        out = np.empty(out_shape, dtype=np.float32)
        for n in range(input.size[0]):
            for c in range(channels):
                out[n, c] = col2im_naive_implementation(
                    reshaped[n, c],
                    image_dims,
                    tuple(block_dims),
                    dilations,
                    pads,
                    strides,
                )
        out_data = _cast_numeric_to_dtype(out, self.dtype)
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `Col2Im` 的形状推断路径，只生成输出图像张量元数据。
    def forward_(self, input, image_shape, block_shape):
        if hasattr(image_shape, "data") and image_shape.data is not None and hasattr(block_shape, "data") and block_shape.data is not None:
            image_dims, _block_dims, _pads, _strides, _dilations, channels, _n_blocks = self._params(input.size, image_shape.data, block_shape.data)
            return {"tensor": Tensor_(input.size[0], channels, *image_dims, dtype=self.dtype), "parameters": None, "graph": None}
        return {"tensor": Tensor_(input.size[0], 1, dtype=self.dtype), "parameters": None, "graph": None}


class DeformConv(Ops):
    # 初始化 `DeformConv` 的卷积、offset group 和输出 dtype 属性。
    def __init__(
        self,
        inputs,
        outputs,
        strides=None,
        pads=None,
        dilations=None,
        group=1,
        kernel_shape=None,
        offset_group=1,
        dtype="float32",
        version="22",
    ):
        super().__init__(inputs, outputs)
        self.strides = list(strides) if strides is not None else None
        self.pads = list(pads) if pads is not None else None
        self.dilations = list(dilations) if dilations is not None else None
        self.group = int(group)
        self.kernel_shape = list(kernel_shape) if kernel_shape is not None else None
        self.offset_group = int(offset_group)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.deform_conv2d_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CConvParams),
                ctypes.c_int,
            ]

    # 解析并校验 DeformConv 的 2D 主路径参数，供 forward、forward_ 和 C 后端复用。
    def _params(self, x_shape, w_shape, offset_shape):
        if len(x_shape) < 3:
            raise ValueError(f"DeformConv input must have at least 3 dimensions, got {x_shape}")
        spatial_rank = len(x_shape) - 2
        kernel_shape = self.kernel_shape if self.kernel_shape is not None else list(w_shape[2:])
        if len(kernel_shape) != spatial_rank:
            raise ValueError(f"DeformConv kernel rank {len(kernel_shape)} does not match spatial rank {spatial_rank}")
        strides = _conv_attr(self.strides, spatial_rank, 1)
        dilations = _conv_attr(self.dilations, spatial_rank, 1)
        pads = [0] * (2 * spatial_rank) if self.pads is None else list(self.pads)
        if len(pads) != 2 * spatial_rank:
            raise ValueError(f"DeformConv pads must contain {2 * spatial_rank} values")
        if self.group <= 0 or self.offset_group <= 0:
            raise ValueError("DeformConv group and offset_group must be positive")
        if x_shape[1] != w_shape[1] * self.group or w_shape[0] % self.group != 0:
            raise ValueError(f"DeformConv shape mismatch: X={x_shape}, W={w_shape}, group={self.group}")
        expected_offset_channels = self.offset_group * int(np.prod(kernel_shape, dtype=np.int64)) * spatial_rank
        if offset_shape[1] != expected_offset_channels:
            raise ValueError(f"DeformConv offset channel dimension {offset_shape[1]} != expected {expected_offset_channels}")
        expected_spatial = _conv_output_spatial(list(x_shape[2:]), kernel_shape, pads, strides, dilations)
        if tuple(offset_shape[2:]) != tuple(expected_spatial):
            raise ValueError(f"DeformConv offset spatial shape {offset_shape[2:]} != expected {expected_spatial}")
        return kernel_shape, pads, strides, dilations, (x_shape[0], w_shape[0], *expected_spatial)

    # 执行 `DeformConv` 的真实张量计算路径，按 offset 和 mask 对输入采样后累加卷积结果。
    def forward(self, x, w, offset, b=None, mask=None):
        kernel_shape, pads, strides, dilations, out_shape = self._params(x.size, w.size, offset.size)

        if (
            self.lib is not None
            and len(x.size) == 4
            and len(w.size) == 4
            and len(offset.size) == 4
            and x.dtype in nn.DTYPE_MAP
            and w.dtype in nn.DTYPE_MAP
            and offset.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
            and (b is None or b.dtype in nn.DTYPE_MAP)
            and (mask is None or mask.dtype in nn.DTYPE_MAP)
        ):
            pads_c = (ctypes.c_int * len(pads))(*pads)
            strides_c = (ctypes.c_int * len(strides))(*strides)
            dilations_c = (ctypes.c_int * len(dilations))(*dilations)
            c_params = CConvParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.group = self.group

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            w_c = self._numpy_to_ctensor(np.ascontiguousarray(w.data), w.dtype)
            offset_c = self._numpy_to_ctensor(np.ascontiguousarray(offset.data), offset.dtype)
            b_c = self._numpy_to_ctensor(np.ascontiguousarray(b.data), b.dtype) if b is not None else ctypes.POINTER(CTensor)()
            mask_c = self._numpy_to_ctensor(np.ascontiguousarray(mask.data), mask.dtype) if mask is not None else ctypes.POINTER(CTensor)()
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.deform_conv2d_forward(x_c, w_c, offset_c, b_c, mask_c, out_c, ctypes.byref(c_params), ctypes.c_int(self.offset_group))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(w_c)
            self.lib.free_tensor(offset_c)
            if b is not None:
                self.lib.free_tensor(b_c)
            if mask is not None:
                self.lib.free_tensor(mask_c)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

        from onnx.reference.ops.op_deform_conv import _deform_conv_implementation

        out = _deform_conv_implementation(
            _tensor_data_as_numeric(x),
            _tensor_data_as_numeric(w),
            _tensor_data_as_numeric(offset),
            None if b is None else _tensor_data_as_numeric(b),
            None if mask is None else _tensor_data_as_numeric(mask),
            dilations,
            self.group,
            kernel_shape,
            self.offset_group,
            pads,
            strides,
        )
        out_data = _cast_numeric_to_dtype(out, self.dtype)
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `DeformConv` 的形状推断路径，输出 shape 由 offset 的空间维和 W 的输出通道确定。
    def forward_(self, x, w, offset, b=None, mask=None):
        _kernel_shape, _pads, _strides, _dilations, out_shape = self._params(x.size, w.size, offset.size)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None, "graph": None}


class ConvInteger(Ops):
    # 初始化 `ConvInteger` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(
        self,
        inputs,
        outputs,
        pads=None,
        strides=None,
        dilations=None,
        group=1,
        kernel_shape=None,
        auto_pad="NOTSET",
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.pads = list(pads) if pads is not None else None
        self.strides = list(strides) if strides is not None else None
        self.dilations = list(dilations) if dilations is not None else None
        self.group = group
        self.kernel_shape = list(kernel_shape) if kernel_shape is not None else None
        self.auto_pad = auto_pad
        self.dtype = "int32"
        self.version = version
        if self.lib:
            self.lib.conv_integer_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CConvParams)
            ]

    # 封装 `_shape` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _shape(self, x_shape, w_shape):
        spatial_rank = len(x_shape) - 2
        kernel_shape = self.kernel_shape if self.kernel_shape is not None else list(w_shape[2:])
        strides = _conv_attr(self.strides, spatial_rank, 1)
        dilations = _conv_attr(self.dilations, spatial_rank, 1)
        pads = _conv_resolve_pads(list(x_shape[2:]), kernel_shape, self.pads, strides, dilations, self.auto_pad)
        return (x_shape[0], w_shape[0]) + _conv_output_spatial(list(x_shape[2:]), kernel_shape, pads, strides, dilations)

    # 执行 `ConvInteger` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, w, x_zero_point=None, w_zero_point=None):
        spatial_rank = x.data.ndim - 2
        if (
            self.lib is not None
            and spatial_rank == 2
            and x.dtype in nn.DTYPE_MAP
            and w.dtype in nn.DTYPE_MAP
            and (x_zero_point is None or x_zero_point.dtype in nn.DTYPE_MAP)
            and (w_zero_point is None or w_zero_point.dtype in nn.DTYPE_MAP)
        ):
            kernel_shape = self.kernel_shape if self.kernel_shape is not None else list(w.size[2:])
            strides = _conv_attr(self.strides, spatial_rank, 1)
            dilations = _conv_attr(self.dilations, spatial_rank, 1)
            pads = _conv_resolve_pads(list(x.size[2:]), kernel_shape, self.pads, strides, dilations, self.auto_pad)
            out_shape = (x.size[0], w.size[0]) + _conv_output_spatial(list(x.size[2:]), kernel_shape, pads, strides, dilations)

            x_zp = _broadcast_conv_zero_point(x_zero_point, x.data.shape, x.dtype)
            w_zp = _broadcast_conv_zero_point(w_zero_point, w.data.shape, w.dtype, axis=0)

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data.astype(nn.DTYPE_TO_NUMPY[x.dtype], copy=False)), x.dtype)
            w_c = self._numpy_to_ctensor(np.ascontiguousarray(w.data.astype(nn.DTYPE_TO_NUMPY[w.dtype], copy=False)), w.dtype)
            x_zp_c = self._numpy_to_ctensor(x_zp, x.dtype)
            w_zp_c = self._numpy_to_ctensor(w_zp, w.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            pads_c = (ctypes.c_int * len(pads))(*pads)
            strides_c = (ctypes.c_int * len(strides))(*strides)
            dilations_c = (ctypes.c_int * len(dilations))(*dilations)
            c_params = CConvParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.group = self.group

            self.lib.conv_integer_forward(x_c, w_c, x_zp_c, w_zp_c, out_c, ctypes.byref(c_params))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            for c_tensor in (x_c, w_c, x_zp_c, w_zp_c, out_c):
                self.lib.free_tensor(c_tensor)
            return {"tensor": Tensor(*out_shape, dtype="int32", data=out_data), "parameters": None}

        x_i = x.data.astype(np.int32) - _reshape_channel_param(x_zero_point, x.data, 1, np.int32)
        w_i = w.data.astype(np.int32) - _reshape_channel_param(w_zero_point, w.data, 0, np.int32)
        strides = _conv_attr(self.strides, spatial_rank, 1)
        dilations = _conv_attr(self.dilations, spatial_rank, 1)
        out = _conv_nd_numpy(
            x_i,
            w_i,
            pads=self.pads,
            strides=strides,
            dilations=dilations,
            group=self.group,
            auto_pad=self.auto_pad,
            acc_dtype=np.int64,
        ).astype(np.int32)
        return {"tensor": Tensor(*out.shape, dtype="int32", data=out), "parameters": None}

    # 执行 `ConvInteger` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, w, x_zero_point=None, w_zero_point=None):
        return {"tensor": Tensor_(*self._shape(x.size, w.size), dtype="int32"), "parameters": None}


class QLinearConv(Ops):
    # 初始化 `QLinearConv` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(
        self,
        inputs,
        outputs,
        pads=None,
        strides=None,
        dilations=None,
        group=1,
        kernel_shape=None,
        auto_pad="NOTSET",
        dtype="uint8",
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.pads = list(pads) if pads is not None else None
        self.strides = list(strides) if strides is not None else None
        self.dilations = list(dilations) if dilations is not None else None
        self.group = group
        self.kernel_shape = list(kernel_shape) if kernel_shape is not None else None
        self.auto_pad = auto_pad
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.qlinear_conv_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CConvParams)
            ]

    # 封装 `_shape` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _shape(self, x_shape, w_shape):
        spatial_rank = len(x_shape) - 2
        kernel_shape = self.kernel_shape if self.kernel_shape is not None else list(w_shape[2:])
        strides = _conv_attr(self.strides, spatial_rank, 1)
        dilations = _conv_attr(self.dilations, spatial_rank, 1)
        pads = _conv_resolve_pads(list(x_shape[2:]), kernel_shape, self.pads, strides, dilations, self.auto_pad)
        return (x_shape[0], w_shape[0]) + _conv_output_spatial(list(x_shape[2:]), kernel_shape, pads, strides, dilations)

    # 执行 `QLinearConv` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, b=None):
        spatial_rank = x.data.ndim - 2
        out_channels = w.data.shape[0]
        if (
            self.lib is not None
            and spatial_rank == 2
            and x.dtype in nn.DTYPE_MAP
            and w.dtype in nn.DTYPE_MAP
            and x_scale.dtype in nn.DTYPE_MAP
            and w_scale.dtype in nn.DTYPE_MAP
            and y_scale.dtype in nn.DTYPE_MAP
            and y_zero_point.dtype in nn.DTYPE_MAP
            and (x_zero_point is None or x_zero_point.dtype in nn.DTYPE_MAP)
            and (w_zero_point is None or w_zero_point.dtype in nn.DTYPE_MAP)
            and (b is None or b.dtype in nn.DTYPE_MAP)
        ):
            kernel_shape = self.kernel_shape if self.kernel_shape is not None else list(w.size[2:])
            strides = _conv_attr(self.strides, spatial_rank, 1)
            dilations = _conv_attr(self.dilations, spatial_rank, 1)
            pads = _conv_resolve_pads(list(x.size[2:]), kernel_shape, self.pads, strides, dilations, self.auto_pad)
            out_shape = (x.size[0], out_channels) + _conv_output_spatial(list(x.size[2:]), kernel_shape, pads, strides, dilations)

            x_zp = _broadcast_conv_zero_point(x_zero_point, x.data.shape, x.dtype)
            w_zp = _broadcast_conv_zero_point(w_zero_point, w.data.shape, w.dtype, axis=0)
            x_s = _broadcast_conv_param(x_scale, x.data.shape, x_scale.dtype)
            w_s = _broadcast_conv_param(w_scale, w.data.shape, w_scale.dtype, axis=0)
            y_s = _broadcast_conv_param(y_scale, out_shape, y_scale.dtype)
            y_zp = _broadcast_conv_param(y_zero_point, out_shape, self.dtype)

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data.astype(nn.DTYPE_TO_NUMPY[x.dtype], copy=False)), x.dtype)
            x_s_c = self._numpy_to_ctensor(x_s, x_scale.dtype)
            x_zp_c = self._numpy_to_ctensor(x_zp, x.dtype)
            w_c = self._numpy_to_ctensor(np.ascontiguousarray(w.data.astype(nn.DTYPE_TO_NUMPY[w.dtype], copy=False)), w.dtype)
            w_s_c = self._numpy_to_ctensor(w_s, w_scale.dtype)
            w_zp_c = self._numpy_to_ctensor(w_zp, w.dtype)
            y_s_c = self._numpy_to_ctensor(y_s, y_scale.dtype)
            y_zp_c = self._numpy_to_ctensor(y_zp, self.dtype)
            b_c = self._numpy_to_ctensor(np.ascontiguousarray(b.data), b.dtype) if b is not None else ctypes.POINTER(CTensor)()
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            pads_c = (ctypes.c_int * len(pads))(*pads)
            strides_c = (ctypes.c_int * len(strides))(*strides)
            dilations_c = (ctypes.c_int * len(dilations))(*dilations)
            c_params = CConvParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.group = self.group

            self.lib.qlinear_conv_forward(
                x_c, x_s_c, x_zp_c,
                w_c, w_s_c, w_zp_c,
                y_s_c, y_zp_c, b_c,
                out_c, ctypes.byref(c_params)
            )
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            for c_tensor in (x_c, x_s_c, x_zp_c, w_c, w_s_c, w_zp_c, y_s_c, y_zp_c, out_c):
                self.lib.free_tensor(c_tensor)
            if b is not None:
                self.lib.free_tensor(b_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        x_zp = _reshape_channel_param(x_zero_point, x.data, 1, np.int32)
        w_zp = _reshape_channel_param(w_zero_point, w.data, 0, np.int32)
        x_s = _reshape_channel_param(x_scale, x.data, 1, np.float64)
        w_s = _reshape_channel_param(w_scale, w.data, 0, np.float64)
        x_real = (x.data.astype(np.int32) - x_zp).astype(np.float64) * x_s
        w_real = (w.data.astype(np.int32) - w_zp).astype(np.float64) * w_s

        bias = None
        if b is not None:
            raw_x_scale = np.asarray(_tensor_data_as_numeric(x_scale), dtype=np.float64)
            raw_w_scale = np.asarray(_tensor_data_as_numeric(w_scale), dtype=np.float64)
            if raw_w_scale.ndim == 0 or raw_w_scale.size == 1:
                bias_scale = raw_x_scale.reshape(-1)[0] * raw_w_scale.reshape(-1)[0]
            else:
                bias_scale = raw_x_scale.reshape(-1)[0] * raw_w_scale.reshape(-1)
            bias = _tensor_data_as_numeric(b).astype(np.float64) * bias_scale

        strides = _conv_attr(self.strides, spatial_rank, 1)
        dilations = _conv_attr(self.dilations, spatial_rank, 1)
        conv = _conv_nd_numpy(
            x_real,
            w_real,
            bias=bias,
            pads=self.pads,
            strides=strides,
            dilations=dilations,
            group=self.group,
            auto_pad=self.auto_pad,
            acc_dtype=np.float64,
        )
        y_s = _reshape_output_channel_param(y_scale, out_channels, spatial_rank, np.float64)
        y_zp = _reshape_output_channel_param(y_zero_point, out_channels, spatial_rank, np.float64)
        quantized = np.rint(conv / y_s + y_zp)
        low, high = _dtype_bounds(self.dtype)
        if low is not None:
            quantized = np.clip(quantized, low, high)
        out = quantized.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, np.uint8))
        return {"tensor": Tensor(*out.shape, dtype=self.dtype, data=out), "parameters": None}

    # 执行 `QLinearConv` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, b=None):
        return {"tensor": Tensor_(*self._shape(x.size, w.size), dtype=self.dtype), "parameters": None}
