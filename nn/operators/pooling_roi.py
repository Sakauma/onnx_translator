# /**
#   ******************************************************************************
#   * @file        pooling_roi.py
#   * @author      Egor Izmaylov
#   * @brief       按算子职责分组保存 `pooling_roi` 相关 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *

class GridSample(Ops):
    # 初始化 `GridSample` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, mode="bilinear", padding_mode="zeros", align_corners=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = bool(align_corners)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.grid_sample_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]

    # 封装 `_mode_code` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _mode_code(self):
        mode = {"linear": "bilinear", "cubic": "bicubic"}.get(self.mode, self.mode)
        if mode == "bilinear":
            return 0
        if mode == "nearest":
            return 1
        if mode == "bicubic":
            return 2
        raise ValueError(f"Unsupported GridSample mode {self.mode!r}")

    # 封装 `_padding_mode_code` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _padding_mode_code(self):
        if self.padding_mode == "zeros":
            return 0
        if self.padding_mode == "border":
            return 1
        if self.padding_mode == "reflection":
            return 2
        raise ValueError(f"Unsupported GridSample padding_mode {self.padding_mode!r}")

    # 执行 `GridSample` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, grid):
        data = np.asarray(x.data)
        grid_data = np.asarray(grid.data)
        if data.ndim != 4 or grid_data.ndim != 4 or grid_data.shape[-1] != 2:
            raise ValueError(f"GridSample expects X [N,C,H,W] and grid [N,Hout,Wout,2], got {data.shape}, {grid_data.shape}")
        n_batches, channels, height, width = data.shape
        if grid_data.shape[0] != n_batches:
            raise ValueError("GridSample batch dimensions must match")
        h_out, w_out = grid_data.shape[1], grid_data.shape[2]
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and grid.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            out_shape = (n_batches, channels, h_out, w_out)
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            grid_c = self._numpy_to_ctensor(np.ascontiguousarray(grid.data), grid.dtype)
            output_shape_c = (ctypes.c_int * 4)(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, 4, nn.DTYPE_MAP[self.dtype])
            self.lib.grid_sample_forward(
                x_c,
                grid_c,
                output_c,
                ctypes.c_int(self._mode_code()),
                ctypes.c_int(self._padding_mode_code()),
                ctypes.c_int(int(self.align_corners)),
            )
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(grid_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        out = np.empty((n_batches, channels, h_out, w_out), dtype=np.float64)
        for n in range(n_batches):
            for oy in range(h_out):
                for ox in range(w_out):
                    x_norm, y_norm = grid_data[n, oy, ox]
                    in_x = _grid_denormalize(float(x_norm), width, self.align_corners)
                    in_y = _grid_denormalize(float(y_norm), height, self.align_corners)
                    mode = {"linear": "bilinear", "cubic": "bicubic"}.get(self.mode, self.mode)
                    if mode == "nearest":
                        sample_y = int(np.rint(_sample_coordinate(in_y, height, self.padding_mode, self.align_corners)))
                        sample_x = int(np.rint(_sample_coordinate(in_x, width, self.padding_mode, self.align_corners)))
                        for c in range(channels):
                            out[n, c, oy, ox] = _get_pixel_2d(data[n, c], sample_y, sample_x, self.padding_mode, self.align_corners)
                    elif mode == "bilinear":
                        for c in range(channels):
                            out[n, c, oy, ox] = _bilinear_sample_2d(data[n, c], in_y, in_x, self.padding_mode, self.align_corners)
                    elif mode == "bicubic":
                        for c in range(channels):
                            out[n, c, oy, ox] = _bicubic_sample_2d(data[n, c], in_y, in_x, self.padding_mode, self.align_corners)
                    else:
                        raise ValueError(f"Unsupported GridSample mode {self.mode!r}")
        out_data = out.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, data.dtype), copy=False)
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `GridSample` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, grid):
        return {"tensor": Tensor_(x.size[0], x.size[1], grid.size[1], grid.size[2], dtype=self.dtype), "parameters": None}


class MaxPool(Ops):
    # 初始化 `MaxPool` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, kernel_shape, pads, strides, dtype, dilations=[1, 1], ceil_mode=0, storage_order=0, auto_pad="NOTSET", version="17"):
        super(MaxPool, self).__init__(inputs, outputs)
        self.kernel_shape = kernel_shape
        self.pads = pads
        self.strides = strides
        self.dilations = dilations
        self.ceil_mode = ceil_mode
        self.storage_order = storage_order
        self.auto_pad = auto_pad
        self.dtype = dtype
        self.version = version

        if self.lib:
            self.lib.max_pool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CPoolParams)
            ]

    # 执行 `MaxPool` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x: Tensor) -> dict:
        if (
            self.lib is not None
            and x.data.ndim == 4
            and len(self.kernel_shape) == 2
            and not (len(self.outputs) > 1 and self.outputs[1])
            and x.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
        ):
            _rank, pads, strides, dilations = _normalize_pool_params(
                x.size, self.kernel_shape, self.pads, self.strides, self.dilations, self.auto_pad
            )
            out_shape = _pool_output_shape(
                x.size, self.kernel_shape, pads, strides, dilations, self.ceil_mode, "NOTSET"
            )

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            pads_c = (ctypes.c_int * len(pads))(*pads)
            strides_c = (ctypes.c_int * len(strides))(*strides)
            dilations_c = (ctypes.c_int * len(dilations))(*dilations)
            kernel_c = (ctypes.c_int * len(self.kernel_shape))(*self.kernel_shape)
            c_params = CPoolParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.kernel_shape = ctypes.cast(kernel_c, ctypes.POINTER(ctypes.c_int))

            self.lib.max_pool_forward(x_c, out_c, ctypes.byref(c_params))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
            out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
            values = {"tensor": out_tensor, "parameters": None, "graph": None}
            self.parameters = {"values": values}
            return values

        out_data, indices_data = _max_pool_nd(
            x.data, self.kernel_shape, self.pads, self.strides, self.dilations, self.ceil_mode, self.storage_order, self.auto_pad
        )
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        out_shape = out_data.shape
        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        if len(self.outputs) > 1 and self.outputs[1]:
            indices_tensor = Tensor(*indices_data.shape, dtype="int64", data=indices_data)
            values = {"tensor": (out_tensor, indices_tensor), "parameters": None, "graph": None}
            self.parameters = {"values": values}
            return values

        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # 执行 `MaxPool` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x: Tensor_) -> dict:
        out_shape = _pool_output_shape(x.size, self.kernel_shape, self.pads, self.strides, self.dilations, self.ceil_mode, self.auto_pad)
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        if len(self.outputs) > 1 and self.outputs[1]:
            values = {"tensor": (output_tensor, Tensor_(*out_shape, dtype="int64")), "parameters": None, "graph": None}
            self.parameters = {"values": values}
            return values
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values


class MaxUnpool(Ops):
    # 初始化 `MaxUnpool` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, kernel_shape, pads=None, strides=None, dtype="float32", version="17"):
        super(MaxUnpool, self).__init__(inputs, outputs)
        self.kernel_shape = list(kernel_shape)
        spatial_rank = len(self.kernel_shape)
        self.pads = list(pads) if pads is not None else [0] * (2 * spatial_rank)
        self.strides = list(strides) if strides is not None else [1] * spatial_rank
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.max_unpool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CPoolParams)
            ]

    # 封装 `_inferred_shape` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _inferred_shape(self, x_shape):
        spatial_rank = len(x_shape) - 2
        if spatial_rank != len(self.kernel_shape):
            raise ValueError(f"MaxUnpool kernel rank {len(self.kernel_shape)} does not match input spatial rank {spatial_rank}")
        if len(self.pads) != 2 * spatial_rank:
            raise ValueError(f"MaxUnpool pads must contain {2 * spatial_rank} values")
        if len(self.strides) != spatial_rank:
            raise ValueError(f"MaxUnpool strides must contain {spatial_rank} values")
        out_shape = [x_shape[0], x_shape[1]]
        for dim in range(spatial_rank):
            out_shape.append(
                (x_shape[dim + 2] - 1) * self.strides[dim]
                - self.pads[dim]
                - self.pads[spatial_rank + dim]
                + self.kernel_shape[dim]
            )
        return tuple(out_shape)

    # 执行 `MaxUnpool` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, indices, output_shape=None):
        inferred_shape = self._inferred_shape(x.data.shape)
        shape = tuple(np.asarray(output_shape.data, dtype=np.int64).tolist()) if output_shape is not None else inferred_shape
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and indices.dtype in nn.DTYPE_MAP:
            spatial_rank = x.data.ndim - 2
            pads_c = (ctypes.c_int * len(self.pads))(*self.pads)
            strides_c = (ctypes.c_int * len(self.strides))(*self.strides)
            dilations_c = (ctypes.c_int * spatial_rank)(*([1] * spatial_rank))
            kernel_c = (ctypes.c_int * len(self.kernel_shape))(*self.kernel_shape)
            c_params = CPoolParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.kernel_shape = ctypes.cast(kernel_c, ctypes.POINTER(ctypes.c_int))

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data.astype(nn.DTYPE_TO_NUMPY[x.dtype], copy=False)), x.dtype)
            indices_c = self._numpy_to_ctensor(np.ascontiguousarray(indices.data.astype(nn.DTYPE_TO_NUMPY[indices.dtype], copy=False)), indices.dtype)
            output_shape_c = (ctypes.c_int * len(shape))(*shape)
            out_c = self.lib.create_tensor(output_shape_c, len(shape), nn.DTYPE_MAP[self.dtype])

            self.lib.max_unpool_forward(x_c, indices_c, out_c, ctypes.byref(c_params))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(indices_c)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*shape, dtype=self.dtype, data=out_data), "parameters": None}

        flat = np.zeros((int(np.prod(inferred_shape)),), dtype=x.data.dtype)
        x_flat = x.data.reshape(-1)
        idx_flat = indices.data.reshape(-1).astype(np.int64)
        for pos, value in zip(idx_flat, x_flat):
            flat[pos] = value
        inferred = flat.reshape(inferred_shape)
        out_data = np.zeros(shape, dtype=x.data.dtype)
        slices = tuple(slice(0, dim) for dim in inferred_shape)
        out_data[slices] = inferred
        out_data = out_data.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype), copy=False)
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `MaxUnpool` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, indices, output_shape=None):
        if isinstance(output_shape, Tensor):
            shape = tuple(np.asarray(output_shape.data, dtype=np.int64).tolist())
        else:
            shape = self._inferred_shape(x.size)
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}


class MaxRoiPool(Ops):
    # 初始化 `MaxRoiPool` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, pooled_shape, spatial_scale=1.0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        if len(pooled_shape) != 2:
            raise ValueError("MaxRoiPool pooled_shape must contain [height, width]")
        self.pooled_shape = tuple(int(v) for v in pooled_shape)
        self.spatial_scale = float(spatial_scale)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.max_roi_pool_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
            ]

    # 执行 `MaxRoiPool` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, rois):
        data = np.asarray(x.data)
        roi_data = np.asarray(rois.data)
        if data.ndim != 4 or roi_data.ndim != 2 or roi_data.shape[1] != 5:
            raise ValueError(f"MaxRoiPool expects X [N,C,H,W] and rois [num_rois,5], got {data.shape}, {roi_data.shape}")
        pooled_h, pooled_w = self.pooled_shape
        num_rois, channels = roi_data.shape[0], data.shape[1]
        out_shape = (num_rois, channels, pooled_h, pooled_w)
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and rois.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            rois_c = self._numpy_to_ctensor(np.ascontiguousarray(rois.data), rois.dtype)
            output_shape_c = (ctypes.c_int * 4)(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, 4, nn.DTYPE_MAP[self.dtype])
            self.lib.max_roi_pool_forward(
                x_c,
                rois_c,
                output_c,
                ctypes.c_int(pooled_h),
                ctypes.c_int(pooled_w),
                ctypes.c_float(self.spatial_scale),
            )
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(rois_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        out = np.zeros((num_rois, channels, pooled_h, pooled_w), dtype=data.dtype)
        height, width = data.shape[2], data.shape[3]
        for roi_idx, roi in enumerate(roi_data):
            batch = int(roi[0])
            x1 = int(round(float(roi[1]) * self.spatial_scale))
            y1 = int(round(float(roi[2]) * self.spatial_scale))
            x2 = int(round(float(roi[3]) * self.spatial_scale))
            y2 = int(round(float(roi[4]) * self.spatial_scale))
            roi_w = max(x2 - x1 + 1, 1)
            roi_h = max(y2 - y1 + 1, 1)
            bin_h = float(roi_h) / float(pooled_h)
            bin_w = float(roi_w) / float(pooled_w)
            for ph in range(pooled_h):
                for pw in range(pooled_w):
                    hstart = int(np.floor(ph * bin_h)) + y1
                    hend = int(np.ceil((ph + 1) * bin_h)) + y1
                    wstart = int(np.floor(pw * bin_w)) + x1
                    wend = int(np.ceil((pw + 1) * bin_w)) + x1
                    hstart, hend = min(max(hstart, 0), height), min(max(hend, 0), height)
                    wstart, wend = min(max(wstart, 0), width), min(max(wend, 0), width)
                    if hend <= hstart or wend <= wstart:
                        out[roi_idx, :, ph, pw] = 0
                    else:
                        out[roi_idx, :, ph, pw] = np.max(data[batch, :, hstart:hend, wstart:wend], axis=(1, 2))
        out = out.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, out.dtype), copy=False)
        return {"tensor": Tensor(*out.shape, dtype=self.dtype, data=out), "parameters": None}

    # 执行 `MaxRoiPool` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, rois):
        return {"tensor": Tensor_(rois.size[0], x.size[1], self.pooled_shape[0], self.pooled_shape[1], dtype=self.dtype), "parameters": None}


class RoiAlign(Ops):
    # 初始化 `RoiAlign` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(
        self,
        inputs,
        outputs,
        output_height=1,
        output_width=1,
        spatial_scale=1.0,
        sampling_ratio=0,
        mode="avg",
        coordinate_transformation_mode="half_pixel",
        dtype="float32",
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.mode = mode
        self.coordinate_transformation_mode = coordinate_transformation_mode
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.roi_align_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
            ]

    # 封装 `_mode_code` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _mode_code(self):
        mode = self.mode.lower()
        if mode == "avg":
            return 0
        if mode == "max":
            return 1
        raise ValueError(f"Unsupported RoiAlign mode {self.mode!r}")

    # 封装 `_coordinate_transformation_mode_code` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _coordinate_transformation_mode_code(self):
        mode = self.coordinate_transformation_mode.lower()
        if mode == "half_pixel":
            return 0
        if mode == "output_half_pixel":
            return 1
        raise ValueError(f"Unsupported RoiAlign coordinate_transformation_mode {self.coordinate_transformation_mode!r}")

    # 执行 `RoiAlign` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, rois, batch_indices):
        data = np.asarray(x.data)
        roi_data = np.asarray(rois.data)
        batches = np.asarray(batch_indices.data, dtype=np.int64).reshape(-1)
        if data.ndim != 4 or roi_data.ndim != 2 or roi_data.shape[1] != 4:
            raise ValueError(f"RoiAlign expects X [N,C,H,W] and rois [num_rois,4], got {data.shape}, {roi_data.shape}")
        if len(batches) != roi_data.shape[0]:
            raise ValueError("RoiAlign batch_indices length must match number of rois")
        num_rois, channels = roi_data.shape[0], data.shape[1]
        out_shape = (num_rois, channels, self.output_height, self.output_width)
        if (
            self.lib is not None
            and x.dtype in nn.DTYPE_MAP
            and rois.dtype in nn.DTYPE_MAP
            and batch_indices.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
        ):
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            rois_c = self._numpy_to_ctensor(np.ascontiguousarray(rois.data), rois.dtype)
            batch_c = self._numpy_to_ctensor(np.ascontiguousarray(batch_indices.data), batch_indices.dtype)
            output_shape_c = (ctypes.c_int * 4)(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, 4, nn.DTYPE_MAP[self.dtype])
            self.lib.roi_align_forward(
                x_c,
                rois_c,
                batch_c,
                output_c,
                ctypes.c_int(self.output_height),
                ctypes.c_int(self.output_width),
                ctypes.c_int(self.sampling_ratio),
                ctypes.c_float(self.spatial_scale),
                ctypes.c_int(self._mode_code()),
                ctypes.c_int(self._coordinate_transformation_mode_code()),
            )
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(rois_c)
            self.lib.free_tensor(batch_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        height, width = data.shape[2], data.shape[3]
        out = np.empty((num_rois, channels, self.output_height, self.output_width), dtype=np.float64)
        half_pixel = self.coordinate_transformation_mode.lower() == "half_pixel"
        offset = 0.5 if half_pixel else 0.0
        for roi_idx, roi in enumerate(roi_data):
            batch = int(batches[roi_idx])
            roi_start_w = float(roi[0]) * self.spatial_scale - offset
            roi_start_h = float(roi[1]) * self.spatial_scale - offset
            roi_end_w = float(roi[2]) * self.spatial_scale - offset
            roi_end_h = float(roi[3]) * self.spatial_scale - offset
            roi_w = roi_end_w - roi_start_w
            roi_h = roi_end_h - roi_start_h
            if not half_pixel:
                roi_w = max(roi_w, 1.0)
                roi_h = max(roi_h, 1.0)
            bin_h = roi_h / self.output_height
            bin_w = roi_w / self.output_width
            grid_h = self.sampling_ratio if self.sampling_ratio > 0 else int(np.ceil(roi_h / self.output_height))
            grid_w = self.sampling_ratio if self.sampling_ratio > 0 else int(np.ceil(roi_w / self.output_width))
            grid_h, grid_w = max(grid_h, 1), max(grid_w, 1)
            count = grid_h * grid_w
            for c in range(channels):
                image = data[batch, c]
                for ph in range(self.output_height):
                    for pw in range(self.output_width):
                        values = []
                        for iy in range(grid_h):
                            yy = roi_start_h + ph * bin_h + (iy + 0.5) * bin_h / grid_h
                            for ix in range(grid_w):
                                xx = roi_start_w + pw * bin_w + (ix + 0.5) * bin_w / grid_w
                                if self.mode.lower() == "max":
                                    values.append(max(_roi_align_weighted_terms(image, yy, xx)))
                                else:
                                    values.append(sum(_roi_align_weighted_terms(image, yy, xx)))
                        if self.mode.lower() == "max":
                            out[roi_idx, c, ph, pw] = max(values) if values else 0.0
                        elif self.mode.lower() == "avg":
                            out[roi_idx, c, ph, pw] = sum(values) / count
                        else:
                            raise ValueError(f"Unsupported RoiAlign mode {self.mode!r}")
        out = out.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, data.dtype), copy=False)
        return {"tensor": Tensor(*out.shape, dtype=self.dtype, data=out), "parameters": None}

    # 执行 `RoiAlign` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, rois, batch_indices):
        return {"tensor": Tensor_(rois.size[0], x.size[1], self.output_height, self.output_width, dtype=self.dtype), "parameters": None}


class AveragePool(Ops):
    # 初始化 `AveragePool` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, kernel_shape, pads, strides, dtype, dilations=[1, 1], count_include_pad=0, ceil_mode=0, auto_pad="NOTSET", version="17"):
        super().__init__(inputs, outputs)
        self.kernel_shape = kernel_shape
        self.pads = pads
        self.strides = strides
        self.dilations = dilations
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode
        self.auto_pad = auto_pad
        self.dtype = dtype
        self.version = version

        if self.lib:
            self.lib.average_pool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CPoolParams), ctypes.c_int
            ]

    # 执行 `AveragePool` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        if (
            self.lib is not None
            and x.data.ndim == 4
            and len(self.kernel_shape) == 2
            and x.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
        ):
            _rank, pads, strides, dilations = _normalize_pool_params(
                x.size, self.kernel_shape, self.pads, self.strides, self.dilations, self.auto_pad
            )
            out_shape = _pool_output_shape(
                x.size, self.kernel_shape, pads, strides, dilations, self.ceil_mode, "NOTSET"
            )

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            pads_c = (ctypes.c_int * len(pads))(*pads)
            strides_c = (ctypes.c_int * len(strides))(*strides)
            dilations_c = (ctypes.c_int * len(dilations))(*dilations)
            kernel_c = (ctypes.c_int * len(self.kernel_shape))(*self.kernel_shape)
            c_params = CPoolParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.kernel_shape = ctypes.cast(kernel_c, ctypes.POINTER(ctypes.c_int))

            self.lib.average_pool_forward(x_c, out_c, ctypes.byref(c_params), int(self.count_include_pad))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        out_data = _average_pool_nd(
            x.data, self.kernel_shape, self.pads, self.strides, self.dilations, self.count_include_pad, self.ceil_mode, self.auto_pad
        )
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `AveragePool` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        out_shape = _pool_output_shape(x.size, self.kernel_shape, self.pads, self.strides, self.dilations, self.ceil_mode, self.auto_pad)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}


class LpPool(Ops):
    # 初始化 `LpPool` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, kernel_shape, pads, strides, dtype, p=2, dilations=[1, 1], ceil_mode=0, auto_pad="NOTSET", version="17"):
        super().__init__(inputs, outputs)
        self.kernel_shape = kernel_shape
        self.pads = pads
        self.strides = strides
        self.dilations = dilations
        self.p = p
        self.ceil_mode = ceil_mode
        self.auto_pad = auto_pad
        self.dtype = dtype
        self.version = version

        if self.lib:
            self.lib.lp_pool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CPoolParams), ctypes.c_int
            ]

    # 执行 `LpPool` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        if (
            self.lib is not None
            and x.data.ndim == 4
            and len(self.kernel_shape) == 2
            and x.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
        ):
            _rank, pads, strides, dilations = _normalize_pool_params(
                x.size, self.kernel_shape, self.pads, self.strides, self.dilations, self.auto_pad
            )
            out_shape = _pool_output_shape(
                x.size, self.kernel_shape, pads, strides, dilations, self.ceil_mode, "NOTSET"
            )

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            pads_c = (ctypes.c_int * len(pads))(*pads)
            strides_c = (ctypes.c_int * len(strides))(*strides)
            dilations_c = (ctypes.c_int * len(dilations))(*dilations)
            kernel_c = (ctypes.c_int * len(self.kernel_shape))(*self.kernel_shape)
            c_params = CPoolParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.kernel_shape = ctypes.cast(kernel_c, ctypes.POINTER(ctypes.c_int))

            self.lib.lp_pool_forward(x_c, out_c, ctypes.byref(c_params), int(self.p))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        out_data = _lp_pool_nd(x.data, self.kernel_shape, self.pads, self.strides, self.dilations, self.p, self.ceil_mode, self.auto_pad)
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `LpPool` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        out_shape = _pool_output_shape(x.size, self.kernel_shape, self.pads, self.strides, self.dilations, self.ceil_mode, self.auto_pad)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}


class GlobalAveragePool(Ops):
    # 初始化 `GlobalAveragePool` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.global_average_pool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)
            ]
    
    # 执行 `GlobalAveragePool` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        if len(x.size) < 2:
            raise ValueError("GlobalAveragePool expects input rank >= 2")
        out_shape = tuple(list(x.size[:2]) + [1] * (len(x.size) - 2))
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.global_average_pool_forward(x_c, out_c)
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
        else:
            spatial_axes = tuple(range(2, len(x.size)))
            out_data = np.mean(x.data, axis=spatial_axes, keepdims=True) if spatial_axes else x.data.copy()
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `GlobalAveragePool` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        out_shape = list(x.size)
        for axis in range(2, len(out_shape)):
            out_shape[axis] = 1
        return {"tensor": Tensor_(*tuple(out_shape), dtype=self.dtype), "parameters": None}


class GlobalMaxPool(Ops):
    # 初始化 `GlobalMaxPool` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.global_max_pool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)
            ]
    
    # 执行 `GlobalMaxPool` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        if len(x.size) < 2:
            raise ValueError("GlobalMaxPool expects input rank >= 2")
        out_shape = tuple(list(x.size[:2]) + [1] * (len(x.size) - 2))
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.global_max_pool_forward(x_c, out_c)
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
        else:
            spatial_axes = tuple(range(2, len(x.size)))
            out_data = np.max(x.data, axis=spatial_axes, keepdims=True) if spatial_axes else x.data.copy()
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `GlobalMaxPool` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        out_shape = list(x.size)
        for axis in range(2, len(out_shape)):
            out_shape[axis] = 1
        return {"tensor": Tensor_(*tuple(out_shape), dtype=self.dtype), "parameters": None}


class GlobalLpPool(Ops):
    # 初始化 `GlobalLpPool` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, p=2, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.p = p
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.global_lp_pool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int
            ]

    # 执行 `GlobalLpPool` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        if len(x.size) < 2:
            raise ValueError("GlobalLpPool expects input rank >= 2")
        out_shape = tuple(list(x.size[:2]) + [1] * (len(x.size) - 2))
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.global_lp_pool_forward(x_c, out_c, ctypes.c_int(self.p))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
        else:
            spatial_axes = tuple(range(2, len(x.size)))
            if spatial_axes:
                out_data = np.sum(np.abs(x.data) ** self.p, axis=spatial_axes, keepdims=True) ** (1.0 / self.p)
            else:
                out_data = np.abs(x.data)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `GlobalLpPool` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        out_shape = list(x.size)
        for axis in range(2, len(out_shape)):
            out_shape[axis] = 1
        return {"tensor": Tensor_(*tuple(out_shape), dtype=self.dtype), "parameters": None}
