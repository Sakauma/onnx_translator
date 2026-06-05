# /**
#   ******************************************************************************
#   * @file        shape_extra_ops.py
#   * @author      Egor Izmaylov
#   * @brief       保存 `shape_extra_ops` 分组中的 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *

class AffineGrid(Ops):
    # 初始化 `AffineGrid` 的构造参数，保存 align_corners、dtype 和版本信息。
    def __init__(self, inputs, outputs, align_corners=0, dtype="float32", version="20"):
        super().__init__(inputs, outputs)
        self.align_corners = align_corners
        self.dtype = dtype
        self.version = version
        self._has_affine_grid_c_backend = False
        if self.lib:
            try:
                self.lib.affine_grid_forward.argtypes = [
                    ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int
                ]
                self._has_affine_grid_c_backend = True
            except AttributeError:
                self._has_affine_grid_c_backend = False

    # 根据 size 输入解析 AffineGrid 的输出形状。
    def _output_shape(self, theta, size_tensor):
        size_values = np.asarray(size_tensor.data, dtype=np.int64).reshape(-1).tolist()
        if len(size_values) not in (4, 5):
            raise ValueError(f"AffineGrid size must be rank 4 or 5, got {size_values}")
        batch = int(size_values[0])
        spatial = tuple(int(v) for v in size_values[2:])
        coord_dim = len(spatial)
        expected_theta = (batch, coord_dim, coord_dim + 1)
        if tuple(theta.size) != expected_theta:
            raise ValueError(f"AffineGrid theta shape must be {expected_theta}, got {theta.size}")
        return (batch,) + spatial + (coord_dim,)

    # 执行 `AffineGrid` 的真实张量计算路径，生成 2D/3D 规范化采样网格。
    def forward(self, theta, size):
        out_shape = self._output_shape(theta, size)
        if self._has_affine_grid_c_backend and theta.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            theta_c = self._numpy_to_ctensor(np.ascontiguousarray(theta.data), theta.dtype)
            size_c = self._numpy_to_ctensor(np.ascontiguousarray(size.data), size.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.affine_grid_forward(theta_c, size_c, out_c, ctypes.c_int(self.align_corners))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(theta_c)
            self.lib.free_tensor(size_c)
            self.lib.free_tensor(out_c)
        else:
            from onnx.reference.ops.op_affine_grid import apply_affine_transform, construct_original_grid

            size_values = np.asarray(size.data, dtype=np.int64).reshape(-1).tolist()
            original_grid = construct_original_grid(size_values[2:], self.align_corners)
            out_data = apply_affine_transform(_tensor_data_as_numeric(theta), original_grid)
            out_data = _cast_numeric_to_dtype(out_data, self.dtype)
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `AffineGrid` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, theta, size):
        if hasattr(size, "data") and size.data is not None:
            return {"tensor": Tensor_(*self._output_shape(theta, size), dtype=self.dtype), "parameters": None, "graph": None}
        rank = len(theta.size) - 1
        coord_dim = theta.size[1] if len(theta.size) == 3 else 2
        return {"tensor": Tensor_(*((theta.size[0],) + (1,) * coord_dim + (coord_dim,)), dtype=self.dtype), "parameters": None, "graph": None}


class Expand(Ops):
    # 初始化 `Expand` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super(Expand, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.expand_forward.argtypes = [ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor)]

    # 执行 `Expand` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor, shape: Tensor) -> dict:
        # 1. 获取目标形状
        target_shape = shape.data.astype(np.int64).flatten().tolist()
        input_shape = list(input.size)
        
        # 2. 检查维度数量 Target 维度不能少于 Input
        if len(target_shape) < len(input_shape):
             raise ValueError(f"Expand: Target shape dims ({len(target_shape)}) < Input dims ({len(input_shape)}). Input: {input_shape}, Target: {target_shape}")

        # 3. 对齐输入维度 (Input 左侧补 1)
        pad_len = len(target_shape) - len(input_shape)
        aligned_input = [1] * pad_len + input_shape
        
        # 4. 逐维度计算最终形状并检查合法性
        final_shape = []
        for i, (t_dim, i_dim) in enumerate(zip(target_shape, aligned_input)):
            # 情况 A: target 为 -1，表示维持 input 维度
            if t_dim == -1:
                final_shape.append(i_dim)
            # 情况 B: input 为 1，广播到 target 维度
            elif i_dim == 1:
                final_shape.append(t_dim)
            # 情况 C: 维度匹配，无需广播
            elif i_dim == t_dim:
                final_shape.append(t_dim)
            # 情况 D: 维度不匹配且 input != 1 (Expand 不支持缩小或错配)
            # 例如: input=5, target=1 (非法) 或 input=5, target=6 (非法)
            else:
                raise ValueError(f"Expand: Dimension mismatch at axis {i}. Input dim {i_dim} cannot be broadcast to target dim {t_dim}.")
                
        final_shape = tuple(final_shape)
        if self.lib is not None and input.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            output_shape_c = (ctypes.c_int * len(final_shape))(*final_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(final_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.expand_forward(input_c, output_c)
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
        else:
            out_data = np.broadcast_to(np.asarray(input.data), final_shape).copy()
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        
        return {"tensor": Tensor(*final_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `Expand` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_, shape: Tensor_) -> dict:
        if hasattr(shape, "data") and shape.data is not None:
            try:
                target_shape = shape.data.astype(np.int64).flatten().tolist()
                input_shape = list(input.size)
                if len(target_shape) >= len(input_shape):
                    pad_len = len(target_shape) - len(input_shape)
                    aligned_input = [1] * pad_len + input_shape
                    final_shape = []
                    for t_dim, i_dim in zip(target_shape, aligned_input):
                        if t_dim == -1:
                            final_shape.append(i_dim)
                        elif i_dim == 1 or i_dim == t_dim:
                            final_shape.append(t_dim)
                        else:
                            raise ValueError
                    return {"tensor": Tensor_(*tuple(final_shape), dtype=self.dtype), "parameters": None, "graph": None}
            except Exception:
                pass
        return {"tensor": Tensor_(1, dtype=self.dtype), "parameters": None, "graph": None}


class Resize(Ops):
    # 初始化 `Resize` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(
        self,
        inputs,
        outputs,
        mode="nearest",
        coord_mode="asymmetric",
        nearest_mode="floor",
        cubic_coeff_a=-0.75,
        exclude_outside=0,
        extrapolation_value=0.0,
        dtype="float32",
        version="17",
    ):
        super().__init__(inputs, outputs)
        # mode: 0=nearest, 1=linear
        self.mode_str = mode
        self.mode = 1 if mode == "linear" else 0
        self.coord_mode_str = coord_mode
        self.nearest_mode_str = nearest_mode
        self.cubic_coeff_a = cubic_coeff_a
        self.exclude_outside = exclude_outside
        self.extrapolation_value = extrapolation_value
        self.coord_mode = {"half_pixel": 0, "asymmetric": 1, "pytorch_half_pixel": 2, "align_corners": 4}.get(coord_mode, 1)
        # nearest_mode 映射: 0=round_prefer_floor, 2=floor, 3=ceil
        self.nearest_mode = {"round_prefer_floor": 0, "floor": 2, "ceil": 3}.get(nearest_mode, 0)
        
        self.dtype = dtype
        self.version = version
        
        if self.lib:
             self.lib.resize_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_float), 
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]

    # 封装 `_should_use_reference` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _should_use_reference(self):
        return (
            self.lib is None
            or self.mode_str == "cubic"
            or self.coord_mode_str == "tf_crop_and_resize"
            or self.nearest_mode_str == "round_prefer_ceil"
            or self.cubic_coeff_a != -0.75
            or self.exclude_outside != 0
            or self.extrapolation_value != 0.0
        )

    # 封装 `_run_reference_resize` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _run_reference_resize(self, x, roi, scales, sizes, out_shape, scales_data):
        from onnx import helper
        from onnx.reference import ReferenceEvaluator

        dtype_to_onnx = {value: key for key, value in nn.onnx_dtype_mapping.items()}
        elem_type = dtype_to_onnx.get(x.dtype, 1)
        inputs = ["X"]
        graph_inputs = [helper.make_tensor_value_info("X", elem_type, list(x.data.shape))]
        feeds = {"X": x.data}
        if roi is not None and getattr(roi, "data", np.array([])).size > 0:
            inputs.append("roi")
            graph_inputs.append(helper.make_tensor_value_info("roi", dtype_to_onnx.get(roi.dtype, 1), list(roi.data.shape)))
            feeds["roi"] = roi.data
        else:
            inputs.append("")
        if scales is not None and getattr(scales, "data", np.array([])).size > 0:
            inputs.append("scales")
            graph_inputs.append(helper.make_tensor_value_info("scales", dtype_to_onnx.get(scales.dtype, 1), list(scales.data.shape)))
            feeds["scales"] = _tensor_data_as_numeric(scales).astype(np.float32, copy=False)
        elif sizes is None:
            inputs.append("scales")
            graph_inputs.append(helper.make_tensor_value_info("scales", 1, list(scales_data.shape)))
            feeds["scales"] = scales_data.astype(np.float32, copy=False)
        else:
            inputs.append("")
        if sizes is not None and getattr(sizes, "data", np.array([])).size > 0:
            inputs.append("sizes")
            graph_inputs.append(helper.make_tensor_value_info("sizes", dtype_to_onnx.get(sizes.dtype, 7), list(sizes.data.shape)))
            feeds["sizes"] = sizes.data.astype(np.int64, copy=False)

        node = helper.make_node(
            "Resize",
            inputs,
            ["Y"],
            mode=self.mode_str,
            coordinate_transformation_mode=self.coord_mode_str,
            nearest_mode=self.nearest_mode_str,
            cubic_coeff_a=float(self.cubic_coeff_a),
            exclude_outside=int(self.exclude_outside),
            extrapolation_value=float(self.extrapolation_value),
        )
        graph = helper.make_graph(
            [node],
            "resize_reference",
            graph_inputs,
            [helper.make_tensor_value_info("Y", elem_type, list(out_shape))],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        return ReferenceEvaluator(model).run(None, feeds)[0]

    # 执行 `Resize` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, roi=None, scales=None, sizes=None):
        in_shape = np.array(x.size)
        
        # 参数解析逻辑
        if scales is not None and scales.data.size > 0:
            s = _tensor_data_as_numeric(scales).flatten()
            out_shape = tuple((in_shape * s).astype(int).tolist())
            scales_data = s.astype(np.float32)
        elif sizes is not None and sizes.data.size > 0:
            target_size = sizes.data.astype(int).flatten()
            out_shape = tuple(target_size.tolist())
            # 重新计算 scales 传给 C (Resize 需要 scales 进行坐标反变换)
            scales_data = (target_size.astype(np.float32) / in_shape.astype(np.float32))
        else:
            raise ValueError("Resize requires scales or sizes")

        if self._should_use_reference():
            out_data = self._run_reference_resize(x, roi, scales, sizes, out_shape, scales_data)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
            return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}
	            
        x_c = self._numpy_to_ctensor(x.data, self.dtype)
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
        
        scales_arr = (ctypes.c_float * len(scales_data))(*scales_data)

        self.lib.resize_forward(
            x_c, output_c, scales_arr, 
            ctypes.c_int(self.coord_mode), 
            ctypes.c_int(self.mode), # 0=nearest, 1=linear
            ctypes.c_int(self.nearest_mode)
        )
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(output_c)
        
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `Resize` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, roi=None, scales=None, sizes=None):
        in_shape = np.array(x.size, dtype=np.int64)
        if sizes is not None and hasattr(sizes, "data") and sizes.data is not None and sizes.data.size > 0:
            out_shape = tuple(sizes.data.astype(np.int64).flatten().tolist())
            return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}
        if scales is not None and hasattr(scales, "data") and scales.data is not None and scales.data.size > 0:
            out_shape = tuple((in_shape * _tensor_data_as_numeric(scales).astype(np.float64).flatten()).astype(np.int64).tolist())
            return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Size(Ops):
    # 初始化 `Size` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="int64", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = "int64" # Size always returns int64
        self.version = version

        if self.lib:
            self.lib.size_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)
            ]

    # 执行 `Size` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        if self.lib is not None and x.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * 0)()
            output_c = self.lib.create_tensor(output_shape_c, 0, nn.DTYPE_MAP[self.dtype])
            self.lib.size_forward(input_c, output_c)
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(dtype=self.dtype, data=out_data.reshape(())), "parameters": None}
        return {
            "tensor": Tensor(dtype=self.dtype, data=np.array(int(np.prod(x.size, dtype=np.int64)), dtype=np.int64)),
            "parameters": None,
        }

    # 执行 `Size` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        return {"tensor": Tensor_(dtype="int64"), "parameters": None}


class Tril(Ops):
    # 初始化 `Tril` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, k=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.k = k 
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.triangular_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int]

    # 执行 `Tril` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, k_tensor=None):
        k_val = self.k
        if k_tensor is not None:
            k_val = int(k_tensor.data.item())
            
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        
        self.lib.triangular_forward(x_c, out_c, ctypes.c_int(k_val), ctypes.c_int(0))
        
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}

    # 执行 `Tril` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, k_tensor=None): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Triu(Ops):
    # 初始化 `Triu` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, k=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.k = k
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.triangular_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int]

    # 执行 `Triu` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, k_tensor=None):
        k_val = self.k
        if k_tensor is not None:
            k_val = int(k_tensor.data.item())
            
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        
        self.lib.triangular_forward(x_c, out_c, ctypes.c_int(k_val), ctypes.c_int(1))
        
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}

    # 执行 `Triu` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, k_tensor=None): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Trilu(Ops):
    # 初始化 `Trilu` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, upper=1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.upper = upper
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.triangular_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int
            ]

    # 执行 `Trilu` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, k_tensor=None):
        k_val = int(k_tensor.data.item()) if k_tensor is not None else 0
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(x.size))(*x.size)
            out_c = self.lib.create_tensor(output_shape_c, len(x.size), nn.DTYPE_MAP[self.dtype])
            self.lib.triangular_forward(x_c, out_c, ctypes.c_int(k_val), ctypes.c_int(int(self.upper)))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*x.size, dtype=self.dtype, data=out_data), "parameters": None}

        fn = np.triu if self.upper else np.tril
        out_data = fn(x.data, k=k_val)
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*x.size, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `Trilu` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, k_tensor=None):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class DepthToSpace(Ops):
    # 初始化 `DepthToSpace` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, blocksize, mode="DCR", dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.blocksize = blocksize
        self.mode_str = mode
        self.mode = 0 if mode == "DCR" else 1
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.depth_to_space_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int
            ]

    # 执行 `DepthToSpace` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input):
        N, C, H, W = input.size
        bs = self.blocksize
        
        if self.mode == 0: # DCR
            new_C = C // (bs * bs)
        else: # CRD
            new_C = C // (bs * bs)
            
        out_shape = (N, new_C, H * bs, W * bs)
        
        out_tensor = Tensor(*out_shape, dtype=self.dtype)
        
        in_c = self._numpy_to_ctensor(input.data, input.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        
        self.lib.depth_to_space_forward(in_c, out_c, ctypes.c_int(bs), ctypes.c_int(self.mode))
        
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(in_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}

    # 执行 `DepthToSpace` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input):
        N, C, H, W = input.size
        bs = self.blocksize
        new_C = C // (bs * bs)
        out_shape = (N, new_C, H * bs, W * bs)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}


class SpaceToDepth(Ops):
    # 初始化 `SpaceToDepth` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, blocksize, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.blocksize = blocksize
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.space_to_depth_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int
            ]

    # 执行 `SpaceToDepth` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input):
        N, C, H, W = input.size
        bs = self.blocksize
        if H % bs != 0 or W % bs != 0:
            raise ValueError(f"SpaceToDepth blocksize {bs} must divide spatial shape {(H, W)}")
        out_shape = (N, C * bs * bs, H // bs, W // bs)
        if self.lib is not None and input.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.space_to_depth_forward(input_c, output_c, ctypes.c_int(bs))
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        data = np.asarray(input.data)
        out_data = data.reshape(N, C, H // bs, bs, W // bs, bs)
        out_data = out_data.transpose(0, 3, 5, 1, 2, 4).reshape(out_shape)
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, data.dtype))
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `SpaceToDepth` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input):
        N, C, H, W = input.size
        bs = self.blocksize
        out_shape = (N, C * bs * bs, H // bs, W // bs)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}


class ReverseSequence(Ops):
    # 初始化 `ReverseSequence` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, time_axis=0, batch_axis=1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.time_axis = time_axis
        self.batch_axis = batch_axis
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.reverse_sequence_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.c_int, ctypes.c_int
            ]

    # 执行 `ReverseSequence` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input, sequence_lens):
        out_tensor = Tensor(*input.size, dtype=self.dtype)
        
        in_c = self._numpy_to_ctensor(input.data, input.dtype)
        seq_c = self._numpy_to_ctensor(sequence_lens.data, sequence_lens.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        
        self.lib.reverse_sequence_forward(in_c, seq_c, out_c, ctypes.c_int(self.time_axis), ctypes.c_int(self.batch_axis))
        
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(in_c); self.lib.free_tensor(seq_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}

    # 执行 `ReverseSequence` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input, sequence_lens):
        return {"tensor": Tensor_(*input.size, dtype=self.dtype), "parameters": None}


class Compress(Ops):
    # 初始化 `Compress` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=None, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.compress_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int
            ]

    # 执行 `Compress` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input, condition):
        cond = np.asarray(condition.data).astype(bool).reshape(-1)
        if self.axis is None:
            out_shape = (int(np.count_nonzero(cond)),)
            if (
                self.lib is not None
                and cond.size <= int(np.prod(input.size))
                and input.dtype in nn.DTYPE_MAP
                and condition.dtype in nn.DTYPE_MAP
                and self.dtype in nn.DTYPE_MAP
            ):
                input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
                cond_c = self._numpy_to_ctensor(np.ascontiguousarray(cond.astype(nn.DTYPE_TO_NUMPY[condition.dtype])), condition.dtype)
                output_shape_c = (ctypes.c_int * 1)(*out_shape)
                output_c = self.lib.create_tensor(output_shape_c, 1, nn.DTYPE_MAP[self.dtype])
                self.lib.compress_forward(input_c, cond_c, output_c, ctypes.c_int(-len(input.size) - 1))
                out_data = self._ctensor_to_numpy(output_c, self.dtype)
                self.lib.free_tensor(input_c)
                self.lib.free_tensor(cond_c)
                self.lib.free_tensor(output_c)
            else:
                out_data = np.compress(cond, np.asarray(input.data).reshape(-1), axis=0)
                out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        real_axis = self.axis if self.axis >= 0 else self.axis + len(input.size)
        if real_axis < 0 or real_axis >= len(input.size):
            raise ValueError(f"Compress axis {self.axis} is out of bounds for rank {len(input.size)}")
        if (
            self.lib is not None
            and cond.size <= input.size[real_axis]
            and input.dtype in nn.DTYPE_MAP
            and condition.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
        ):
            out_shape = list(input.size)
            out_shape[real_axis] = int(np.count_nonzero(cond))
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            cond_c = self._numpy_to_ctensor(np.ascontiguousarray(cond.astype(nn.DTYPE_TO_NUMPY[condition.dtype])), condition.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.compress_forward(input_c, cond_c, output_c, ctypes.c_int(real_axis))
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(cond_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        out_data = np.compress(cond, np.asarray(input.data), axis=real_axis)
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `Compress` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input, condition):
        if condition is not None and hasattr(condition, "data") and condition.data is not None:
            num_kept = int(np.count_nonzero(condition.data))
        else:
            num_kept = 1
        if self.axis is None:
            return {"tensor": Tensor_(num_kept, dtype=self.dtype), "parameters": None}
        out_shape = list(input.size)
        real_axis = self.axis if self.axis >= 0 else self.axis + len(input.size)
        out_shape[real_axis] = num_kept
        return {"tensor": Tensor_(*tuple(out_shape), dtype=self.dtype), "parameters": None}


class ScatterElements(Ops):
    # 初始化 `ScatterElements` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=0, reduction="none", dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.reduction = {"none": 0, "add": 1, "mul": 2}.get(reduction, 0)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.scatter_elements_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.c_int, ctypes.c_int
            ]

    # 执行 `ScatterElements` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, data, indices, updates):
        out_tensor = Tensor(*data.size, dtype=self.dtype, data=data.data.copy())
        
        d_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        i_c = self._numpy_to_ctensor(indices.data, indices.dtype)
        u_c = self._numpy_to_ctensor(updates.data, updates.dtype)
        
        self.lib.scatter_elements_forward(d_c, i_c, u_c, ctypes.c_int(self.axis), ctypes.c_int(self.reduction))
        
        out_tensor.data = self._ctensor_to_numpy(d_c, self.dtype)
        self.lib.free_tensor(d_c); self.lib.free_tensor(i_c); self.lib.free_tensor(u_c)
        return {"tensor": out_tensor, "parameters": None}

    # 执行 `ScatterElements` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, data, indices, updates):
        return {"tensor": Tensor_(*data.size, dtype=self.dtype), "parameters": None}
