# /**
#   ******************************************************************************
#   * @file        shape_constant_ops.py
#   * @author      Egor Izmaylov
#   * @brief       保存 `shape_constant_ops` 分组中的 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *

class Shape(Ops):
    # 初始化 `Shape` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, start=0, end=None, dtype="int64", version="17"):
        super(Shape, self).__init__(inputs, outputs)
        self.start = start
        self.end = end
        self.dtype = "int64" # Shape 输出永远是 int64
        self.version = version

    # 执行 `Shape` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        dims = list(input.size)
        # 处理 start/end
        end = len(dims) if self.end is None else self.end
        sliced_dims = dims[self.start : end]
        
        out_data = np.array(sliced_dims, dtype=np.int64)
        out_tensor = Tensor(len(sliced_dims), dtype="int64", data=out_data)
        
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # 执行 `Shape` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        # Shape 的输出形状取决于 input 的 rank
        dims = list(input.size)
        end = len(dims) if self.end is None else self.end
        out_len = len(dims[self.start : end])
        return {"tensor": Tensor_(out_len, dtype="int64"), "parameters": None, "graph": None}


class Constant(Ops):
    # 初始化 `Constant` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, value=None, dtype="float32", version="17"):
        super(Constant, self).__init__(inputs, outputs)
        self.value = value
        self.dtype = dtype
        self.version = version

    # 封装 `_value_array` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _value_array(self):
        if isinstance(self.value, Tensor):
            return np.asarray(self.value.data).copy(), tuple(self.value.size)
        if isinstance(self.value, np.ndarray):
            return np.asarray(self.value).copy(), self.value.shape
        arr = np.asarray(self.value, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, None))
        return arr.copy(), arr.shape

    # 执行 `Constant` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self) -> dict:
        out_data, val_shape = self._value_array()
        return {"tensor": Tensor(*val_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `Constant` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self) -> dict:
        # [Fix] 为了支持 Shape 推断，Constant 需要返回真实数据
        out_data, val_shape = self._value_array()
        return {"tensor": Tensor(*val_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}


class ConstantOfShape(Ops):
    # 初始化 `ConstantOfShape` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, value=None, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.value_tensor = None
        if value is not None:
             value_array = np.asarray(value)
             self.value_tensor = Tensor(*value_array.shape, dtype=dtype, data=value_array)
        else:
             # 默认值为 0.0
             self.value_tensor = Tensor(dtype="float32", data=np.array(0.0, dtype=np.float32))
        
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.constant_of_shape_forward.argtypes = [ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor)]

    # 执行 `ConstantOfShape` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, shape_tensor):
        target_shape = tuple(shape_tensor.data.astype(np.int64).flatten().tolist())
        value_data = np.asarray(self.value_tensor.data)
        if value_data.size != 1:
            raise ValueError("ConstantOfShape expects a single-element value tensor")
        if self.lib is not None and self.dtype in nn.DTYPE_MAP and self.value_tensor.dtype in nn.DTYPE_MAP:
            output_shape_c = (ctypes.c_int * len(target_shape))(*target_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(target_shape), nn.DTYPE_MAP[self.dtype])
            value_c = self._numpy_to_ctensor(np.ascontiguousarray(value_data), self.value_tensor.dtype)
            self.lib.constant_of_shape_forward(output_c, value_c)
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(output_c)
            self.lib.free_tensor(value_c)
        else:
            fill_value = value_data.reshape(-1)[0] if value_data.shape else value_data.item()
            out_dtype = nn.DTYPE_TO_NUMPY.get(self.dtype, value_data.dtype)
            out_data = np.full(target_shape, fill_value, dtype=out_dtype)
        return {"tensor": Tensor(*target_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `ConstantOfShape` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, shape_tensor):
        if hasattr(shape_tensor, "data") and shape_tensor.data is not None:
            target_shape = tuple(shape_tensor.data.astype(np.int64).flatten().tolist())
            return {"tensor": Tensor_(*target_shape, dtype=self.dtype), "parameters": None}
        return {"tensor": Tensor_(1, dtype=self.dtype), "parameters": None}


class Range(Ops):
    # 初始化 `Range` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `Range` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, start, limit, delta):
        # max(ceil((limit - start) / delta), 0)
        s = start.data.item()
        l = limit.data.item()
        d = delta.data.item()
        length = max(int(np.ceil((l - s) / d)), 0)
        
        out_shape = (length,)
        start_c = self._numpy_to_ctensor(start.data, start.dtype)
        limit_c = self._numpy_to_ctensor(limit.data, limit.dtype)
        delta_c = self._numpy_to_ctensor(delta.data, delta.dtype)
        output_shape_c = (ctypes.c_int * 1)(length)
        output_c = self.lib.create_tensor(output_shape_c, 1, nn.DTYPE_MAP[self.dtype])
        self.lib.range_forward(start_c, limit_c, delta_c, output_c)
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(start_c); self.lib.free_tensor(limit_c); self.lib.free_tensor(delta_c); self.lib.free_tensor(output_c)
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `Range` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, start, limit, delta):
        if all(hasattr(t, "data") and t.data is not None for t in (start, limit, delta)):
            s = start.data.item()
            l = limit.data.item()
            d = delta.data.item()
            length = max(int(np.ceil((l - s) / d)), 0)
            return {"tensor": Tensor_(length, dtype=self.dtype), "parameters": None}
        return {"tensor": Tensor_(1, dtype=self.dtype), "parameters": None}


class Tile(Ops):
    # 初始化 `Tile` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.tile_forward.argtypes = [ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor)]

    # 执行 `Tile` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input, repeats):
        rep = repeats.data.astype(np.int64).flatten()
        in_shape = np.array(input.size)
        if len(rep) != len(in_shape):
            raise ValueError(f"Tile: repeats dim {len(rep)} != input dim {len(in_shape)}")
            
        out_shape = tuple((in_shape * rep).tolist())
        if self.lib is not None and input.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.tile_forward(input_c, output_c)
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
        else:
            out_data = np.tile(np.asarray(input.data), rep)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `Tile` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input, repeats):
        if hasattr(repeats, "data") and repeats.data is not None:
            rep = repeats.data.astype(np.int64).flatten()
            in_shape = np.array(input.size)
            if len(rep) == len(in_shape):
                out_shape = tuple((in_shape * rep).tolist())
                return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}
        return {"tensor": Tensor_(*input.size, dtype=self.dtype), "parameters": None}


class Pad(Ops):
    # 初始化 `Pad` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, mode="constant", dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.mode = mode # constant, reflect, edge; wrap is kept as a non-ONNX extension
        self.dtype = dtype
        self.version = version
        mode_map = {"constant": 0, "reflect": 1, "edge": 2, "wrap": 3}
        if mode not in mode_map:
            raise NotImplementedError(f"Pad mode {mode!r} is not supported")
        self.mode_int = mode_map[mode]
        if self.lib:
            self.lib.pad_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor),
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor), ctypes.c_int
            ]

    # 封装 `_calc_shape` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _calc_shape(self, data_shape, pads):
        p = np.asarray(pads, dtype=np.int64).flatten()
        ndim = len(data_shape)
        if len(p) != 2 * ndim:
            raise ValueError(f"Pad expects {2 * ndim} pad values for rank {ndim}, got {len(p)}")

        out_shape = []
        for i in range(ndim):
            dim = int(data_shape[i] + p[i] + p[i + ndim])
            if dim < 0:
                raise ValueError(f"Pad produces negative dimension {dim} on axis {i}")
            out_shape.append(dim)
        return tuple(out_shape)

    # 执行 `Pad` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, data, pads, constant_value=None):
        # pads: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        p = pads.data.astype(np.int64).flatten()
        out_shape = self._calc_shape(data.size, p)

        if (
            self.lib is not None
            and data.dtype in nn.DTYPE_MAP
            and pads.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
            and (constant_value is None or constant_value.dtype in nn.DTYPE_MAP)
        ):
            data_c = self._numpy_to_ctensor(np.ascontiguousarray(data.data), data.dtype)
            pads_c = self._numpy_to_ctensor(np.ascontiguousarray(p), "int64")
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            const_c = (
                self._numpy_to_ctensor(np.ascontiguousarray(constant_value.data), constant_value.dtype)
                if constant_value is not None else ctypes.POINTER(nn.CTensor)()
            )
            self.lib.pad_forward(data_c, output_c, pads_c, const_c, ctypes.c_int(self.mode_int))
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(data_c)
            self.lib.free_tensor(pads_c)
            self.lib.free_tensor(output_c)
            if constant_value is not None:
                self.lib.free_tensor(const_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        in_data = np.asarray(data.data)
        ndim = in_data.ndim
        begins = p[:ndim]
        ends = p[ndim:]

        slices = []
        positive_pads = []
        for axis, (begin, end) in enumerate(zip(begins, ends)):
            crop_start = int(max(-begin, 0))
            crop_end = in_data.shape[axis] - int(max(-end, 0))
            slices.append(slice(crop_start, crop_end))
            positive_pads.append((int(max(begin, 0)), int(max(end, 0))))

        cropped = in_data[tuple(slices)]
        if self.mode == "constant":
            if constant_value is None:
                if self.dtype == "string":
                    pad_value = ""
                elif self.dtype == "bool":
                    pad_value = False
                else:
                    pad_value = 0
            else:
                pad_array = np.asarray(constant_value.data)
                pad_value = pad_array.reshape(-1)[0] if pad_array.shape else pad_array.item()
            out_data = np.pad(cropped, positive_pads, mode="constant", constant_values=pad_value)
        else:
            out_data = np.pad(cropped, positive_pads, mode=self.mode)

        out_dtype = nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype)
        out_data = np.asarray(out_data, dtype=out_dtype)
        if tuple(out_data.shape) != out_shape:
            out_data = out_data.reshape(out_shape)
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}
    
    # 执行 `Pad` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, data, pads, constant_value=None):
        if hasattr(pads, "data") and pads.data is not None:
            out_shape = self._calc_shape(data.size, pads.data)
        else:
            out_shape = data.size
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}


class Split(Ops):
    # 初始化 `Split` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        self.version = version
        # Split 复用 Slice
        if self.lib:
            self.lib.slice_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor), 
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
            ]

    # 执行 `Split` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input, split=None):
        axis = self.axis if self.axis >= 0 else self.axis + len(input.size)
        dim_len = input.size[axis]

        if split is not None:
            split_sizes = split.data.astype(np.int64).flatten().tolist()
        else:
            num_outputs = len(self.outputs)
            div, remainder = divmod(dim_len, num_outputs)
            split_sizes = [div + (1 if idx < remainder else 0) for idx in range(num_outputs)]

        if any(size < 0 for size in split_sizes) or sum(split_sizes) != dim_len:
            raise ValueError(f"Split sizes {split_sizes} do not sum to axis dimension {dim_len}")

        if self.lib is not None and input.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            starts = [0] * len(input.size)
            steps = [1] * len(input.size)
            result_tensors = []
            offset = 0
            for size in split_sizes:
                out_shape = list(input.size)
                out_shape[axis] = int(size)
                starts[axis] = offset
                output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
                output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
                starts_c = (ctypes.c_int * len(starts))(*starts)
                steps_c = (ctypes.c_int * len(steps))(*steps)
                self.lib.slice_forward(input_c, output_c, starts_c, steps_c)
                out_data = self._ctensor_to_numpy(output_c, self.dtype)
                result_tensors.append(Tensor(*out_shape, dtype=self.dtype, data=out_data))
                self.lib.free_tensor(output_c)
                offset += int(size)
            self.lib.free_tensor(input_c)
        else:
            if split is not None:
                split_points = np.cumsum(split_sizes)[:-1]
                arrays = np.split(np.asarray(input.data), split_points, axis=axis)
            else:
                arrays = np.array_split(np.asarray(input.data), len(self.outputs), axis=axis)
            result_tensors = [
                Tensor(*array.shape, dtype=self.dtype, data=np.asarray(array, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, array.dtype)))
                for array in arrays
            ]
        return {"tensor": result_tensors, "parameters": None}

    # 执行 `Split` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input, split=None):
        num_outputs = len(self.outputs)
        axis = self.axis if self.axis >= 0 else self.axis + len(input.size)
        if len(input.size) <= axis:
            out_shapes = [input.size] * num_outputs
        elif split is not None and hasattr(split, "data") and split.data is not None:
            split_sizes = split.data.astype(np.int64).flatten().tolist()
            out_shapes = []
            for size in split_sizes[:num_outputs]:
                out_shape = list(input.size)
                out_shape[axis] = int(size)
                out_shapes.append(tuple(out_shape))
            while len(out_shapes) < num_outputs:
                out_shapes.append(tuple(input.size))
        else:
            dim_len = input.size[axis]
            div, remainder = divmod(dim_len, num_outputs)
            split_sizes = [div + (1 if idx < remainder else 0) for idx in range(num_outputs)]
            out_shapes = []
            for size in split_sizes:
                out_shape = list(input.size)
                out_shape[axis] = int(size)
                out_shapes.append(tuple(out_shape))

        return {"tensor": [Tensor_(*shape, dtype=self.dtype) for shape in out_shapes], "parameters": None}
