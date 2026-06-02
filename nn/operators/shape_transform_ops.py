"""文件功能：保存 `shape_transform_ops` 分组中的 ONNX 算子实现。
作者：Egor Izmaylov
时间：2026-06-02
"""

from .common import *

class Flatten(Ops):
    # 初始化 `Flatten` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=1, dtype="float32", version="17"):
        super(Flatten, self).__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        self.version = version

    # 封装 `_calc_shape` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _calc_shape(self, input_shape):
        # 处理 axis 负数情况
        axis = self.axis if self.axis >= 0 else len(input_shape) + self.axis
        dim_0 = 1
        for i in range(axis):
            dim_0 *= input_shape[i]
        dim_1 = 1
        for i in range(axis, len(input_shape)):
            dim_1 *= input_shape[i]
        return (dim_0, dim_1)

    # 执行 `Flatten` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        out_shape = self._calc_shape(input.size)

        if self.lib is not None and input.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            output_shape_c = (ctypes.c_int * 2)(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, 2, nn.DTYPE_MAP[self.dtype])

            self.lib.flatten_forward(input_c, output_c)

            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
        else:
            out_data = np.asarray(input.data).reshape(out_shape)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # 执行 `Flatten` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        out_shape = self._calc_shape(input.size)
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values


class Reshape(Ops):
    # 初始化 `Reshape` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17", allowzero=0):
        super(Reshape, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        self.allowzero = allowzero

    # 封装 `_resolve_shape` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _resolve_shape(self, input_shape, target_shape):
        final_shape = []
        infer_idx = -1
        known_size = 1
        input_size = 1
        for dim in input_shape:
            input_size *= dim

        for i, dim in enumerate(target_shape):
            dim = int(dim)
            if dim == -1:
                if infer_idx != -1:
                    raise ValueError("Reshape target shape can contain at most one -1 dimension")
                infer_idx = i
                final_shape.append(-1)
            elif dim == 0 and not self.allowzero:
                if i >= len(input_shape):
                    raise ValueError("Reshape target shape uses 0 beyond the input rank")
                copied_dim = int(input_shape[i])
                final_shape.append(copied_dim)
                known_size *= copied_dim
            else:
                final_shape.append(dim)
                known_size *= dim

        if infer_idx != -1:
            if known_size == 0:
                if input_size != 0:
                    raise ValueError("Cannot infer Reshape -1 dimension when known dimensions multiply to 0")
                inferred = 0
            else:
                if input_size % known_size != 0:
                    raise ValueError(f"Cannot reshape input of size {input_size} to target {tuple(target_shape)}")
                inferred = input_size // known_size
            final_shape[infer_idx] = inferred
        elif input_size != known_size:
            raise ValueError(f"Cannot reshape input of size {input_size} to target {tuple(final_shape)}")

        return tuple(final_shape)

    # 执行 `Reshape` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, data: Tensor, shape: Tensor) -> dict:
        target_shape = shape.data.astype(np.int64).flatten().tolist()
        final_shape = self._resolve_shape(data.size, target_shape)

        if self.lib is not None and data.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(data.data), data.dtype)
            output_shape_c = (ctypes.c_int * len(final_shape))(*final_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(final_shape), nn.DTYPE_MAP[self.dtype])

            self.lib.reshape_forward(input_c, output_c)

            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
        else:
            out_data = np.asarray(data.data).reshape(final_shape)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))

        out_tensor = Tensor(*final_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # 执行 `Reshape` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, data: Tensor_, shape: Tensor_) -> dict:
        target_shape = None
        
        # 尝试从 shape 参数中获取真实数据
        if hasattr(shape, "data") and shape.data is not None:
            try:
                target_shape = shape.data.astype(np.int64).flatten().tolist()
            except Exception:
                target_shape = None
        
        if target_shape is None:
            print(f"Warning: Reshape (forward_) cannot infer target shape for input {data.size}. Returning input shape.")
            output_tensor = Tensor_(*data.size, dtype=self.dtype)
        else:
            output_tensor = Tensor_(*self._resolve_shape(data.size, target_shape), dtype=self.dtype)

        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values


class Transpose(Ops):
    # 初始化 `Transpose` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, perm=None, dtype="float32", version="17"):
        super(Transpose, self).__init__(inputs, outputs)
        self.perm = None if perm is None else list(perm)
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.transpose_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor), ctypes.POINTER(ctypes.c_int)
            ]

    # 封装 `_resolve_perm` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _resolve_perm(self, rank):
        if self.perm is None or len(self.perm) == 0:
            return list(reversed(range(rank)))
        if len(self.perm) != rank:
            raise ValueError(
                f"❌ Transpose Error: Input rank {rank} does not match perm length {len(self.perm)} ({self.perm})"
            )
        normalized = []
        for ax in self.perm:
            axis = ax + rank if ax < 0 else ax
            if axis < 0 or axis >= rank:
                raise IndexError(f"❌ Transpose Index Error: Perm {self.perm} is out of bounds for rank {rank}")
            if axis in normalized:
                raise ValueError(f"Transpose perm contains duplicate axis {ax}")
            normalized.append(axis)
        return normalized

    # 执行 `Transpose` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        perm = self._resolve_perm(len(input.size))
        out_shape = [input.size[i] for i in perm]

        if self.lib is not None and input.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            perm_arr = (ctypes.c_int * len(perm))(*perm)

            self.lib.transpose_forward(input_c, output_c, perm_arr)

            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
        else:
            out_data = np.transpose(np.asarray(input.data), axes=perm)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # def forward_(self, input: Tensor_) -> dict:
    #     try:
    #         out_shape = [input.size[i] for i in self.perm]
    #     except IndexError:
    #         # 如果维度不够，可能是上游 Reshape 失败。返回一个安全的 dummy
    #         # print(f"[Warning] Transpose input rank {len(input.size)} mismatch perm {self.perm}")
    #         out_shape = input.size
            
    #     output_tensor = Tensor_(*out_shape, dtype=self.dtype)
    #     values = {"tensor": output_tensor, "parameters": None, "graph": None}
    #     self.parameters = {"values": values}
    #     return values
    # 执行 `Transpose` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        perm = self._resolve_perm(len(input.size))
        out_shape = [input.size[i] for i in perm]

        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values


class Squeeze(Ops):
    # 初始化 `Squeeze` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axes=None, dtype="float32", version="17"):
        super(Squeeze, self).__init__(inputs, outputs)
        self.axes = axes
        self.dtype = dtype
        self.version = version

    # 封装 `_calc_shape` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _calc_shape(self, in_shape, axes):
        # 如果 axes 为 None，挤压所有为 1 的维度
        ndim = len(in_shape)
        if axes is not None:
            norm_axes = []
            for ax in axes:
                axis = ax + ndim if ax < 0 else ax
                if axis < 0 or axis >= ndim:
                    raise ValueError(f"Squeeze axis {ax} is out of bounds for input rank {ndim}")
                if axis in norm_axes:
                    raise ValueError(f"Squeeze axis {ax} appears more than once")
                if in_shape[axis] != 1:
                    raise ValueError(f"Cannot squeeze axis {ax} with dimension {in_shape[axis]}")
                norm_axes.append(axis)
        else:
            norm_axes = None

        new_shape = []
        for i, dim in enumerate(in_shape):
            if norm_axes is not None:
                if i in norm_axes:
                    continue # Squeeze
                new_shape.append(dim)
            else:
                if dim != 1:
                    new_shape.append(dim)
        return tuple(new_shape)

    # 执行 `Squeeze` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, data: Tensor, axes: Tensor = None) -> dict:
        # axes 是输入 tensor，不是属性
        target_axes = self.axes
        if axes is not None:
            target_axes = axes.data.flatten().tolist()
        
        out_shape = self._calc_shape(data.size, target_axes)

        if self.lib is not None and data.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(data.data), data.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            self.lib.reshape_forward(input_c, output_c)

            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
        else:
            out_data = np.asarray(data.data).reshape(out_shape)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # 执行 `Squeeze` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, data: Tensor_, axes: Tensor_ = None) -> dict:
        # [Fix] 尝试从输入 Tensor 读取 axes
        target_axes = self.axes
        if target_axes is None and axes is not None and hasattr(axes, 'data') and axes.data is not None:
            try: target_axes = axes.data.flatten().tolist()
            except: pass
            
        if target_axes is not None:
            out_shape = self._calc_shape(data.size, target_axes)
        elif axes is None:
            out_shape = self._calc_shape(data.size, None)
        else:
            out_shape = data.size # 无法获知 axes，保持原样 (比返回 (1,) 安全)
            
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}


class Unsqueeze(Ops):
    # 初始化 `Unsqueeze` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axes=None, dtype="float32", version="17"):
        super(Unsqueeze, self).__init__(inputs, outputs)
        self.axes = axes
        self.dtype = dtype
        self.version = version

    # 封装 `_calc_shape` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _calc_shape(self, in_shape, axes):
        # Unsqueeze: 在指定位置插入维度 1
        output_rank = len(in_shape) + len(axes)
        norm_axes = []
        for ax in axes:
            axis = ax + output_rank if ax < 0 else ax
            if axis < 0 or axis >= output_rank:
                raise ValueError(f"Unsqueeze axis {ax} is out of bounds for output rank {output_rank}")
            if axis in norm_axes:
                raise ValueError(f"Unsqueeze axis {ax} appears more than once")
            norm_axes.append(axis)

        # 排序 axes 以便按顺序插入
        axes = sorted(norm_axes)
        new_shape = list(in_shape)
        for ax in axes:
            new_shape.insert(ax, 1)
        return tuple(new_shape)

    # 执行 `Unsqueeze` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, data: Tensor, axes: Tensor = None) -> dict:
        target_axes = self.axes
        if axes is not None:
            target_axes = axes.data.flatten().tolist()
            
        out_shape = self._calc_shape(data.size, target_axes)

        if self.lib is not None and data.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(data.data), data.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            self.lib.reshape_forward(input_c, output_c)

            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
        else:
            out_data = np.asarray(data.data).reshape(out_shape)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # 执行 `Unsqueeze` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, data: Tensor_, axes: Tensor_ = None) -> dict:
        target_axes = self.axes
        if target_axes is None and axes is not None and hasattr(axes, 'data') and axes.data is not None:
            try: target_axes = axes.data.flatten().tolist()
            except: pass

        if target_axes is not None:
            out_shape = self._calc_shape(data.size, target_axes)
        else:
            out_shape = data.size 
            
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}


class Concat(Ops):
    # 初始化 `Concat` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=0, dtype="float32", version="17"):
        super(Concat, self).__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        self.version = version
        
        # 注册 C 函数参数类型
        if self.lib:
            self.lib.concat_forward.argtypes = [
                ctypes.POINTER(ctypes.POINTER(nn.CTensor)), 
                ctypes.c_int, 
                ctypes.POINTER(nn.CTensor), 
                ctypes.c_int
            ]

    # 封装 `_calc_shape` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _calc_shape(self, input_tensors):
        if not input_tensors:
            raise ValueError("Concat requires at least one input")
        base_shape = list(input_tensors[0].size)
        ndim = len(base_shape)
        axis = self.axis if self.axis >= 0 else self.axis + ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"Concat axis {self.axis} is out of bounds for rank {ndim}")
        
        total_dim = 0
        for t in input_tensors:
            if len(t.size) != ndim:
                raise ValueError(f"Concat input rank mismatch: {t.size} vs {tuple(base_shape)}")
            for dim_idx, (left, right) in enumerate(zip(t.size, base_shape)):
                if dim_idx != axis and left != right:
                    raise ValueError(f"Concat dimension mismatch at axis {dim_idx}: {left} vs {right}")
            total_dim += t.size[axis]
        
        base_shape[axis] = total_dim
        return tuple(base_shape), axis

    # 执行 `Concat` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, *inputs) -> dict:
        input_list = list(inputs)
        out_shape, axis = self._calc_shape(input_list)
        if self.lib is not None and self.dtype in nn.DTYPE_MAP and all(t.dtype in nn.DTYPE_MAP for t in input_list):
            input_ctensors = [self._numpy_to_ctensor(np.ascontiguousarray(t.data), t.dtype) for t in input_list]
            input_array = (ctypes.POINTER(nn.CTensor) * len(input_ctensors))(*input_ctensors)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.concat_forward(input_array, len(input_ctensors), output_c, ctypes.c_int(axis))
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            for c_tensor in input_ctensors:
                self.lib.free_tensor(c_tensor)
            self.lib.free_tensor(output_c)
        else:
            out_data = np.concatenate([np.asarray(tensor.data) for tensor in input_list], axis=axis)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # 执行 `Concat` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, *inputs) -> dict:
        input_list = list(inputs)
        out_shape, _ = self._calc_shape(input_list)
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values


class Slice(Ops):
    # 初始化 `Slice` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super(Slice, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.slice_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor), 
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
            ]

    # 执行 `Slice` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, data: Tensor, starts: Tensor, ends: Tensor, axes: Tensor = None, steps: Tensor = None) -> dict:
        _starts = starts.data.flatten().tolist()
        _ends = ends.data.flatten().tolist()
        _axes = axes.data.flatten().tolist() if axes is not None else list(range(len(_starts)))
        _steps = steps.data.flatten().tolist() if steps is not None else [1] * len(_starts)
        
        ndim = len(data.size)
        
        # 扩展参数至完整维度
        full_starts = [0] * ndim
        full_ends = list(data.size)
        full_steps = [1] * ndim
        
        for i, axis in enumerate(_axes):
            if axis < 0: axis += ndim
            s, e, st = _starts[i], _ends[i], _steps[i]
            
            dim_len = data.size[axis]
            if s < 0: s += dim_len
            if e < 0: e += dim_len
            
            if st > 0:
                # 正向：区间 [0, dim_len]
                s = max(0, min(s, dim_len))
                e = max(0, min(e, dim_len))
            else:
                # 反向：区间 [-1, dim_len-1]
                # end 可以是 -1，表示包含索引 0
                s = max(0, min(s, dim_len - 1))
                e = max(-1, min(e, dim_len - 1))
            
            full_starts[axis] = s
            full_ends[axis] = e
            full_steps[axis] = st
            
        out_shape = []
        for i in range(ndim):
            if full_steps[i] > 0:
                length = max(0, (full_ends[i] - full_starts[i] + full_steps[i] - 1) // full_steps[i])
            else:
                length = max(0, (full_ends[i] - full_starts[i] + full_steps[i] + 1) // full_steps[i])
            out_shape.append(length)
        out_shape = tuple(out_shape)
            
        if self.lib is not None and data.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(data.data), data.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            c_starts = (ctypes.c_int * ndim)(*full_starts)
            c_steps = (ctypes.c_int * ndim)(*full_steps)

            self.lib.slice_forward(input_c, output_c, c_starts, c_steps)

            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
        else:
            slices = []
            for start, end, step in zip(full_starts, full_ends, full_steps):
                py_end = None if step < 0 and end == -1 else end
                slices.append(slice(start, py_end, step))
            out_data = np.asarray(data.data)[tuple(slices)]
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # 执行 `Slice` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, data: Tensor_, starts: Tensor_, ends: Tensor_, axes: Tensor_ = None, steps: Tensor_ = None) -> dict:
        if (
            hasattr(starts, "data") and starts.data is not None
            and hasattr(ends, "data") and ends.data is not None
            and (axes is None or (hasattr(axes, "data") and axes.data is not None))
            and (steps is None or (hasattr(steps, "data") and steps.data is not None))
        ):
            _starts = starts.data.astype(np.int64).flatten().tolist()
            _ends = ends.data.astype(np.int64).flatten().tolist()
            _axes = axes.data.astype(np.int64).flatten().tolist() if axes is not None else list(range(len(_starts)))
            _steps = steps.data.astype(np.int64).flatten().tolist() if steps is not None else [1] * len(_starts)
            ndim = len(data.size)
            full_starts = [0] * ndim
            full_ends = list(data.size)
            full_steps = [1] * ndim
            for i, axis in enumerate(_axes):
                if axis < 0:
                    axis += ndim
                s, e, st = _starts[i], _ends[i], _steps[i]
                dim_len = data.size[axis]
                if s < 0:
                    s += dim_len
                if e < 0:
                    e += dim_len
                if st > 0:
                    s = max(0, min(s, dim_len))
                    e = max(0, min(e, dim_len))
                else:
                    s = max(0, min(s, dim_len - 1))
                    e = max(-1, min(e, dim_len - 1))
                full_starts[axis] = s
                full_ends[axis] = e
                full_steps[axis] = st
            out_shape = []
            for i in range(ndim):
                if full_steps[i] > 0:
                    length = max(0, (full_ends[i] - full_starts[i] + full_steps[i] - 1) // full_steps[i])
                else:
                    length = max(0, (full_ends[i] - full_starts[i] + full_steps[i] + 1) // full_steps[i])
                out_shape.append(length)
            output_tensor = Tensor_(*tuple(out_shape), dtype=self.dtype)
        else:
            output_tensor = Tensor_(*data.size, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values


class Cast(Ops):
    # 初始化 `Cast` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Cast, self).__init__(inputs, outputs)
        self.dtype = dtype # 这里的 dtype 就是目标类型
        self.version = version
        if self.lib:
            self.lib.cast_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

    # 执行 `Cast` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        if self.lib is not None and input.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            output_shape_c = (ctypes.c_int * len(input.size))(*input.size)
            output_c = self.lib.create_tensor(output_shape_c, len(input.size), nn.DTYPE_MAP[self.dtype])
            self.lib.cast_forward(input_c, output_c)
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*input.size, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}
        np_dtype = nn.DTYPE_TO_NUMPY.get(self.dtype)
        if np_dtype is None:
            raise ValueError(f"Cast target dtype {self.dtype!r} is not supported")
        out_data = np.asarray(input.data).astype(np_dtype)
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}
    # 执行 `Cast` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}


class CastLike(Ops):
    # 初始化 `CastLike` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype=None, version="17"):
        super(CastLike, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.cast_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

    # 执行 `CastLike` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor, target_type: Tensor) -> dict:
        out_dtype = self.dtype or target_type.dtype
        if self.lib is not None and input.dtype in nn.DTYPE_MAP and out_dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            output_shape_c = (ctypes.c_int * len(input.size))(*input.size)
            output_c = self.lib.create_tensor(output_shape_c, len(input.size), nn.DTYPE_MAP[out_dtype])
            self.lib.cast_forward(input_c, output_c)
            out_data = self._ctensor_to_numpy(output_c, out_dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*input.size, dtype=out_dtype, data=out_data), "parameters": None, "graph": None}
        np_dtype = nn.DTYPE_TO_NUMPY.get(out_dtype)
        if np_dtype is None:
            raise ValueError(f"CastLike target dtype {out_dtype!r} is not supported")
        out_data = input.data.astype(np_dtype)
        return {"tensor": Tensor(*input.size, dtype=out_dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `CastLike` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_, target_type: Tensor_) -> dict:
        out_dtype = self.dtype or target_type.dtype
        return {"tensor": Tensor_(*input.size, dtype=out_dtype), "parameters": None, "graph": None}


class Sum(Ops):
    # 初始化 `Sum` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super(Sum, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.sum_forward.argtypes = [
                ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.POINTER(CTensor)
            ]

    # 执行 `Sum` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, *inputs: Tensor) -> dict:
        if not inputs:
            raise ValueError("Sum requires at least one input")
        arrays = np.broadcast_arrays(*(x.data for x in inputs))
        if self.lib is not None and self.dtype in nn.DTYPE_MAP and all(x.dtype in nn.DTYPE_MAP for x in inputs):
            input_ctensors = [
                self._numpy_to_ctensor(np.ascontiguousarray(arr.astype(nn.DTYPE_TO_NUMPY[x.dtype], copy=False)), x.dtype)
                for x, arr in zip(inputs, arrays)
            ]
            input_array = (ctypes.POINTER(CTensor) * len(input_ctensors))(*input_ctensors)
            output_shape_c = (ctypes.c_int * len(arrays[0].shape))(*arrays[0].shape)
            output_c = self.lib.create_tensor(output_shape_c, len(arrays[0].shape), nn.DTYPE_MAP[self.dtype])
            self.lib.sum_forward(input_array, len(input_ctensors), output_c)
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            for c_tensor in input_ctensors:
                self.lib.free_tensor(c_tensor)
            self.lib.free_tensor(output_c)
        else:
            out_data = np.zeros(arrays[0].shape, dtype=np.result_type(*(arr.dtype for arr in arrays)))
            for arr in arrays:
                out_data = out_data + arr
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `Sum` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, *inputs: Tensor_) -> dict:
        if not inputs:
            raise ValueError("Sum requires at least one input")
        out_shape = np.broadcast_shapes(*(x.size for x in inputs))
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None, "graph": None}
