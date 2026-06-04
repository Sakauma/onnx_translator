# /**
#   ******************************************************************************
#   * @file        index_scatter_ops.py
#   * @author      Egor Izmaylov
#   * @brief       保存 `index_scatter_ops` 分组中的 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *

class Gather(Ops):
    # 初始化 `Gather` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=0, dtype="float32", version="17"):
        super(Gather, self).__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.gather_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor), 
                ctypes.POINTER(nn.CTensor), ctypes.c_int
            ]

    # 执行 `Gather` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, data: Tensor, indices: Tensor) -> dict:
        # 计算输出形状: data.shape[:axis] + indices.shape + data.shape[axis+1:]
        axis = self.axis if self.axis >= 0 else self.axis + len(data.size)
        out_shape = data.size[:axis] + indices.size + data.size[axis+1:]

        if self.lib is not None and data.dtype in nn.DTYPE_MAP and indices.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            data_c = self._numpy_to_ctensor(np.ascontiguousarray(data.data), data.dtype)
            indices_c = self._numpy_to_ctensor(np.ascontiguousarray(indices.data), indices.dtype)

            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            self.lib.gather_forward(data_c, indices_c, output_c, ctypes.c_int(axis))

            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(data_c); self.lib.free_tensor(indices_c); self.lib.free_tensor(output_c)
        else:
            out_data = np.take(np.asarray(data.data), np.asarray(indices.data, dtype=np.int64), axis=axis)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `Gather` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, data: Tensor_, indices: Tensor_) -> dict:
        try:
            axis = self.axis if self.axis >= 0 else self.axis + len(data.size)
            d_size = list(data.size) if isinstance(data.size, tuple) else data.size
            i_size = list(indices.size) if isinstance(indices.size, tuple) else indices.size
            # [Fix] 增加安全切片
            if axis >= len(d_size): axis = len(d_size) - 1
            out_shape = tuple(d_size[:axis] + i_size + d_size[axis+1:])
        except:
            out_shape = data.size # 兜底

        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None, "graph": None}


class ScatterND(Ops):
    # 初始化 `ScatterND` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, reduction="none", dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.reduction = {"none": 0, "add": 1, "mul": 2}.get(reduction, 0)
        self.dtype = dtype
        self.version = version

    # 执行 `ScatterND` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, data, indices, updates):
        out_tensor = Tensor(*data.size, dtype=self.dtype, data=data.data.copy())
        
        d_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        i_c = self._numpy_to_ctensor(indices.data, indices.dtype)
        u_c = self._numpy_to_ctensor(updates.data, updates.dtype)
        
        self.lib.scatter_nd_forward(d_c, i_c, u_c, ctypes.c_int(self.reduction))
        
        out_data = self._ctensor_to_numpy(d_c, self.dtype)
        out_tensor.data = out_data
        
        self.lib.free_tensor(d_c); self.lib.free_tensor(i_c); self.lib.free_tensor(u_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # 执行 `ScatterND` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, data, indices, updates):
        return {"tensor": Tensor_(*data.size, dtype=self.dtype), "parameters": None}


class GatherND(Ops):
    # 初始化 `GatherND` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, batch_dims=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.batch_dims = batch_dims
        self.dtype = dtype
        self.version = version

    # 执行 `GatherND` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, data, indices):
        # 计算形状
        # Output shape = indices.shape[:-1] + data.shape[indices.shape[-1] + batch_dims:]
        idx_shape = list(indices.size)
        data_shape = list(data.size)
        k = idx_shape[-1]
        out_shape = idx_shape[:-1] + data_shape[k + self.batch_dims:]
        out_shape = tuple(out_shape)

        if self.lib is not None and data.dtype in nn.DTYPE_MAP and indices.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            data_c = self._numpy_to_ctensor(np.ascontiguousarray(data.data), data.dtype)
            idx_c = self._numpy_to_ctensor(np.ascontiguousarray(indices.data), indices.dtype)

            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            self.lib.gather_nd_forward(data_c, idx_c, output_c, ctypes.c_int(self.batch_dims))

            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(data_c); self.lib.free_tensor(idx_c); self.lib.free_tensor(output_c)
        else:
            data_arr = np.asarray(data.data)
            indices_arr = np.asarray(indices.data, dtype=np.int64)
            out_data = np.empty(out_shape, dtype=data_arr.dtype)
            for prefix in np.ndindex(*idx_shape[:-1]):
                batch_prefix = prefix[:self.batch_dims]
                gather_index = tuple(int(item) for item in indices_arr[prefix])
                out_data[prefix] = data_arr[batch_prefix + gather_index]
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `GatherND` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, data, indices):
        idx_shape = list(indices.size)
        data_shape = list(data.size)
        k = idx_shape[-1]
        out_shape = idx_shape[:-1] + data_shape[k + self.batch_dims:]
        return {"tensor": Tensor_(*tuple(out_shape), dtype=self.dtype), "parameters": None}


class GatherElements(Ops):
    # 初始化 `GatherElements` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        self.version = version

    # 执行 `GatherElements` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, data, indices):
        # GatherElements 输出形状与 Indices 相同
        out_shape = indices.size

        axis = self.axis if self.axis >= 0 else self.axis + len(data.size)
        if self.lib is not None and data.dtype in nn.DTYPE_MAP and indices.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            data_c = self._numpy_to_ctensor(np.ascontiguousarray(data.data), data.dtype)
            idx_c = self._numpy_to_ctensor(np.ascontiguousarray(indices.data), indices.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            self.lib.gather_elements_forward(data_c, idx_c, output_c, ctypes.c_int(axis))

            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(data_c); self.lib.free_tensor(idx_c); self.lib.free_tensor(output_c)
        else:
            out_data = np.take_along_axis(np.asarray(data.data), np.asarray(indices.data, dtype=np.int64), axis=axis)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `GatherElements` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, data, indices):
        return {"tensor": Tensor_(*indices.size, dtype=self.dtype), "parameters": None}


class NonZero(Ops):
    # 初始化 `NonZero` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="int64", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = "int64" # NonZero 必须返回 int64
        self.version = version

    # 执行 `NonZero` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input):
        count = np.count_nonzero(input.data)
        ndim = len(input.size)
        out_shape = (ndim, count)
        
        output_tensor = Tensor(*out_shape, dtype=self.dtype)

        in_c = self._numpy_to_ctensor(input.data, input.dtype)
        out_c = self._numpy_to_ctensor(output_tensor.data, self.dtype)
        
        self.lib.nonzero_forward(in_c, out_c)
        
        out_data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(in_c); self.lib.free_tensor(out_c)
        
        output_tensor.data = out_data
        return {"tensor": output_tensor, "parameters": None}

    # 执行 `NonZero` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input):
        count = int(np.count_nonzero(input.data)) if hasattr(input, "data") and input.data is not None else 1
        return {"tensor": Tensor_(len(input.size), count, dtype=self.dtype), "parameters": None}


class TopK(Ops):
    # 初始化 `TopK` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=-1, largest=1, sorted=1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.largest = largest
        self.sorted = sorted
        self.dtype = dtype # Values 的类型
        self.version = version
        
        if self.lib:
            self.lib.topk_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]

    # 执行 `TopK` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, k_tensor):
        K = int(k_tensor.data.item())
        axis = self.axis if self.axis >= 0 else self.axis + len(x.size)
        if axis < 0 or axis >= len(x.size):
            raise ValueError(f"TopK axis {self.axis} is out of bounds for rank {len(x.size)}")
        if K < 0 or K > x.size[axis]:
            raise ValueError(f"TopK K={K} must be in [0, {x.size[axis]}]")
        
        out_shape = list(x.size)
        out_shape[axis] = K
        out_shape = tuple(out_shape)

        if self.lib is None or x.dtype not in nn.DTYPE_MAP or self.dtype not in nn.DTYPE_MAP:
            data = _tensor_data_as_numeric(x)
            order_data = -data if self.largest else data
            order = np.argsort(order_data, axis=axis, kind="stable")
            top_indices = np.take(order, np.arange(K), axis=axis).astype(np.int64, copy=False)
            top_values = np.take_along_axis(data, top_indices, axis=axis)
            if self.sorted:
                values_data = top_values
                indices_data = top_indices
            else:
                values_data = top_values
                indices_data = top_indices
            values_data = _cast_numeric_to_dtype(values_data, self.dtype)
            return {
                "tensor": [
                    Tensor(*out_shape, dtype=self.dtype, data=values_data),
                    Tensor(*out_shape, dtype="int64", data=indices_data),
                ],
                "parameters": None,
            }
        
        values_tensor = Tensor(*out_shape, dtype=self.dtype)
        indices_tensor = Tensor(*out_shape, dtype="int64")
        
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        v_c = self._numpy_to_ctensor(values_tensor.data, self.dtype)
        i_c = self._numpy_to_ctensor(indices_tensor.data, "int64")
        
        self.lib.topk_forward(x_c, v_c, i_c, ctypes.c_int(self.axis), ctypes.c_int(self.largest), ctypes.c_int(self.sorted), ctypes.c_int(K))
        
        v_data = self._ctensor_to_numpy(v_c, self.dtype)
        i_data = self._ctensor_to_numpy(i_c, "int64")
        
        values_tensor.data = v_data
        indices_tensor.data = i_data
        
        self.lib.free_tensor(x_c); self.lib.free_tensor(v_c); self.lib.free_tensor(i_c)
        
        # 返回列表
        return {"tensor": [values_tensor, indices_tensor], "parameters": None}

    # 执行 `TopK` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, k_tensor):
        axis = self.axis if self.axis >= 0 else self.axis + len(x.size)
        out_shape = list(x.size)
        if k_tensor is not None and hasattr(k_tensor, "data") and k_tensor.data is not None:
            out_shape[axis] = int(k_tensor.data.item())
        else:
            out_shape[axis] = 1
        out_shape = tuple(out_shape)
        return {"tensor": [Tensor_(*out_shape, dtype=self.dtype), Tensor_(*out_shape, dtype="int64")], "parameters": None}


class CumSum(Ops):
    # 初始化 `CumSum` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, exclusive=0, reverse=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.exclusive = exclusive
        self.reverse = reverse
        self.dtype = dtype
        self.version = version

    # 执行 `CumSum` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, axis_tensor):
        axis = int(axis_tensor.data.item())
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        
        x_c = self._numpy_to_ctensor(x.data, self.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        
        self.lib.cumsum_forward(x_c, out_c, ctypes.c_int(axis), ctypes.c_int(self.exclusive), ctypes.c_int(self.reverse))
        
        out_data = self._ctensor_to_numpy(out_c, self.dtype)
        out_tensor.data = out_data
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        
        return {"tensor": out_tensor, "parameters": None}

    # 执行 `CumSum` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, axis_tensor):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class CumProd(Ops):
    # 初始化 `CumProd` 的构造参数，保存 exclusive、reverse、dtype 和版本信息。
    def __init__(self, inputs, outputs, exclusive=0, reverse=0, dtype="float32", version="26"):
        super().__init__(inputs, outputs)
        self.exclusive = exclusive
        self.reverse = reverse
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.cumprod_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]

    # 执行 `CumProd` 的真实张量计算路径，按指定轴计算累计乘积并写回原 dtype。
    def forward(self, x, axis_tensor):
        axis = int(axis_tensor.data.item())
        out_tensor = Tensor(*x.size, dtype=self.dtype)

        x_c = self._numpy_to_ctensor(x.data, self.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)

        self.lib.cumprod_forward(x_c, out_c, ctypes.c_int(axis), ctypes.c_int(self.exclusive), ctypes.c_int(self.reverse))

        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)

        return {"tensor": out_tensor, "parameters": None}

    # 执行 `CumProd` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, axis_tensor):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class OneHot(Ops):
    # 初始化 `OneHot` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=-1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype # 由 values 决定，或者外部指定
        self.version = version
        
        if self.lib:
            self.lib.one_hot_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int
            ]

    # 执行 `OneHot` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, indices, depth_tensor, values):
        depth = int(depth_tensor.data.item())
        if depth < 0:
            raise ValueError(f"OneHot depth must be non-negative, got {depth}")
        
        out_shape = list(indices.size)
        axis = self.axis if self.axis >= 0 else self.axis + len(out_shape) + 1
        if axis < 0 or axis > len(out_shape):
            raise ValueError(f"OneHot axis {self.axis} is out of bounds for output rank {len(out_shape) + 1}")
        out_shape.insert(axis, depth)
        out_shape = tuple(out_shape)
        
        out_dtype = values.dtype
        values_arr = np.asarray(values.data)
        if values_arr.size != 2:
            raise ValueError("OneHot values input must contain exactly two elements")
        if (
            self.lib is not None
            and out_dtype in nn.DTYPE_MAP
            and values.dtype in nn.DTYPE_MAP
            and indices.dtype in nn.DTYPE_MAP
        ):
            indices_c = self._numpy_to_ctensor(np.ascontiguousarray(indices.data), indices.dtype)
            values_c = self._numpy_to_ctensor(np.ascontiguousarray(values_arr), values.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[out_dtype])
            self.lib.one_hot_forward(indices_c, values_c, output_c, ctypes.c_int(axis))
            out_data = self._ctensor_to_numpy(output_c, out_dtype)
            self.lib.free_tensor(indices_c)
            self.lib.free_tensor(values_c)
            self.lib.free_tensor(output_c)
        else:
            off_value, on_value = values_arr.reshape(-1)[:2]
            np_dtype = values_arr.dtype if out_dtype == "string" else nn.DTYPE_TO_NUMPY.get(out_dtype, values_arr.dtype)
            out_data = np.full(out_shape, off_value, dtype=np_dtype)
            indices_arr = np.asarray(indices.data, dtype=np.int64)
            for idx in np.ndindex(indices_arr.shape):
                if depth == 0:
                    continue
                class_index = int(indices_arr[idx])
                if class_index < 0:
                    class_index += depth
                if 0 <= class_index < depth:
                    out_idx = list(idx)
                    out_idx.insert(axis, class_index)
                    out_data[tuple(out_idx)] = on_value
        
        return {"tensor": Tensor(*out_shape, dtype=out_dtype, data=out_data), "parameters": None}

    # 执行 `OneHot` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, indices, depth_tensor, values):
        out_shape = list(indices.size)
        axis = self.axis if self.axis >= 0 else self.axis + len(out_shape) + 1
        if axis < 0 or axis > len(out_shape):
            raise ValueError(f"OneHot axis {self.axis} is out of bounds for output rank {len(out_shape) + 1}")
        depth = 1
        if depth_tensor is not None and hasattr(depth_tensor, "data") and depth_tensor.data is not None:
            depth = int(depth_tensor.data.item())
            if depth < 0:
                raise ValueError(f"OneHot depth must be non-negative, got {depth}")
        out_shape.insert(axis, depth)
        out_dtype = getattr(values, "dtype", self.dtype)
        return {"tensor": Tensor_(*tuple(out_shape), dtype=out_dtype), "parameters": None}
