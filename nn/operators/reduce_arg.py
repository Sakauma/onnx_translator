# /**
#   ******************************************************************************
#   * @file        reduce_arg.py
#   * @author      Egor Izmaylov
#   * @brief       按算子职责分组保存 `reduce_arg` 相关 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *

class ReduceBase(Ops):
    # 初始化 `ReduceBase` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axes=None, keepdims=1, dtype="float32", version="17", noop_with_empty_axes=0):
        super().__init__(inputs, outputs)
        self.axes = axes # 初始 axes，可能为 None
        self.keepdims = keepdims
        self.dtype = dtype
        self.version = version
        self.noop_with_empty_axes = noop_with_empty_axes

        # 注册参数类型
        if self.lib:
            func_name = self._get_c_func_name()
            if hasattr(self.lib, func_name):
                getattr(self.lib, func_name).argtypes = [
                    ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CReduceParams)
                ]

    # 封装 `_get_c_func_name` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _get_c_func_name(self):
        raise NotImplementedError

    # 封装 `_prepare_axes` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _prepare_axes(self, input_shape, runtime_axes=None):
        ndim = len(input_shape)
        # 优先级: 运行时输入 > 属性 > 默认(全归约)
        target_axes = None
        
        if runtime_axes is not None:
            # 如果 axes 是作为 Tensor 输入传进来的
            target_axes = runtime_axes.data.astype(np.int64).flatten().tolist()
            if not target_axes and not self.noop_with_empty_axes:
                target_axes = list(range(ndim))
        elif self.axes is not None:
            target_axes = self.axes
        else:
            # 默认归约所有维度
            target_axes = list(range(ndim))
            
        # 归一化负索引
        normalized_axes = []
        for ax in target_axes:
            if ax < 0: ax += ndim
            normalized_axes.append(ax)
        
        # 去重并排序
        return sorted(list(set(normalized_axes)))

    # 封装 `_numpy_reduce` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _numpy_reduce(self, data, axes):
        arr = np.asarray(data.data)
        if not axes:
            out_data = arr.copy()
        else:
            axis = tuple(axes)
            keepdims = bool(self.keepdims)
            op_name = self.__class__.__name__
            if op_name == "ReduceMean":
                out_data = np.mean(arr, axis=axis, keepdims=keepdims)
            elif op_name == "ReduceSum":
                out_data = np.sum(arr, axis=axis, keepdims=keepdims)
            elif op_name == "ReduceMax":
                out_data = np.max(arr, axis=axis, keepdims=keepdims)
            elif op_name == "ReduceMin":
                out_data = np.min(arr, axis=axis, keepdims=keepdims)
            elif op_name == "ReduceProd":
                out_data = np.prod(arr, axis=axis, keepdims=keepdims)
            elif op_name == "ReduceL1":
                out_data = np.sum(np.abs(arr), axis=axis, keepdims=keepdims)
            elif op_name == "ReduceL2":
                out_data = np.sqrt(np.sum(np.square(arr), axis=axis, keepdims=keepdims))
            elif op_name == "ReduceLogSum":
                out_data = np.log(np.sum(arr, axis=axis, keepdims=keepdims))
            elif op_name == "ReduceLogSumExp":
                out_data = np.log(np.sum(np.exp(arr), axis=axis, keepdims=keepdims))
            elif op_name == "ReduceSumSquare":
                out_data = np.sum(np.square(arr), axis=axis, keepdims=keepdims)
            else:
                raise ValueError(f"Unsupported reduce op {op_name}")

        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return Tensor(*out_data.shape, dtype=self.dtype, data=out_data)

    # 封装 `_calc_out_shape` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _calc_out_shape(self, input_shape, axes):
        out_shape = []
        for i in range(len(input_shape)):
            if i in axes:
                if self.keepdims:
                    out_shape.append(1)
            else:
                out_shape.append(input_shape[i])
        
        if not out_shape and not self.keepdims:
            # 这种情况下结果是标量，shape 为 ()
            pass 
            
        return tuple(out_shape)

    # 执行 `ReduceBase` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, data, axes_tensor=None):
        real_axes = self._prepare_axes(data.size, axes_tensor)
        out_shape = self._calc_out_shape(data.size, real_axes)

        if self.lib is None or not real_axes:
            return {"tensor": self._numpy_reduce(data, real_axes), "parameters": None}
        
        axes_arr = (ctypes.c_int * len(real_axes))(*real_axes)
        c_params = CReduceParams()
        c_params.axes = ctypes.cast(axes_arr, ctypes.POINTER(ctypes.c_int))
        c_params.num_axes = len(real_axes)
        c_params.keepdims = self.keepdims
        
        input_c = self._numpy_to_ctensor(data.data, data.dtype)
        # 处理标量输出形状
        shape_len = len(out_shape) if out_shape else 0
        output_shape_c = (ctypes.c_int * shape_len)(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, shape_len, nn.DTYPE_MAP[self.dtype])
        
        getattr(self.lib, self._get_c_func_name())(input_c, output_c, ctypes.byref(c_params))
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c); self.lib.free_tensor(output_c)
        
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `ReduceBase` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, data, axes_tensor=None):
        runtime_axes = axes_tensor if (
            axes_tensor is not None
            and hasattr(axes_tensor, "data")
            and axes_tensor.data is not None
        ) else None
        real_axes = self._prepare_axes(data.size, runtime_axes)
        out_shape = self._calc_out_shape(data.size, real_axes)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}


class ReduceMean(ReduceBase):
    # 封装 `_get_c_func_name` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _get_c_func_name(self): return "reduce_mean_forward"


class ReduceSum(ReduceBase):
    # 封装 `_get_c_func_name` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _get_c_func_name(self): return "reduce_sum_forward"


class ReduceMax(ReduceBase):
    # 封装 `_get_c_func_name` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _get_c_func_name(self): return "reduce_max_forward"


class ReduceMin(ReduceBase):
    # 封装 `_get_c_func_name` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _get_c_func_name(self): return "reduce_min_forward"


class ReduceProd(ReduceBase):
    # 封装 `_get_c_func_name` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _get_c_func_name(self): return "reduce_prod_forward"


class ArgBase(Ops):
    # 初始化 `ArgBase` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=0, keepdims=1, select_last_index=0, dtype="int64", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.keepdims = keepdims
        self.select_last_index = select_last_index
        self.dtype = "int64" # ArgMax 输出必定是索引
        self.version = version
        if self.lib:
            func_name = self._get_c_func_name()
            if hasattr(self.lib, func_name):
                getattr(self.lib, func_name).argtypes = [
                    ctypes.POINTER(CTensor),
                    ctypes.POINTER(CTensor),
                    ctypes.c_int,
                    ctypes.c_int,
                ]

    # 封装 `_get_c_func_name` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _get_c_func_name(self): raise NotImplementedError

    # 封装 `_arg_numpy` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _arg_numpy(self, values, axis):
        if isinstance(self, ArgMax):
            if self.select_last_index:
                reversed_idx = np.argmax(np.flip(values, axis=axis), axis=axis)
                return values.shape[axis] - 1 - reversed_idx
            return np.argmax(values, axis=axis)
        if self.select_last_index:
            reversed_idx = np.argmin(np.flip(values, axis=axis), axis=axis)
            return values.shape[axis] - 1 - reversed_idx
        return np.argmin(values, axis=axis)

    # 执行 `ArgBase` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, data):
        ndim = len(data.size)
        axis = self.axis if self.axis >= 0 else self.axis + ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"axis {self.axis} is out of bounds for rank {ndim}")

        out_shape = list(data.size)
        if self.keepdims:
            out_shape[axis] = 1
        else:
            out_shape.pop(axis)

        if self.lib is not None and data.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(data.data), data.dtype)
            shape_len = len(out_shape)
            output_shape_c = (ctypes.c_int * shape_len)(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, shape_len, nn.DTYPE_MAP[self.dtype])
            getattr(self.lib, self._get_c_func_name())(
                input_c,
                output_c,
                ctypes.c_int(axis),
                ctypes.c_int(self.select_last_index),
            )
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        out_data = self._arg_numpy(np.asarray(data.data), axis).astype(np.int64)
        if self.keepdims:
            out_data = np.expand_dims(out_data, axis=axis)
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `ArgBase` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, data):
        ndim = len(data.size)
        axis = self.axis if self.axis >= 0 else self.axis + ndim
        out_shape = list(data.size)
        if self.keepdims: out_shape[axis] = 1
        else: out_shape.pop(axis)
        return {"tensor": Tensor_(*tuple(out_shape), dtype=self.dtype), "parameters": None}


class ArgMax(ArgBase):
    # 封装 `_get_c_func_name` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _get_c_func_name(self): return "argmax_forward"


class ArgMin(ArgBase):
    # 封装 `_get_c_func_name` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _get_c_func_name(self): return "argmin_forward"


class ReduceL1(ReduceBase):
    # 封装 `_get_c_func_name` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _get_c_func_name(self): return "reduce_l1_forward"


class ReduceL2(ReduceBase):
    # 封装 `_get_c_func_name` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _get_c_func_name(self): return "reduce_l2_forward"


class ReduceLogSum(ReduceBase):
    # 封装 `_get_c_func_name` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _get_c_func_name(self): return "reduce_log_sum_forward"


class ReduceLogSumExp(ReduceBase):
    # 封装 `_get_c_func_name` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _get_c_func_name(self): return "reduce_log_sum_exp_forward"


class ReduceSumSquare(ReduceBase):
    # 封装 `_get_c_func_name` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _get_c_func_name(self): return "reduce_sum_square_forward"
