# /**
#   ******************************************************************************
#   * @file        normalization_ops.py
#   * @author      Egor Izmaylov
#   * @brief       保存 `normalization_ops` 分组中的 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *

class Softmax(Ops):
    # 初始化 `Softmax` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis, dtype, version="17"):
        super(Softmax, self).__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        
        if self.lib:
            self.lib.softmax_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int
            ]

    # 执行 `Softmax` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        out_shape = input.size
        
        input_c = self._numpy_to_ctensor(input.data, input.dtype)
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), DTYPE_MAP[self.dtype])
        
        self.lib.softmax_forward(input_c, output_c, ctypes.c_int(self.axis))
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)
        
        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # 执行 `Softmax` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values


class LRN(Ops):
    # 初始化 `LRN` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, size, alpha=0.0001, beta=0.75, bias=1.0, dtype="float32", version="17"):
        super(LRN, self).__init__(inputs, outputs)
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.bias = bias
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.lrn_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float
            ]

    # 执行 `LRN` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x: Tensor) -> dict:
        if len(x.size) < 3:
            raise ValueError(f"LRN expects input rank >= 3, got {x.size}")
        if self.size <= 0:
            raise ValueError("LRN size must be positive")
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(x.size))(*x.size)
            out_c = self.lib.create_tensor(output_shape_c, len(x.size), nn.DTYPE_MAP[self.dtype])
            self.lib.lrn_forward(
                x_c, out_c, ctypes.c_int(self.size),
                ctypes.c_float(self.alpha), ctypes.c_float(self.beta), ctypes.c_float(self.bias)
            )
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*x.size, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}
        data = _tensor_data_as_numeric(x).astype(np.float32, copy=False)
        square_sum = np.zeros_like(data, dtype=np.float32)
        channels = data.shape[1]
        lower = (self.size - 1) // 2
        upper = self.size - 1 - lower
        for c in range(channels):
            begin = max(0, c - lower)
            end = min(channels, c + upper + 1)
            square_sum[:, c, ...] = np.sum(data[:, begin:end, ...] ** 2, axis=1)
        out_data = data / ((self.bias + (self.alpha / self.size) * square_sum) ** self.beta)
        out_data = _cast_numeric_to_dtype(out_data, self.dtype)
        return {"tensor": Tensor(*x.size, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `LRN` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x: Tensor_) -> dict:
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None, "graph": None}


class MeanVarianceNormalization(Ops):
    # 初始化 `MeanVarianceNormalization` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axes=None, dtype="float32", version="17"):
        super(MeanVarianceNormalization, self).__init__(inputs, outputs)
        self.axes = list(axes) if axes is not None else [0, 2, 3]
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.mean_variance_normalization_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CReduceParams)
            ]

    # 封装 `_axes_for_rank` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _axes_for_rank(self, rank):
        axes = []
        for ax in self.axes:
            axis = ax + rank if ax < 0 else ax
            if axis < 0 or axis >= rank:
                raise ValueError(f"MeanVarianceNormalization axis {ax} is out of bounds for rank {rank}")
            axes.append(axis)
        return tuple(sorted(set(axes)))

    # 执行 `MeanVarianceNormalization` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x: Tensor) -> dict:
        axes = self._axes_for_rank(len(x.size))
        if self.lib is not None and axes and x.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            axes_arr = (ctypes.c_int * len(axes))(*axes)
            c_params = CReduceParams()
            c_params.axes = ctypes.cast(axes_arr, ctypes.POINTER(ctypes.c_int))
            c_params.num_axes = len(axes)
            c_params.keepdims = 1
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(x.size))(*x.size)
            out_c = self.lib.create_tensor(output_shape_c, len(x.size), nn.DTYPE_MAP[self.dtype])
            self.lib.mean_variance_normalization_forward(x_c, out_c, ctypes.byref(c_params))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*x.size, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}
        data = _tensor_data_as_numeric(x).astype(np.float32, copy=False)
        mean = np.mean(data, axis=axes, keepdims=True)
        variance = np.mean((data - mean) ** 2, axis=axes, keepdims=True)
        out_data = (data - mean) / np.sqrt(variance)
        out_data = _cast_numeric_to_dtype(out_data, self.dtype)
        return {"tensor": Tensor(*x.size, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `MeanVarianceNormalization` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x: Tensor_) -> dict:
        self._axes_for_rank(len(x.size))
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None, "graph": None}


class RMSNormalization(Ops):
    # 初始化 `RMSNormalization` 的构造参数，保存 axis、epsilon、stash_type 和输出 dtype。
    def __init__(self, inputs, outputs, axis=-1, epsilon=1e-5, stash_type=1, dtype="float32", version="23"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.epsilon = epsilon
        self.stash_type = stash_type
        self.stash_dtype = nn.onnx_dtype_mapping.get(stash_type, "float32")
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.rms_normalization_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.c_int, ctypes.c_float, ctypes.c_int
            ]

    # 规范化 axis 并校验 scale 是否可按 ONNX 单向广播规则广播到输入张量。
    def _resolve_axis_and_scale(self, x_shape, scale_shape):
        rank = len(x_shape)
        axis = self.axis if self.axis >= 0 else self.axis + rank
        if axis < 0 or axis >= rank:
            raise ValueError(f"RMSNormalization axis {self.axis} is out of bounds for rank {rank}")
        if len(scale_shape) > rank:
            raise ValueError(f"RMSNormalization scale rank {len(scale_shape)} exceeds input rank {rank}")
        try:
            np.broadcast_shapes(tuple(x_shape), tuple(scale_shape))
        except ValueError as exc:
            raise ValueError(f"RMSNormalization scale shape {scale_shape} is not broadcastable to {x_shape}") from exc
        return axis

    # 执行 `RMSNormalization` 的真实张量计算路径，按后缀维度计算 RMS 并应用 scale。
    def forward(self, x, scale):
        x_data = _tensor_data_as_numeric(x)
        scale_data = _tensor_data_as_numeric(scale)
        axis = self._resolve_axis_and_scale(x_data.shape, scale_data.shape)
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and scale.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            scale_c = self._numpy_to_ctensor(np.ascontiguousarray(scale.data), scale.dtype)
            output_shape_c = (ctypes.c_int * len(x.size))(*x.size)
            out_c = self.lib.create_tensor(output_shape_c, len(x.size), nn.DTYPE_MAP[self.dtype])
            self.lib.rms_normalization_forward(
                x_c, scale_c, out_c, ctypes.c_int(axis), ctypes.c_float(self.epsilon), ctypes.c_int(self.stash_type)
            )
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(scale_c)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*x.size, dtype=self.dtype, data=out_data), "parameters": None}

        work_dtype = np.float64 if self.stash_dtype == "float64" else np.float32
        work = x_data.astype(work_dtype, copy=False)
        rms_axes = tuple(range(axis, work.ndim))
        mean_square = np.mean(work * work, axis=rms_axes, keepdims=True)
        normalized = work / np.sqrt(mean_square + self.epsilon)
        out_data = normalized * np.asarray(scale_data, dtype=work_dtype)
        out_data = _cast_numeric_to_dtype(out_data, self.dtype)
        return {"tensor": Tensor(*x.size, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `RMSNormalization` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, scale):
        self._resolve_axis_and_scale(x.size, scale.size)
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Mean(Ops):
    # 初始化 `Mean` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.mean_forward.argtypes = [
                ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.POINTER(CTensor)
            ]

    # 执行 `Mean` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, *inputs):
        if not inputs:
            raise ValueError("Mean requires at least one input")
        arrays = np.broadcast_arrays(*(x.data for x in inputs))
        if self.lib is not None and self.dtype in nn.DTYPE_MAP and all(x.dtype in nn.DTYPE_MAP for x in inputs):
            input_ctensors = [
                self._numpy_to_ctensor(np.ascontiguousarray(arr.astype(nn.DTYPE_TO_NUMPY[x.dtype], copy=False)), x.dtype)
                for x, arr in zip(inputs, arrays)
            ]
            input_array = (ctypes.POINTER(CTensor) * len(input_ctensors))(*input_ctensors)
            output_shape_c = (ctypes.c_int * len(arrays[0].shape))(*arrays[0].shape)
            output_c = self.lib.create_tensor(output_shape_c, len(arrays[0].shape), nn.DTYPE_MAP[self.dtype])
            self.lib.mean_forward(input_array, len(input_ctensors), output_c)
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            for c_tensor in input_ctensors:
                self.lib.free_tensor(c_tensor)
            self.lib.free_tensor(output_c)
        else:
            decoded_arrays = np.broadcast_arrays(*(_tensor_data_as_numeric(x) for x in inputs))
            out_data = np.mean(np.stack(decoded_arrays, axis=0), axis=0)
            out_data = _cast_numeric_to_dtype(out_data, self.dtype)
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `Mean` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, *inputs):
        if not inputs:
            raise ValueError("Mean requires at least one input")
        out_shape = np.broadcast_shapes(*(x.size for x in inputs))
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}


class IsInf(Ops):
    # 初始化 `IsInf` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, detect_negative=1, detect_positive=1, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.detect_neg = detect_negative
        self.detect_pos = detect_positive
        self.dtype = "bool"
        self.version = version
        
        if self.lib:
            self.lib.isinf_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int
            ]

    # 执行 `IsInf` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.isinf_forward(x_c, out_c, ctypes.c_int(self.detect_pos), ctypes.c_int(self.detect_neg))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}

    # 执行 `IsInf` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Round(Ops):
    # 初始化 `Round` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Round` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x): return {"tensor": self._execute_unary(x, "round_forward"), "parameters": None}
    # 执行 `Round` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Erf(Ops):
    # 初始化 `Erf` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Erf` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x): return {"tensor": self._execute_unary(x, "erf_forward"), "parameters": None}
    # 执行 `Erf` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class BatchNormalization(Ops):
    # 初始化 `BatchNormalization` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, epsilon=1e-5, momentum=0.9, training_mode=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.epsilon = epsilon
        self.momentum = momentum
        self.training_mode = training_mode
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.batch_norm_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.c_float
            ]
            try:
                self.lib.batch_norm_training_forward.argtypes = [
                    ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                    ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                    ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                    ctypes.c_float, ctypes.c_float
                ]
                self._has_batch_norm_training_c_backend = True
            except AttributeError:
                self._has_batch_norm_training_c_backend = False

    # 封装 `_reshape_param` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _reshape_param(param, rank):
        return _tensor_data_as_numeric(param).reshape((-1,) + (1,) * (rank - 2))

    # 封装 `_normalize` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _normalize(self, x_data, scale_data, bias_data, mean_data, var_data):
        y = scale_data * (x_data - mean_data) / np.sqrt(var_data + self.epsilon) + bias_data
        return _cast_numeric_to_dtype(y, self.dtype)

    # 执行 `BatchNormalization` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, scale, B, mean, var):
        x_data = _tensor_data_as_numeric(x)
        rank = x_data.ndim
        scale_data = self._reshape_param(scale, rank)
        bias_data = self._reshape_param(B, rank)

        if self.training_mode:
            if (
                self.lib is not None
                and getattr(self, "_has_batch_norm_training_c_backend", False)
                and self.dtype in nn.DTYPE_MAP
                and all(t.dtype in nn.DTYPE_MAP for t in (x, scale, B, mean, var))
            ):
                x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
                s_c = self._numpy_to_ctensor(np.ascontiguousarray(scale.data), scale.dtype)
                b_c = self._numpy_to_ctensor(np.ascontiguousarray(B.data), B.dtype)
                m_c = self._numpy_to_ctensor(np.ascontiguousarray(mean.data), mean.dtype)
                v_c = self._numpy_to_ctensor(np.ascontiguousarray(var.data), var.dtype)
                output_shape_c = (ctypes.c_int * len(x.size))(*x.size)
                channel_shape_c = (ctypes.c_int * 1)(x.size[1])
                out_c = self.lib.create_tensor(output_shape_c, len(x.size), nn.DTYPE_MAP[self.dtype])
                running_mean_c = self.lib.create_tensor(channel_shape_c, 1, nn.DTYPE_MAP[self.dtype])
                running_var_c = self.lib.create_tensor(channel_shape_c, 1, nn.DTYPE_MAP[self.dtype])
                self.lib.batch_norm_training_forward(
                    x_c, s_c, b_c, m_c, v_c,
                    out_c, running_mean_c, running_var_c,
                    ctypes.c_float(self.epsilon), ctypes.c_float(self.momentum)
                )
                y_data = self._ctensor_to_numpy(out_c, self.dtype)
                running_mean = self._ctensor_to_numpy(running_mean_c, self.dtype)
                running_var = self._ctensor_to_numpy(running_var_c, self.dtype)
                for tensor_c in (x_c, s_c, b_c, m_c, v_c, out_c, running_mean_c, running_var_c):
                    self.lib.free_tensor(tensor_c)
                outputs = (
                    Tensor(*x.size, dtype=self.dtype, data=y_data),
                    Tensor(x.size[1], dtype=self.dtype, data=running_mean),
                    Tensor(x.size[1], dtype=self.dtype, data=running_var),
                )
            else:
                axes = tuple(axis for axis in range(rank) if axis != 1)
                saved_mean = np.mean(x_data, axis=axes)
                saved_var = np.var(x_data, axis=axes)
                running_mean = _tensor_data_as_numeric(mean) * self.momentum + saved_mean * (1.0 - self.momentum)
                running_var = _tensor_data_as_numeric(var) * self.momentum + saved_var * (1.0 - self.momentum)
                y_data = self._normalize(
                    x_data,
                    scale_data,
                    bias_data,
                    saved_mean.reshape((-1,) + (1,) * (rank - 2)),
                    saved_var.reshape((-1,) + (1,) * (rank - 2)),
                )
                outputs = (
                    Tensor(*x.size, dtype=self.dtype, data=y_data),
                    Tensor(*saved_mean.shape, dtype=self.dtype, data=_cast_numeric_to_dtype(running_mean, self.dtype)),
                    Tensor(*saved_var.shape, dtype=self.dtype, data=_cast_numeric_to_dtype(running_var, self.dtype)),
                )
        else:
            if (
                self.lib is not None
                and self.dtype in nn.DTYPE_MAP
                and all(t.dtype in nn.DTYPE_MAP for t in (x, scale, B, mean, var))
            ):
                x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
                s_c = self._numpy_to_ctensor(np.ascontiguousarray(scale.data), scale.dtype)
                b_c = self._numpy_to_ctensor(np.ascontiguousarray(B.data), B.dtype)
                m_c = self._numpy_to_ctensor(np.ascontiguousarray(mean.data), mean.dtype)
                v_c = self._numpy_to_ctensor(np.ascontiguousarray(var.data), var.dtype)
                output_shape_c = (ctypes.c_int * len(x.size))(*x.size)
                out_c = self.lib.create_tensor(output_shape_c, len(x.size), nn.DTYPE_MAP[self.dtype])
                self.lib.batch_norm_forward(x_c, s_c, b_c, m_c, v_c, out_c, ctypes.c_float(self.epsilon))
                y_data = self._ctensor_to_numpy(out_c, self.dtype)
                self.lib.free_tensor(x_c)
                self.lib.free_tensor(s_c)
                self.lib.free_tensor(b_c)
                self.lib.free_tensor(m_c)
                self.lib.free_tensor(v_c)
                self.lib.free_tensor(out_c)
            else:
                y_data = self._normalize(
                    x_data,
                    scale_data,
                    bias_data,
                    self._reshape_param(mean, rank),
                    self._reshape_param(var, rank),
                )
            outputs = (Tensor(*x.size, dtype=self.dtype, data=y_data),)

        selected = tuple(value for name, value in zip(self.outputs, outputs) if name)
        return {"tensor": selected[0] if len(selected) == 1 else selected, "parameters": None}

    # 执行 `BatchNormalization` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, scale, B, mean, var):
        outputs = [Tensor_(*x.size, dtype=self.dtype)]
        if self.training_mode:
            outputs.extend([Tensor_(x.size[1], dtype=self.dtype), Tensor_(x.size[1], dtype=self.dtype)])
        selected = tuple(value for name, value in zip(self.outputs, outputs) if name)
        return {"tensor": selected[0] if len(selected) == 1 else selected, "parameters": None}


class InstanceNormalization(Ops):
    # 初始化 `InstanceNormalization` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, epsilon=1e-5, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.epsilon = epsilon
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.instance_norm_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.c_float
            ]

    # 执行 `InstanceNormalization` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, scale, B):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        s_c = self._numpy_to_ctensor(scale.data, scale.dtype)
        b_c = self._numpy_to_ctensor(B.data, B.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        
        self.lib.instance_norm_forward(x_c, s_c, b_c, out_c, ctypes.c_float(self.epsilon))
        
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        
        self.lib.free_tensor(x_c); self.lib.free_tensor(s_c); self.lib.free_tensor(b_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}

    # 执行 `InstanceNormalization` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, scale, B): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class LayerNormalization(Ops):
    # 初始化 `LayerNormalization` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=-1, epsilon=1e-5, stash_type=1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.epsilon = epsilon
        self.stash_type = stash_type
        self.stash_dtype = nn.onnx_dtype_mapping.get(stash_type, "float32")
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.layer_norm_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_float
            ]
            try:
                self.lib.layer_norm_multi_output_forward.argtypes = [
                    ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                    ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                    ctypes.c_int, ctypes.c_float
                ]
                self._has_layer_norm_stats_c_backend = True
            except AttributeError:
                self._has_layer_norm_stats_c_backend = False

    # 执行 `LayerNormalization` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, scale=None, B=None):
        x_data = _tensor_data_as_numeric(x)
        rank = x_data.ndim
        axis = self.axis if self.axis >= 0 else self.axis + rank
        if axis < 0 or axis >= rank:
            raise ValueError(f"LayerNormalization axis {self.axis} is out of bounds for rank {rank}")
        row_number = int(np.prod(x_data.shape[:axis], dtype=np.int64)) if axis > 0 else 1
        col_number = int(np.prod(x_data.shape[axis:], dtype=np.int64))
        wants_aux_outputs = len([name for name in self.outputs if name]) > 1
        can_use_c_backend = (
            self.lib is not None
            and self.dtype in nn.DTYPE_MAP
            and self.stash_dtype in nn.DTYPE_MAP
            and x.dtype in nn.DTYPE_MAP
            and (scale is None or scale.dtype in nn.DTYPE_MAP)
            and (B is None or B.dtype in nn.DTYPE_MAP)
            and (scale is None or int(np.prod(scale.size, dtype=np.int64)) == col_number)
            and (B is None or int(np.prod(B.size, dtype=np.int64)) == col_number)
        )
        if (
            can_use_c_backend
            and not wants_aux_outputs
        ):
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            scale_c = (
                self._numpy_to_ctensor(np.ascontiguousarray(scale.data), scale.dtype)
                if scale is not None else ctypes.POINTER(CTensor)()
            )
            b_c = (
                self._numpy_to_ctensor(np.ascontiguousarray(B.data), B.dtype)
                if B is not None else ctypes.POINTER(CTensor)()
            )
            output_shape_c = (ctypes.c_int * len(x.size))(*x.size)
            out_c = self.lib.create_tensor(output_shape_c, len(x.size), nn.DTYPE_MAP[self.dtype])
            self.lib.layer_norm_forward(x_c, scale_c, b_c, out_c, ctypes.c_int(axis), ctypes.c_float(self.epsilon))
            y_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            if scale is not None:
                self.lib.free_tensor(scale_c)
            if B is not None:
                self.lib.free_tensor(b_c)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*x.size, dtype=self.dtype, data=y_data), "parameters": None}
        if can_use_c_backend and wants_aux_outputs and getattr(self, "_has_layer_norm_stats_c_backend", False):
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            scale_c = (
                self._numpy_to_ctensor(np.ascontiguousarray(scale.data), scale.dtype)
                if scale is not None else ctypes.POINTER(CTensor)()
            )
            b_c = (
                self._numpy_to_ctensor(np.ascontiguousarray(B.data), B.dtype)
                if B is not None else ctypes.POINTER(CTensor)()
            )
            output_shape_c = (ctypes.c_int * len(x.size))(*x.size)
            reduction_shape = tuple(x_data.shape[:axis]) + (1,) * (rank - axis)
            reduction_shape_c = (ctypes.c_int * len(reduction_shape))(*reduction_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(x.size), nn.DTYPE_MAP[self.dtype])
            mean_c = self.lib.create_tensor(reduction_shape_c, len(reduction_shape), nn.DTYPE_MAP[self.stash_dtype])
            inv_std_c = self.lib.create_tensor(reduction_shape_c, len(reduction_shape), nn.DTYPE_MAP[self.stash_dtype])
            self.lib.layer_norm_multi_output_forward(
                x_c, scale_c, b_c, out_c, mean_c, inv_std_c, ctypes.c_int(axis), ctypes.c_float(self.epsilon)
            )
            y_data = self._ctensor_to_numpy(out_c, self.dtype)
            mean_data = self._ctensor_to_numpy(mean_c, self.stash_dtype)
            inv_std_data = self._ctensor_to_numpy(inv_std_c, self.stash_dtype)
            self.lib.free_tensor(x_c)
            if scale is not None:
                self.lib.free_tensor(scale_c)
            if B is not None:
                self.lib.free_tensor(b_c)
            self.lib.free_tensor(out_c)
            self.lib.free_tensor(mean_c)
            self.lib.free_tensor(inv_std_c)
            outputs = (
                Tensor(*x.size, dtype=self.dtype, data=y_data),
                Tensor(*reduction_shape, dtype=self.stash_dtype, data=mean_data),
                Tensor(*reduction_shape, dtype=self.stash_dtype, data=inv_std_data),
            )
            selected = tuple(value for name, value in zip(self.outputs, outputs) if name)
            return {"tensor": selected[0] if len(selected) == 1 else selected, "parameters": None}
        stash_np_dtype = np.float32 if self.stash_dtype == "bfloat16" else nn.DTYPE_TO_NUMPY.get(self.stash_dtype, np.float32)
        work = x_data.astype(stash_np_dtype, copy=False).reshape(row_number, col_number)
        mean = np.mean(work, axis=1, keepdims=True)
        inv_std = np.reciprocal(np.sqrt(np.mean((work - mean) ** 2, axis=1, keepdims=True) + self.epsilon))
        normalized = ((work - mean) * inv_std).reshape(x_data.shape)
        if scale is not None:
            normalized = normalized * _tensor_data_as_numeric(scale)
        if B is not None:
            normalized = normalized + _tensor_data_as_numeric(B)

        reduction_shape = tuple(x_data.shape[:axis]) + (1,) * (rank - axis)
        y = Tensor(*x.size, dtype=self.dtype, data=_cast_numeric_to_dtype(normalized, self.dtype))
        mean_tensor = Tensor(*reduction_shape, dtype=self.stash_dtype, data=_cast_numeric_to_dtype(mean.reshape(reduction_shape), self.stash_dtype))
        inv_std_tensor = Tensor(*reduction_shape, dtype=self.stash_dtype, data=_cast_numeric_to_dtype(inv_std.reshape(reduction_shape), self.stash_dtype))
        outputs = (y, mean_tensor, inv_std_tensor)
        selected = tuple(value for name, value in zip(self.outputs, outputs) if name)
        return {"tensor": selected[0] if len(selected) == 1 else selected, "parameters": None}

    # 执行 `LayerNormalization` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, scale=None, B=None):
        rank = len(x.size)
        axis = self.axis if self.axis >= 0 else self.axis + rank
        if axis < 0 or axis >= rank:
            raise ValueError(f"LayerNormalization axis {self.axis} is out of bounds for rank {rank}")
        reduction_shape = tuple(x.size[:axis]) + (1,) * (rank - axis)
        outputs = (
            Tensor_(*x.size, dtype=self.dtype),
            Tensor_(*reduction_shape, dtype=self.stash_dtype),
            Tensor_(*reduction_shape, dtype=self.stash_dtype),
        )
        selected = tuple(value for name, value in zip(self.outputs, outputs) if name)
        return {"tensor": selected[0] if len(selected) == 1 else selected, "parameters": None}


class Hardmax(Ops):
    # 初始化 `Hardmax` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=-1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.hardmax_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]

    # 执行 `Hardmax` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input):
        out_tensor = Tensor(*input.size, dtype=self.dtype)
        input_c = self._numpy_to_ctensor(input.data, input.dtype)
        output_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        
        self.lib.hardmax_forward(input_c, output_c, ctypes.c_int(self.axis))
        
        out_tensor.data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c); self.lib.free_tensor(output_c)
        return {"tensor": out_tensor, "parameters": None}

    # 执行 `Hardmax` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input):
        return {"tensor": Tensor_(*input.size, dtype=self.dtype), "parameters": None}


class LogSoftmax(Ops):
    # 初始化 `LogSoftmax` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=-1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.log_softmax_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]

    # 执行 `LogSoftmax` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input):
        out_tensor = Tensor(*input.size, dtype=self.dtype)
        input_c = self._numpy_to_ctensor(input.data, input.dtype)
        output_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        
        self.lib.log_softmax_forward(input_c, output_c, ctypes.c_int(self.axis))
        
        out_tensor.data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c); self.lib.free_tensor(output_c)
        return {"tensor": out_tensor, "parameters": None}

    # 执行 `LogSoftmax` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input):
        return {"tensor": Tensor_(*input.size, dtype=self.dtype), "parameters": None}


class LpNormalization(Ops):
    # 初始化 `LpNormalization` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=-1, p=2, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.p = p
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.lp_normalization_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int]

    # 执行 `LpNormalization` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input):
        axis = self.axis if self.axis >= 0 else self.axis + len(input.size)
        if axis < 0 or axis >= len(input.size):
            raise ValueError(f"LpNormalization axis {self.axis} is out of bounds for rank {len(input.size)}")
        if self.lib is not None and input.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            output_shape_c = (ctypes.c_int * len(input.size))(*input.size)
            output_c = self.lib.create_tensor(output_shape_c, len(input.size), nn.DTYPE_MAP[self.dtype])
            self.lib.lp_normalization_forward(input_c, output_c, ctypes.c_int(axis), ctypes.c_int(self.p))
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*input.size, dtype=self.dtype, data=out_data), "parameters": None}

        data = _tensor_data_as_numeric(input)
        norm = np.power(np.power(np.abs(data), self.p).sum(axis=axis), 1.0 / self.p)
        norm = np.expand_dims(norm, axis)
        out_data = np.where(norm == 0, 0, data / norm)
        out_data = _cast_numeric_to_dtype(out_data, self.dtype)
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `LpNormalization` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input):
        return {"tensor": Tensor_(*input.size, dtype=self.dtype), "parameters": None}


class GroupNormalization(Ops):
    # 初始化 `GroupNormalization` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, num_groups, epsilon=1e-5, dtype="float32", version="18"):
        super().__init__(inputs, outputs)
        self.num_groups = num_groups
        self.epsilon = epsilon
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.group_norm_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_float
            ]

    # 执行 `GroupNormalization` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, scale, bias):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        s_c = self._numpy_to_ctensor(scale.data, scale.dtype)
        b_c = self._numpy_to_ctensor(bias.data, bias.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        
        self.lib.group_norm_forward(x_c, s_c, b_c, out_c, ctypes.c_int(self.num_groups), ctypes.c_float(self.epsilon))
        
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(s_c); self.lib.free_tensor(b_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}

    # 执行 `GroupNormalization` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, scale, bias):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}
