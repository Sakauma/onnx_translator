# /**
#   ******************************************************************************
#   * @file        random_ops.py
#   * @author      Egor Izmaylov
#   * @brief       保存 `random_ops` 分组中的 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *


# 将 ONNX TensorProto dtype id 或本地 dtype 字符串统一解析成本项目 dtype 名称。
def _resolve_random_dtype(dtype, default=None):
    if dtype is None:
        return default
    if isinstance(dtype, str):
        return dtype
    return nn.onnx_dtype_mapping.get(dtype, default or "float32")


# 将随机算子的浮点样本写回目标 dtype；bfloat16 必须编码为 uint16 位模式。
def _cast_random_output(values, dtype):
    return _cast_numeric_to_dtype(np.asarray(values, dtype=np.float32), dtype)


class EyeLike(Ops):
    # 初始化 `EyeLike` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, k=0, dtype=None, version="17"):
        super(EyeLike, self).__init__(inputs, outputs)
        self.k = k
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.eye_like_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.c_int]

    # 执行 `EyeLike` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        if len(input.size) != 2:
            raise ValueError(f"EyeLike expects a 2-D input, got shape {input.size}")
        out_dtype = self.dtype or input.dtype
        if self.lib is not None and out_dtype in nn.DTYPE_MAP:
            output_shape_c = (ctypes.c_int * 2)(*input.size)
            out_c = self.lib.create_tensor(output_shape_c, 2, nn.DTYPE_MAP[out_dtype])
            self.lib.eye_like_forward(out_c, ctypes.c_int(self.k))
            out_data = self._ctensor_to_numpy(out_c, out_dtype)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*input.size, dtype=out_dtype, data=out_data), "parameters": None, "graph": None}
        np_dtype = nn.DTYPE_TO_NUMPY.get(out_dtype)
        if np_dtype is None:
            raise ValueError(f"EyeLike dtype {out_dtype!r} is not supported")
        out_data = np.eye(input.size[0], input.size[1], k=self.k, dtype=np_dtype)
        return {"tensor": Tensor(*input.size, dtype=out_dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `EyeLike` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        if len(input.size) != 2:
            raise ValueError(f"EyeLike expects a 2-D input, got shape {input.size}")
        return {"tensor": Tensor_(*input.size, dtype=self.dtype or input.dtype), "parameters": None, "graph": None}


class RandomUniformLike(Ops):
    # 初始化 `RandomUniformLike` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, high=1.0, low=0.0, seed=0.0, dtype=None, version="17"):
        super().__init__(inputs, outputs)
        self.high = high
        self.low = low
        self.seed = seed
        self.dtype = _resolve_random_dtype(dtype) # None means infer from input, matching ONNX Like-op semantics.
        self.version = version

    # 执行 `RandomUniformLike` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input):
        target_dtype = self.dtype if self.dtype else input.dtype
        out_tensor = Tensor(*input.size, dtype=target_dtype)
        if self.lib is not None and target_dtype in nn.DTYPE_MAP:
            out_c = self._numpy_to_ctensor(out_tensor.data, target_dtype)
            self.lib.random_uniform_like_forward(out_c, ctypes.c_float(self.low), ctypes.c_float(self.high), ctypes.c_float(self.seed))
            out_data = self._ctensor_to_numpy(out_c, target_dtype)
            self.lib.free_tensor(out_c)
        else:
            seed = None if self.seed is None or self.seed == 0.0 else int(self.seed)
            rng = np.random.default_rng(seed)
            out_data = _cast_random_output(rng.uniform(self.low, self.high, size=input.size), target_dtype)
        out_tensor.data = out_data
        
        return {"tensor": out_tensor, "parameters": None}

    # 执行 `RandomUniformLike` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input):
        target_dtype = self.dtype if self.dtype else input.dtype
        return {"tensor": Tensor_(*input.size, dtype=target_dtype), "parameters": None}


class RandomUniform(Ops):
    # 初始化 `RandomUniform` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, high=1.0, low=0.0, seed=0.0, dtype=1, shape=None, version="17"):
        super().__init__(inputs, outputs)
        self.high = high
        self.low = low
        self.seed = seed
        self.dtype = _resolve_random_dtype(dtype, "float32")
        self.shape_val = shape
        self.version = version

    # 执行 `RandomUniform` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self):
        if self.shape_val is None:
            raise ValueError("RandomUniform requires 'shape' attribute")
        out_shape = tuple(self.shape_val)
        if self.lib is not None and self.dtype in nn.DTYPE_MAP:
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), DTYPE_MAP[self.dtype])
            self.lib.random_uniform_like_forward(output_c, ctypes.c_float(self.low), ctypes.c_float(self.high), ctypes.c_float(self.seed))
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(output_c)
        else:
            seed = None if self.seed is None or self.seed == 0.0 else int(self.seed)
            rng = np.random.default_rng(seed)
            out_data = _cast_random_output(rng.uniform(self.low, self.high, size=out_shape), self.dtype)
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `RandomUniform` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self):
        out_shape = tuple(self.shape_val) if self.shape_val is not None else (1,)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}


class RandomNormal(Ops):
    # 初始化 `RandomNormal` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, mean=0.0, scale=1.0, seed=0.0, dtype=1, shape=None, version="17"):
        super().__init__(inputs, outputs)
        self.mean = mean
        self.scale = scale
        self.seed = seed
        self.dtype = _resolve_random_dtype(dtype, "float32")
        self.shape_val = shape # list
        self.version = version
        if self.lib:
            self.lib.random_normal_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float, ctypes.c_float, ctypes.c_float]

    # 执行 `RandomNormal` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self):
        # Shape 必须是初始化属性
        if self.shape_val is None:
            raise ValueError("RandomNormal requires 'shape' attribute")

        out_shape = tuple(self.shape_val)
        if self.lib is not None and self.dtype in nn.DTYPE_MAP:
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), DTYPE_MAP[self.dtype])

            self.lib.random_normal_forward(output_c, ctypes.c_float(self.mean), ctypes.c_float(self.scale), ctypes.c_float(self.seed))

            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(output_c)
        else:
            seed = None if self.seed is None or self.seed == 0.0 else int(self.seed)
            rng = np.random.default_rng(seed)
            out_data = _cast_random_output(rng.normal(self.mean, self.scale, size=out_shape), self.dtype)
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `RandomNormal` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self):
        out_shape = tuple(self.shape_val) if self.shape_val is not None else (1,)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}


class RandomNormalLike(Ops):
    # 初始化 `RandomNormalLike` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, mean=0.0, scale=1.0, seed=0.0, dtype=None, version="17"):
        super().__init__(inputs, outputs)
        self.mean = mean
        self.scale = scale
        self.seed = seed
        self.dtype = _resolve_random_dtype(dtype)
        self.version = version
        if self.lib:
            self.lib.random_normal_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float, ctypes.c_float, ctypes.c_float]

    # 执行 `RandomNormalLike` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input):
        target_dtype = self.dtype if self.dtype else input.dtype
        out_shape = input.size

        if self.lib is not None and target_dtype in nn.DTYPE_MAP:
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), DTYPE_MAP[target_dtype])

            self.lib.random_normal_forward(output_c, ctypes.c_float(self.mean), ctypes.c_float(self.scale), ctypes.c_float(self.seed))

            out_data = self._ctensor_to_numpy(output_c, target_dtype)
            self.lib.free_tensor(output_c)
        else:
            seed = None if self.seed is None or self.seed == 0.0 else int(self.seed)
            rng = np.random.default_rng(seed)
            out_data = _cast_random_output(rng.normal(self.mean, self.scale, size=out_shape), target_dtype)
        return {"tensor": Tensor(*out_shape, dtype=target_dtype, data=out_data), "parameters": None}

    # 执行 `RandomNormalLike` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input):
        target_dtype = self.dtype if self.dtype else input.dtype
        return {"tensor": Tensor_(*input.size, dtype=target_dtype), "parameters": None}


class Bernoulli(Ops):
    # 初始化 `Bernoulli` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, seed=0.0, dtype=None, version="17"):
        super().__init__(inputs, outputs)
        self.seed = seed
        self.dtype = _resolve_random_dtype(dtype)
        self.version = version
        if self.lib:
            self.lib.bernoulli_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]

    # 执行 `Bernoulli` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input):
        target_dtype = self.dtype if self.dtype else input.dtype
        out_shape = input.size
        if self.lib is None or target_dtype not in nn.DTYPE_MAP:
            seed = None if self.seed is None or self.seed == 0.0 else int(self.seed)
            rng = np.random.default_rng(seed)
            probs = _tensor_data_as_numeric(input).astype(np.float64, copy=False)
            out_data = _cast_random_output(rng.binomial(1, p=probs), target_dtype)
            return {"tensor": Tensor(*out_shape, dtype=target_dtype, data=out_data), "parameters": None}
        
        input_c = self._numpy_to_ctensor(input.data, input.dtype)
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), DTYPE_MAP[target_dtype])
        
        self.lib.bernoulli_forward(input_c, output_c, ctypes.c_float(self.seed))
        
        out_data = self._ctensor_to_numpy(output_c, target_dtype)
        self.lib.free_tensor(input_c); self.lib.free_tensor(output_c)
        return {"tensor": Tensor(*out_shape, dtype=target_dtype, data=out_data), "parameters": None}

    # 执行 `Bernoulli` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input):
        target_dtype = self.dtype if self.dtype else input.dtype
        return {"tensor": Tensor_(*input.size, dtype=target_dtype), "parameters": None}


class Dropout(Ops):
    # 初始化 `Dropout` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, seed=None, ratio=0.5, training_mode=0, version="17"):
        super().__init__(inputs, outputs)
        self.seed = seed
        self.default_ratio = ratio
        self.training_mode = training_mode
        self.version = version
        if self.lib:
            self.lib.dropout_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float, ctypes.c_int]

    # 执行 `Dropout` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, data, ratio=None, training_mode=None):
        r = float(self.default_ratio)
        if ratio is not None:
            r = float(_tensor_data_as_numeric(ratio).item())
        
        if r < 0.0 or r >= 1.0:
            raise ValueError(f"Dropout ratio must be in [0, 1), got {r}")

        mode = bool(self.training_mode)
        if training_mode is not None:
            mode = bool(training_mode.data.item())

        if self.lib is not None and data.dtype in nn.DTYPE_MAP and (not mode or r == 0.0):
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(data.data), data.dtype)
            output_shape_c = (ctypes.c_int * len(data.size))(*data.size)
            output_c = self.lib.create_tensor(output_shape_c, len(data.size), nn.DTYPE_MAP[data.dtype])
            self.lib.dropout_forward(input_c, output_c, ctypes.c_float(r), ctypes.c_int(int(mode)))
            out_data = self._ctensor_to_numpy(output_c, data.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
            mask_data = np.ones(data.size, dtype=np.bool_)
        elif mode and r > 0.0:
            if self.seed is None:
                mask_data = np.random.default_rng().random(data.size) >= r
            else:
                rng = np.random.RandomState(int(self.seed))
                mask_data = rng.uniform(0.0, 1.0, data.size) >= r
            numeric = _tensor_data_as_numeric(data)
            out_data = _cast_numeric_to_dtype(numeric * mask_data.astype(numeric.dtype) / (1.0 - r), data.dtype)
        else:
            mask_data = np.ones(data.size, dtype=np.bool_)
            out_data = data.data.copy()

        output_tensor = Tensor(*data.size, dtype=data.dtype, data=out_data.astype(data.data.dtype, copy=False))
        if len(self.outputs) > 1 and self.outputs[1]:
            mask_tensor = Tensor(*data.size, dtype="bool", data=mask_data)
            return {"tensor": (output_tensor, mask_tensor), "parameters": None}
        return {"tensor": output_tensor, "parameters": None}

    # 执行 `Dropout` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, data, ratio=None, training_mode=None):
        output_tensor = Tensor_(*data.size, dtype=data.dtype)
        if len(self.outputs) > 1 and self.outputs[1]:
            return {"tensor": (output_tensor, Tensor_(*data.size, dtype="bool")), "parameters": None}
        return {"tensor": output_tensor, "parameters": None}
