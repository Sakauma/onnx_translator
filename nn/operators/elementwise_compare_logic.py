"""文件功能：保存 `elementwise_compare_logic` 分组中的 ONNX 算子实现。
作者：Egor Izmaylov
时间：2026-06-02
"""

from .common import *

class Equal(Ops):
    # 初始化 `Equal` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Equal` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "equal_forward"), "parameters": None}
    # 执行 `Equal` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}


class Greater(Ops):
    # 初始化 `Greater` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Greater` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "greater_forward"), "parameters": None}
    # 执行 `Greater` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}


class Less(Ops):
    # 初始化 `Less` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Less` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "less_forward"), "parameters": None}
    # 执行 `Less` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}


class GreaterOrEqual(Ops):
    # 初始化 `GreaterOrEqual` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `GreaterOrEqual` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "greater_or_equal_forward"), "parameters": None}
    # 执行 `GreaterOrEqual` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}


class LessOrEqual(Ops):
    # 初始化 `LessOrEqual` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `LessOrEqual` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "less_or_equal_forward"), "parameters": None}
    # 执行 `LessOrEqual` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}


class Not(Ops):
    # 初始化 `Not` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Not` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "not_forward"), "parameters": None}
    # 执行 `Not` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class And(Ops):
    # 初始化 `And` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `And` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "and_forward"), "parameters": None}
    # 执行 `And` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}


class Or(Ops):
    # 初始化 `Or` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Or` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "or_forward"), "parameters": None}
    # 执行 `Or` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}


class Xor(Ops):
    # 初始化 `Xor` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Xor` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "xor_forward"), "parameters": None}
    # 执行 `Xor` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}


class IsNaN(Ops):
    # 初始化 `IsNaN` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `IsNaN` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "isnan_forward"), "parameters": None}
    # 执行 `IsNaN` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Sin(Ops):
    # 初始化 `Sin` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Sin` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "sin_forward"), "parameters": None}
    # 执行 `Sin` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Tan(Ops):
    # 初始化 `Tan` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Tan` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "tan_forward"), "parameters": None}
    # 执行 `Tan` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Atan(Ops):
    # 初始化 `Atan` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Atan` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "atan_forward"), "parameters": None}
    # 执行 `Atan` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Sign(Ops):
    # 初始化 `Sign` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Sign` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "sign_forward"), "parameters": None}
    # 执行 `Sign` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Identity(Ops):
    # 初始化 `Identity` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.identity_forward.argtypes = [ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor)]
    # 执行 `Identity` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        if isinstance(x, Tensor):
            out_dtype = self.dtype or x.dtype
            if self.lib is not None and x.dtype in nn.DTYPE_MAP and out_dtype == x.dtype:
                input_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
                output_shape_c = (ctypes.c_int * len(x.size))(*x.size)
                output_c = self.lib.create_tensor(output_shape_c, len(x.size), nn.DTYPE_MAP[out_dtype])
                self.lib.identity_forward(input_c, output_c)
                out_data = self._ctensor_to_numpy(output_c, out_dtype)
                self.lib.free_tensor(input_c)
                self.lib.free_tensor(output_c)
            else:
                out_data = np.asarray(x.data).copy()
                out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(out_dtype, out_data.dtype))
            return {"tensor": Tensor(*x.size, dtype=out_dtype, data=out_data), "parameters": None}
        return {"tensor": x, "parameters": None}
    # 执行 `Identity` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        if isinstance(x, Tensor_):
            return {"tensor": Tensor_(*x.size, dtype=self.dtype or x.dtype), "parameters": None}
        return {"tensor": x, "parameters": None}


class Mod(Ops):
    # 初始化 `Mod` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, fmod=0, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.fmod = fmod 
        self.version = version
        
        if self.lib:
            self.lib.mod_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor), 
                ctypes.POINTER(nn.CTensor), ctypes.c_int
            ]

    # 执行 `Mod` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, a, b):
        try:
            a_bc, b_bc = np.broadcast_arrays(a.data, b.data)
        except ValueError:
            raise ValueError(f"Mod operator broadcast failed: {a.size} vs {b.size}")
        
        out_shape = a_bc.shape
        
        if self.dtype:
            out_dtype = self.dtype
        else:
            # 如果没指定 dtype，自动推断
            res_type = np.result_type(a_bc, b_bc)
            out_dtype = nn.NUMPY_TO_DTYPE.get(res_type.type, "float32")
        
        np_type_a = nn.DTYPE_TO_NUMPY[a.dtype]
        np_type_b = nn.DTYPE_TO_NUMPY[b.dtype]
        
        a_data_safe = np.ascontiguousarray(a_bc.astype(np_type_a))
        b_data_safe = np.ascontiguousarray(b_bc.astype(np_type_b))
        if self.lib is None or out_dtype not in nn.DTYPE_MAP:
            if self.fmod == 1:
                out_data = np.fmod(a_data_safe, b_data_safe)
            elif np.issubdtype(a_data_safe.dtype, np.floating):
                out_data = np.nan_to_num(np.fmod(a_data_safe, b_data_safe))
            else:
                out_data = np.nan_to_num(np.mod(a_data_safe, b_data_safe))
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(out_dtype, out_data.dtype))
            return {"tensor": Tensor(*out_shape, dtype=out_dtype, data=out_data), "parameters": None}
        
        a_c = self._numpy_to_ctensor(a_data_safe, a.dtype)
        b_c = self._numpy_to_ctensor(b_data_safe, b.dtype)
        
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[out_dtype])
    
        self.lib.mod_forward(a_c, b_c, output_c, ctypes.c_int(self.fmod))
        
        out_data = self._ctensor_to_numpy(output_c, out_dtype)
        self.lib.free_tensor(a_c)
        self.lib.free_tensor(b_c)
        self.lib.free_tensor(output_c)
        
        return {"tensor": Tensor(*out_shape, dtype=out_dtype, data=out_data), "parameters": None}

    # 执行 `Mod` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, a, b):
        shape = np.broadcast_shapes(a.size, b.size)
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}


class Where(Ops):
    # 初始化 `Where` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.where_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor),
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor)
            ]
    # 执行 `Where` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, cond, x, y):
        cond_data, x_data, y_data = np.broadcast_arrays(
            np.asarray(cond.data, dtype=np.bool_),
            np.asarray(x.data),
            np.asarray(y.data),
        )
        out_shape = cond_data.shape
        if (
            self.lib is not None
            and cond.dtype in nn.DTYPE_MAP
            and x.dtype in nn.DTYPE_MAP
            and y.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
        ):
            cond_c = self._numpy_to_ctensor(np.ascontiguousarray(cond_data), cond.dtype)
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x_data), x.dtype)
            y_c = self._numpy_to_ctensor(np.ascontiguousarray(y_data), y.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.where_forward(cond_c, x_c, y_c, output_c)
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(cond_c)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(y_c)
            self.lib.free_tensor(output_c)
        else:
            out_data = np.where(cond_data, x_data, y_data)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}
    # 执行 `Where` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, cond, x, y):
        try: shape = np.broadcast_shapes(cond.size, x.size, y.size)
        except: shape = x.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}


class BitwiseAnd(Ops):
    # 初始化 `BitwiseAnd` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="int32", version="18"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `BitwiseAnd` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "bitwise_and_forward"), "parameters": None}
    # 执行 `BitwiseAnd` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}


class BitwiseOr(Ops):
    # 初始化 `BitwiseOr` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="int32", version="18"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `BitwiseOr` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "bitwise_or_forward"), "parameters": None}
    # 执行 `BitwiseOr` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}


class BitwiseXor(Ops):
    # 初始化 `BitwiseXor` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="int32", version="18"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `BitwiseXor` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "bitwise_xor_forward"), "parameters": None}
    # 执行 `BitwiseXor` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}


class BitwiseNot(Ops):
    # 初始化 `BitwiseNot` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="int32", version="18"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `BitwiseNot` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "bitwise_not_forward"), "parameters": None}
    # 执行 `BitwiseNot` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class BitShift(Ops):
    # 初始化 `BitShift` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, direction="LEFT", dtype="int32", version="11"):
        super().__init__(inputs, outputs)
        self.direction = direction.upper() # "LEFT" or "RIGHT"
        self.direction_int = 0 if self.direction == "LEFT" else 1
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.bit_shift_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), 
                ctypes.POINTER(CTensor), ctypes.c_int
            ]

    # 执行 `BitShift` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, a, b):
        out_tensor = self._execute_binary_custom(a, b)
        return {"tensor": out_tensor, "parameters": None}

    # 封装 `_execute_binary_custom` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _execute_binary_custom(self, input_a, input_b):
        try:
            a_bc, b_bc = np.broadcast_arrays(input_a.data, input_b.data)
        except ValueError as e:
            raise e
        
        out_shape = a_bc.shape
        out_dtype = self.dtype
        
        a_c = self._numpy_to_ctensor(np.ascontiguousarray(a_bc), input_a.dtype)
        b_c = self._numpy_to_ctensor(np.ascontiguousarray(b_bc), input_b.dtype)
        
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), DTYPE_MAP[out_dtype])
        
        self.lib.bit_shift_forward(a_c, b_c, output_c, ctypes.c_int(self.direction_int))
        
        out_data = self._ctensor_to_numpy(output_c, out_dtype)
        self.lib.free_tensor(a_c); self.lib.free_tensor(b_c); self.lib.free_tensor(output_c)
        
        return Tensor(*out_shape, dtype=out_dtype, data=out_data)

    # 执行 `BitShift` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}
