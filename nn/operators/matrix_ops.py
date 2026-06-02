# /**
#   ******************************************************************************
#   * @file        matrix_ops.py
#   * @author      Egor Izmaylov
#   * @brief       保存 `matrix_ops` 分组中的 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *

class Gemm(Ops):
    # 初始化 `Gemm` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, alpha, beta, transA, transB, dtype, version="17"):
        super(Gemm, self).__init__(inputs, outputs)
        self.alpha = alpha
        self.beta = beta
        self.transA = transA
        self.transB = transB
        self.dtype = dtype

        if self.lib:
            self.lib.gemm_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int
            ]

    # 执行 `Gemm` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, A: Tensor, B: Tensor, C: Tensor = None) -> dict:
        # 维度推断 (假设 A, B 至少 2D)
        M = A.size[0] if self.transA == 0 else A.size[1]
        N = B.size[1] if self.transB == 0 else B.size[0]
        out_shape = (M, N)

        a_c = self._numpy_to_ctensor(A.data, A.dtype)
        b_c = self._numpy_to_ctensor(B.data, B.dtype)
        #c_c = self._numpy_to_ctensor(C.data, C.dtype) if C is not None else ctypes.POINTER(CTensor)()
        c_c = ctypes.POINTER(CTensor)()
        if C is not None:
            c_data = C.data
            if C.data.ndim == 1:
                if c_data.shape[0] == N:
                    c_data = c_data.reshape(1, -1)
                elif c_data.shape[0] == M:
                    c_data = c_data.reshape(-1, 1)
            c_c = self._numpy_to_ctensor(np.ascontiguousarray(c_data), C.dtype)

        output_shape_c = (ctypes.c_int * 2)(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, 2, DTYPE_MAP[self.dtype])

        self.lib.gemm_forward(a_c, b_c, c_c, output_c, 
                              ctypes.c_float(self.alpha), ctypes.c_float(self.beta), 
                              ctypes.c_int(self.transA), ctypes.c_int(self.transB))

        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(a_c); self.lib.free_tensor(b_c); self.lib.free_tensor(output_c)
        if C is not None: self.lib.free_tensor(c_c)

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # 执行 `Gemm` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, A: Tensor_, B: Tensor_, C: Tensor_ = None) -> dict:
        M = A.size[0] if self.transA == 0 else A.size[1]
        N = B.size[1] if self.transB == 0 else B.size[0]
        output_tensor = Tensor_(M, N, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values


class PRelu(Ops):
    # 初始化 `PRelu` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super(PRelu, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.prelu_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)
            ]

    # 执行 `PRelu` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x: Tensor, slope: Tensor) -> dict:
        x_data, slope_data = np.broadcast_arrays(x.data, slope.data)
        if self.lib is not None and self.dtype in nn.DTYPE_MAP and x.dtype in nn.DTYPE_MAP and slope.dtype in nn.DTYPE_MAP:
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x_data.astype(nn.DTYPE_TO_NUMPY[x.dtype], copy=False)), x.dtype)
            slope_c = self._numpy_to_ctensor(
                np.ascontiguousarray(slope_data.astype(nn.DTYPE_TO_NUMPY[slope.dtype], copy=False)), slope.dtype
            )
            output_shape_c = (ctypes.c_int * len(x_data.shape))(*x_data.shape)
            output_c = self.lib.create_tensor(output_shape_c, len(x_data.shape), nn.DTYPE_MAP[self.dtype])
            self.lib.prelu_forward(x_c, slope_c, output_c)
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(slope_c)
            self.lib.free_tensor(output_c)
        else:
            out_data = np.where(x_data >= 0, x_data, x_data * slope_data)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `PRelu` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x: Tensor_, slope: Tensor_) -> dict:
        out_shape = np.broadcast_shapes(x.size, slope.size)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None, "graph": None}


class Det(Ops):
    # 初始化 `Det` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super(Det, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.det_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

    # 执行 `Det` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x: Tensor) -> dict:
        if len(x.size) < 2 or x.size[-1] != x.size[-2]:
            raise ValueError(f"Det expects input shape [..., M, M], got {x.size}")
        out_shape = tuple(x.size[:-2])
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.det_forward(input_c, output_c)
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

        out_data = np.linalg.det(x.data)
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `Det` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x: Tensor_) -> dict:
        if len(x.size) < 2 or x.size[-1] != x.size[-2]:
            raise ValueError(f"Det expects input shape [..., M, M], got {x.size}")
        return {"tensor": Tensor_(*x.size[:-2], dtype=self.dtype), "parameters": None, "graph": None}


class MatMulInteger(Ops):
    # 初始化 `MatMulInteger` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="int32", version="17"):
        super(MatMulInteger, self).__init__(inputs, outputs)
        self.dtype = "int32"
        self.version = version
        if self.lib:
            self.lib.matmul_integer_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)
            ]

    # 封装 `_zero_point_data` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _zero_point_data(zero_point, dtype):
        if zero_point is None:
            return np.array(0, dtype=nn.DTYPE_TO_NUMPY[dtype])
        return zero_point.data

    # 执行 `MatMulInteger` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, A: Tensor, B: Tensor, a_zero_point: Tensor = None, b_zero_point: Tensor = None) -> dict:
        if (
            self.lib is not None
            and A.dtype in nn.DTYPE_MAP
            and B.dtype in nn.DTYPE_MAP
            and (a_zero_point is None or a_zero_point.dtype in nn.DTYPE_MAP)
            and (b_zero_point is None or b_zero_point.dtype in nn.DTYPE_MAP)
        ):
            data_a, data_b, out_shape_for_c, final_shape = _prepare_matmul_c_shapes(A, B)
            a_zp = _broadcast_matmul_param(a_zero_point, data_a.shape, A.dtype, "row")
            b_zp = _broadcast_matmul_param(b_zero_point, data_b.shape, B.dtype, "col")

            a_c = self._numpy_to_ctensor(np.ascontiguousarray(data_a.astype(nn.DTYPE_TO_NUMPY[A.dtype], copy=False)), A.dtype)
            b_c = self._numpy_to_ctensor(np.ascontiguousarray(data_b.astype(nn.DTYPE_TO_NUMPY[B.dtype], copy=False)), B.dtype)
            a_zp_c = self._numpy_to_ctensor(a_zp, A.dtype)
            b_zp_c = self._numpy_to_ctensor(b_zp, B.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape_for_c))(*out_shape_for_c)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape_for_c), nn.DTYPE_MAP[self.dtype])
            self.lib.matmul_integer_forward(a_c, b_c, a_zp_c, b_zp_c, out_c)
            out_data = self._ctensor_to_numpy(out_c, self.dtype).reshape(final_shape)
            self.lib.free_tensor(a_c)
            self.lib.free_tensor(b_c)
            self.lib.free_tensor(a_zp_c)
            self.lib.free_tensor(b_zp_c)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*final_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}
        if a_zero_point is not None or b_zero_point is not None:
            data_a, data_b, _out_shape_for_c, final_shape = _prepare_matmul_c_shapes(A, B)
            a_zp = _broadcast_matmul_param(a_zero_point, data_a.shape, A.dtype, "row").astype(np.int32)
            b_zp = _broadcast_matmul_param(b_zero_point, data_b.shape, B.dtype, "col").astype(np.int32)
            a = data_a.astype(np.int32) - a_zp
            b = data_b.astype(np.int32) - b_zp
            out_data = np.matmul(a, b).astype(np.int32).reshape(final_shape)
        else:
            a = A.data.astype(np.int32)
            b = B.data.astype(np.int32)
            out_data = np.matmul(a, b).astype(np.int32)
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `MatMulInteger` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, A: Tensor_, B: Tensor_, a_zero_point: Tensor_ = None, b_zero_point: Tensor_ = None) -> dict:
        return {"tensor": Tensor_(*_matmul_output_shape(A.size, B.size), dtype=self.dtype), "parameters": None, "graph": None}


class QLinearMatMul(Ops):
    # 初始化 `QLinearMatMul` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="uint8", version="17"):
        super(QLinearMatMul, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.qlinear_matmul_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)
            ]

    # 执行 `QLinearMatMul` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point):
        out_dtype = y_zero_point.dtype if y_zero_point is not None else self.dtype
        if (
            self.lib is not None
            and out_dtype in nn.DTYPE_MAP
            and all(t.dtype in nn.DTYPE_MAP for t in (a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point))
        ):
            data_a, data_b, out_shape_for_c, final_shape = _prepare_matmul_c_shapes(a, b)
            a_scale_data = _broadcast_matmul_param(a_scale, data_a.shape, a_scale.dtype, "row")
            a_zp_data = _broadcast_matmul_param(a_zero_point, data_a.shape, a.dtype, "row")
            b_scale_data = _broadcast_matmul_param(b_scale, data_b.shape, b_scale.dtype, "col")
            b_zp_data = _broadcast_matmul_param(b_zero_point, data_b.shape, b.dtype, "col")
            y_scale_data = _broadcast_output_param(y_scale, out_shape_for_c, y_scale.dtype)
            y_zp_data = _broadcast_output_param(y_zero_point, out_shape_for_c, out_dtype)

            a_c = self._numpy_to_ctensor(np.ascontiguousarray(data_a.astype(nn.DTYPE_TO_NUMPY[a.dtype], copy=False)), a.dtype)
            a_scale_c = self._numpy_to_ctensor(a_scale_data, a_scale.dtype)
            a_zp_c = self._numpy_to_ctensor(a_zp_data, a.dtype)
            b_c = self._numpy_to_ctensor(np.ascontiguousarray(data_b.astype(nn.DTYPE_TO_NUMPY[b.dtype], copy=False)), b.dtype)
            b_scale_c = self._numpy_to_ctensor(b_scale_data, b_scale.dtype)
            b_zp_c = self._numpy_to_ctensor(b_zp_data, b.dtype)
            y_scale_c = self._numpy_to_ctensor(y_scale_data, y_scale.dtype)
            y_zp_c = self._numpy_to_ctensor(y_zp_data, out_dtype)
            output_shape_c = (ctypes.c_int * len(out_shape_for_c))(*out_shape_for_c)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape_for_c), nn.DTYPE_MAP[out_dtype])
            self.lib.qlinear_matmul_forward(a_c, a_scale_c, a_zp_c, b_c, b_scale_c, b_zp_c, y_scale_c, y_zp_c, out_c)
            out_data = self._ctensor_to_numpy(out_c, out_dtype).reshape(final_shape)
            for c_tensor in (a_c, a_scale_c, a_zp_c, b_c, b_scale_c, b_zp_c, y_scale_c, y_zp_c, out_c):
                self.lib.free_tensor(c_tensor)
            return {"tensor": Tensor(*final_shape, dtype=out_dtype, data=out_data), "parameters": None, "graph": None}

        data_a, data_b, out_shape_for_c, final_shape = _prepare_matmul_c_shapes(a, b)
        a_scale_data = _broadcast_matmul_param(a_scale, data_a.shape, a_scale.dtype, "row").astype(np.float64)
        a_zp_data = _broadcast_matmul_param(a_zero_point, data_a.shape, a.dtype, "row").astype(np.int32)
        b_scale_data = _broadcast_matmul_param(b_scale, data_b.shape, b_scale.dtype, "col").astype(np.float64)
        b_zp_data = _broadcast_matmul_param(b_zero_point, data_b.shape, b.dtype, "col").astype(np.int32)

        a_real = (data_a.astype(np.int32) - a_zp_data).astype(np.float64) * a_scale_data
        b_real = (data_b.astype(np.int32) - b_zp_data).astype(np.float64) * b_scale_data
        matmul_real = np.matmul(a_real, b_real)

        y_scale_data = _broadcast_output_param(y_scale, out_shape_for_c, y_scale.dtype).astype(np.float64)
        y_zp_data = _broadcast_output_param(y_zero_point, out_shape_for_c, out_dtype).astype(np.float64)
        out = np.rint(matmul_real / y_scale_data + y_zp_data).astype(np.int64).reshape(final_shape)
        if y_zero_point.dtype == "uint8":
            out = np.clip(out, 0, 255).astype(np.uint8)
            out_dtype = "uint8"
        elif y_zero_point.dtype == "int8":
            out = np.clip(out, -128, 127).astype(np.int8)
            out_dtype = "int8"
        else:
            out_dtype = self.dtype
            out = out.astype(nn.DTYPE_TO_NUMPY.get(out_dtype, np.uint8))
        return {"tensor": Tensor(*out.shape, dtype=out_dtype, data=out), "parameters": None, "graph": None}

    # 执行 `QLinearMatMul` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point):
        out_dtype = y_zero_point.dtype if y_zero_point is not None else self.dtype
        return {"tensor": Tensor_(*_matmul_output_shape(a.size, b.size), dtype=out_dtype), "parameters": None, "graph": None}


class MatMul(Ops):
    # 初始化 `MatMul` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(MatMul, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `MatMul` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input_a: Tensor, input_b: Tensor) -> dict:
        if self.lib is None or self.dtype not in nn.DTYPE_MAP:
            out_data = np.matmul(np.asarray(input_a.data), np.asarray(input_b.data))
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
            return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

        data_a = input_a.data
        data_b = input_b.data
        
        is_a_1d = (data_a.ndim == 1)
        is_b_1d = (data_b.ndim == 1)

        if is_a_1d:
            data_a = data_a[np.newaxis, :]
            
        if is_b_1d:
            data_b = data_b[:, np.newaxis]

        shape_a = list(data_a.shape)
        shape_b = list(data_b.shape)
        
        ndim = max(len(shape_a), len(shape_b))
        M = shape_a[-2]
        K_a = shape_a[-1]
        K_b = shape_b[-2]
        N = shape_b[-1]
        
        if K_a != K_b:
            raise ValueError(f"MatMul shape mismatch: {K_a} != {K_b} (Original shapes: A={input_a.size}, B={input_b.size})")
            
        batch_a = shape_a[:-2]
        batch_b = shape_b[:-2]
        
        try:
            batch_out = np.broadcast_shapes(batch_a, batch_b)
        except ValueError:
            raise ValueError(f"MatMul batch broadcast failed: {batch_a} vs {batch_b}")
            
        out_shape_for_c = list(batch_out) + [M, N]
        
        input_a_c = self._numpy_to_ctensor(data_a, input_a.dtype)
        input_b_c = self._numpy_to_ctensor(data_b, input_b.dtype)
        
        output_shape_c = (ctypes.c_int * len(out_shape_for_c))(*out_shape_for_c)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape_for_c), nn.DTYPE_MAP[self.dtype])
        
        self.lib.matmul_forward(input_a_c, input_b_c, output_c)
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_a_c)
        self.lib.free_tensor(input_b_c)
        self.lib.free_tensor(output_c)

        final_shape = list(out_shape_for_c)
        
        if is_b_1d:
            final_shape.pop(-1)
        if is_a_1d:
            idx_to_pop = -1 if is_b_1d else -2
            final_shape.pop(idx_to_pop)
            
        # 如果变成了标量或形状改变，reshape 数据
        if tuple(final_shape) != tuple(out_shape_for_c):
            out_data = out_data.reshape(final_shape)

        out_tensor = Tensor(*final_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # 执行 `MatMul` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input_a: Tensor_, input_b: Tensor_) -> dict:
        shape_a = list(input_a.size) if isinstance(input_a.size, (list, tuple)) else [input_a.size]
        shape_b = list(input_b.size) if isinstance(input_b.size, (list, tuple)) else [input_b.size]
        if len(shape_a) < 1 or len(shape_b) < 1:
            raise ValueError(f"MatMul inputs must have rank >= 1, got {input_a.size} and {input_b.size}")
        
        is_a_1d = (len(shape_a) == 1)
        is_b_1d = (len(shape_b) == 1)
        
        if is_a_1d: shape_a = [1] + shape_a
        if is_b_1d: shape_b = shape_b + [1]

        if shape_a[-1] != shape_b[-2]:
            raise ValueError(f"MatMul shape mismatch: {shape_a[-1]} != {shape_b[-2]}")
            
        M = shape_a[-2]
        N = shape_b[-1]
        
        batch_a = shape_a[:-2]
        batch_b = shape_b[:-2]
        
        batch_out = np.broadcast_shapes(batch_a, batch_b)
        if is_a_1d and is_b_1d:
            final_shape = list(batch_out)
        elif is_a_1d:
            final_shape = list(batch_out) + [N]
        elif is_b_1d:
            final_shape = list(batch_out) + [M]
        else:
            final_shape = list(batch_out) + [M, N]
        
        output_tensor = Tensor_(*final_shape, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values
