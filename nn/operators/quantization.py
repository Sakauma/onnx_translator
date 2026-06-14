# /**
#   ******************************************************************************
#   * @file        quantization.py
#   * @author      Egor Izmaylov
#   * @brief       按算子职责分组保存 `quantization` 相关 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *


# 将 blocked quantization 的 scale/zero_point 参数展开到输入张量形状，实际数值计算仍交给 C 后端执行。
def _expand_blocked_qdq_param(param, input_shape, axis, block_size):
    if param is None or block_size <= 0:
        return param
    arr = np.asarray(param.data)
    rank = len(input_shape)
    if arr.ndim != rank:
        return param
    safe_axis = axis if axis >= 0 else axis + rank
    if safe_axis < 0 or safe_axis >= rank:
        raise ValueError(f"QDQ axis {axis} is out of bounds for rank {rank}")
    expected_shape = list(input_shape)
    expected_shape[safe_axis] = (input_shape[safe_axis] + block_size - 1) // block_size
    if list(arr.shape) == list(input_shape):
        return param
    if list(arr.shape) != expected_shape:
        raise ValueError(
            f"Blocked QDQ parameter shape {arr.shape} does not match expected {tuple(expected_shape)}"
        )
    expanded = np.repeat(arr, block_size, axis=safe_axis)
    slices = tuple(slice(0, dim) for dim in input_shape)
    expanded = np.ascontiguousarray(expanded[slices].astype(nn.DTYPE_TO_NUMPY[param.dtype], copy=False))
    return Tensor(*expanded.shape, dtype=param.dtype, data=expanded)


class QuantizeLinear(Ops):
    # 初始化 `QuantizeLinear` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=1, dtype=None, output_dtype=None, block_size=0, precision=0, saturate=1, version="17"):
        super(QuantizeLinear, self).__init__(inputs, outputs)
        self.output_dtype = output_dtype
        self.dtype = dtype or output_dtype or "uint8"
        self.axis = axis # 保存 axis
        self.block_size = int(block_size or 0)
        self.precision = int(precision or 0)
        self.saturate = int(saturate)
        self.version = version

    # 封装 `_default_zero_point` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _default_zero_point(self):
        return Tensor(1, dtype=self.dtype, data=np.zeros((1,), dtype=nn.DTYPE_TO_NUMPY[self.dtype]))

    # 执行 `QuantizeLinear` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, y_scale, y_zero_point=None) -> Tensor:
        scale_tensor = y_scale
        zp_tensor = y_zero_point if y_zero_point is not None else self._default_zero_point()

        if self.block_size > 0:
            scale_tensor = _expand_blocked_qdq_param(scale_tensor, list(x.data.shape), self.axis, self.block_size)
            zp_tensor = _expand_blocked_qdq_param(zp_tensor, list(x.data.shape), self.axis, self.block_size)

        # 检查是否需要广播处理 (Scale 是 1D 但 Input 是 ND)
        if y_scale.data.ndim == 1 and x.data.ndim > 1:
            new_shape = [1] * x.data.ndim
            safe_axis = self.axis if self.axis >= 0 else self.axis + x.data.ndim
            if safe_axis < x.data.ndim:
                new_shape[safe_axis] = y_scale.data.size
            scale_tensor = Tensor(*new_shape, dtype=y_scale.dtype, data=y_scale.data.reshape(new_shape))
            if zp_tensor.data.size == y_scale.data.size:
                zp_tensor = Tensor(*new_shape, dtype=zp_tensor.dtype, data=zp_tensor.data.reshape(new_shape))

        c_func_name = "quantize_linear_forward"
        extra_int_arg = None
        if self.precision and hasattr(self.lib, "quantize_linear_forward_precision"):
            c_func_name = "quantize_linear_forward_precision"
            extra_int_arg = self.precision
        out_tensor = self._execute_ternary(x, scale_tensor, zp_tensor, c_func_name, extra_int_arg=extra_int_arg)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # 执行 `QuantizeLinear` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, y_scale, y_zero_point=None) -> Tensor_:
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None, "graph": None}


class DequantizeLinear(Ops):
    # 初始化 `DequantizeLinear` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype=None, axis=1, output_dtype=None, block_size=0, version="17"):
        super(DequantizeLinear, self).__init__(inputs, outputs)
        self.output_dtype = output_dtype
        self.dtype = dtype or output_dtype or "float32" # 通常为 float32
        self.axis = axis
        self.block_size = int(block_size or 0)
        self.version = version

    # 执行 `DequantizeLinear` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, x_scale, x_zero_point=None) -> Tensor:
        scale_tensor = x_scale
        zp_tensor = x_zero_point
        if zp_tensor is None:
            zp_tensor = Tensor(1, dtype=x.dtype, data=np.zeros((1,), dtype=nn.DTYPE_TO_NUMPY[x.dtype]))
        if self.block_size > 0:
            scale_tensor = _expand_blocked_qdq_param(scale_tensor, list(x.data.shape), self.axis, self.block_size)
            zp_tensor = _expand_blocked_qdq_param(zp_tensor, list(x.data.shape), self.axis, self.block_size)
        if x_scale.data.ndim == 1 and x.data.ndim > 1:
            new_shape = [1] * x.data.ndim
            safe_axis = self.axis if self.axis >= 0 else self.axis + x.data.ndim
            if safe_axis < 0 or safe_axis >= x.data.ndim:
                raise ValueError(f"DequantizeLinear axis {self.axis} is out of bounds for rank {x.data.ndim}")
            new_shape[safe_axis] = x_scale.data.size
            scale_tensor = Tensor(*new_shape, dtype=x_scale.dtype, data=x_scale.data.reshape(new_shape))
            if zp_tensor.data.size == x_scale.data.size:
                zp_tensor = Tensor(*new_shape, dtype=zp_tensor.dtype, data=zp_tensor.data.reshape(new_shape))
        if self.lib is None:
            x_bc, scale_bc, zp_bc = np.broadcast_arrays(x.data, _tensor_data_as_numeric(scale_tensor), zp_tensor.data)
            out_data = (x_bc.astype(np.float64) - zp_bc.astype(np.float64)) * scale_bc.astype(np.float64)
            out_data = _cast_numeric_to_dtype(out_data, self.dtype)
            out_tensor = Tensor(*out_data.shape, dtype=self.dtype, data=out_data)
            values = {"tensor": out_tensor, "parameters": None, "graph": None}
            self.parameters = {"values": values}
            return values
        out_tensor = self._execute_ternary(x, scale_tensor, zp_tensor, "dequantize_linear_forward")
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # 执行 `DequantizeLinear` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, x_scale, x_zero_point=None) -> Tensor_:
        output_tensor = Tensor_(*x.size, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values
