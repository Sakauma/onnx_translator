"""ONNX operator runtime and shape-inference implementations.

Egor Izmaylov: This module is intentionally the narrow Python control layer
between imported ONNX nodes and the C backend in ``tensor_ops``.  Each operator
should keep expensive numeric work on the C path whenever the dtype is supported;
Python code is used for ONNX attribute normalization, dynamic shape reasoning,
sequence/optional/control-flow orchestration, and fallback behavior for payloads
such as strings that cannot be represented by the C tensor ABI.
"""

from nn import Ops
from nn import Tensor, Tensor_, DTYPE_MAP, CTensor
import nn
import ctypes
import numpy as np
from typing import List, Union
import os
import unicodedata

class CConvParams(ctypes.Structure):
    # Egor Izmaylov: ctypes mirrors the C structs exactly; field order is part of
    # the ABI contract with tensor_ops.c, so keep these definitions synchronized
    # with tensor_ops/tensor_ops.h whenever convolution parameters change.
    _fields_ = [
        ("pads", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("dilations", ctypes.POINTER(ctypes.c_int)),
        ("group", ctypes.c_int)
    ]

class CPoolParams(ctypes.Structure):
    # Egor Izmaylov: pooling operators pass all spatial metadata as raw int
    # arrays to avoid per-element Python callbacks inside the C implementation.
    _fields_ = [
        ("pads", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("dilations", ctypes.POINTER(ctypes.c_int)),
        ("kernel_shape", ctypes.POINTER(ctypes.c_int))
    ]
    
class CReduceParams(ctypes.Structure):
    # Egor Izmaylov: reductions share one C parameter block because ONNX reduce
    # ops differ mainly by accumulation rule while axes/keepdims semantics match.
    _fields_ = [
        ("axes", ctypes.POINTER(ctypes.c_int)),
        ("num_axes", ctypes.c_int),
        ("keepdims", ctypes.c_int)
    ]

class RELU(Ops):
    """ReLU激活函数操作类"""
    
    # Egor Izmaylov: Function `RELU.__init__` initializes RELU, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        """
        初始化ReLU操作
        
        Args:
            inputs: 输入节点列表
            outputs: 输出节点列表
            dtype: 数据类型
            version: 操作版本号
        """
        super(RELU, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `RELU.forward` executes the concrete runtime path for RELU, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input: Tensor) -> Tensor:
        """
        ReLU函数的C后端实现，使用真实数据进行计算
        
        Args:
            input: 输入张量
            
        Returns:
            Tensor: 经过ReLU激活后的输出张量
        """
        out_tensor = self._execute_unary(input, "relu_forward")# 调用通用的一元执行模板
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # Egor Izmaylov: Function `RELU.forward_` performs shape-only inference for RELU, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> Tensor_:
        """
        ReLU函数的Python实现，不使用真实数据进行计算
        
        Args:
            input: 输入张量占位符
            
        Returns:
            Tensor_: 输出张量占位符
        """
        #output_tensor = input
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

class COS(Ops):
    """余弦函数操作类"""
    
    # Egor Izmaylov: Function `COS.__init__` initializes COS, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        """
        初始化COS操作
        
        Args:
            inputs: 输入节点列表
            outputs: 输出节点列表
            dtype: 数据类型
            version: 操作版本号
        """
        super(COS, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `COS.forward` executes the concrete runtime path for COS, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input: Tensor) -> Tensor:
        """
        余弦函数的C后端实现，使用真实数据进行计算
        
        Args:
            input: 输入张量
            
        Returns:
            Tensor: 经过余弦函数计算后的输出张量
        """
        out_tensor = self._execute_unary(input, "cos_forward")
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # Egor Izmaylov: Function `COS.forward_` performs shape-only inference for COS, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> Tensor_:
        """
        余弦函数的Python实现，不使用真实数据进行计算
        
        Args:
            input: 输入张量占位符
            
        Returns:
            Tensor_: 输出张量占位符
        """
        #output_tensor = input
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

class ABS(Ops):
    """Abs激活函数操作类"""
    
    # Egor Izmaylov: Function `ABS.__init__` initializes ABS, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        """
        初始化ABS操作
        
        Args:
            inputs: 输入节点列表
            outputs: 输出节点列表
            dtype: 数据类型
            version: 操作版本号
        """
        super(ABS, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `ABS.forward` executes the concrete runtime path for ABS, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input: Tensor) -> Tensor:
        """
        Abs函数的C后端实现，使用真实数据进行计算
        """
        out_tensor = self._execute_unary(input, "abs_forward")
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # Egor Izmaylov: Function `ABS.forward_` performs shape-only inference for ABS, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> Tensor_:
        """
        Abs函数的Python实现，不使用真实数据进行计算
        """
        #output_tensor = input
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values
    
class ADD(Ops):
    """加法操作类 (A + B)，支持广播和混合精度"""

    # Egor Izmaylov: Function `ADD.__init__` initializes ADD, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        """
        初始化ADD操作
        
        Args:
            inputs: 输入节点列表 (应有2个)
            outputs: 输出节点列表 (应有1个)
            dtype: 预期的输出数据类型 (来自ONNX)
            version: 操作版本号
        """
        super(ADD, self).__init__(inputs, outputs)
        self.dtype = dtype # 这是ONNX图推断的输出类型
        self.version = version

    # Egor Izmaylov: Function `ADD.forward` executes the concrete runtime path for ADD, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, *inputs) -> Tensor:
        """
        加法函数的C后端实现，在Python层处理广播
        """
        if len(inputs) != 2:
            raise ValueError(f"ADD operator expects 2 inputs, but got {len(inputs)}")
        
        out_tensor = self._execute_binary(inputs[0], inputs[1], "add_forward")
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # Egor Izmaylov: Function `ADD.forward_` performs shape-only inference for ADD, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, *inputs) -> Tensor_:
        """
        Add函数的Python实现，不使用真实数据进行计算 (用于图推断)
        """
        if len(inputs) != 2:
            raise ValueError(f"ADD operator expects 2 inputs (Tensor_), but got {len(inputs)}")
        
        a = inputs[0]
        b = inputs[1]

        # 1. 计算广播后的形状 (不计算数据)
        temp_a = np.empty(a.size, dtype=np.uint8) # 使用uint8节省内存
        temp_b = np.empty(b.size, dtype=np.uint8)
        try:
            output_shape = np.broadcast(temp_a, temp_b).shape
        except ValueError as e:
            print(f"Error during broadcasting shapes {a.size} and {b.size}")
            raise e

        # 2. 计算类型提升
        dtype_a = nn.DTYPE_TO_NUMPY[a.dtype]
        dtype_b = nn.DTYPE_TO_NUMPY[b.dtype]
        output_dtype_np = np.result_type(dtype_a, dtype_b)
        
        if output_dtype_np.type in nn.NUMPY_TO_DTYPE:
             output_dtype_str = nn.NUMPY_TO_DTYPE[output_dtype_np.type]
        elif 'float' in str(output_dtype_np):
             output_dtype_str = "float64"
        elif 'int' in str(output_dtype_np):
             output_dtype_str = "int64"
        else:
             output_dtype_str = "float32" # 最终备用
        
        # 3. 创建输出的 Tensor_
        output_tensor = Tensor_(*output_shape, dtype=output_dtype_str)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values
    
class SUB(Ops):
    """减法操作类 (A - B)，支持广播和混合精度"""

    # Egor Izmaylov: Function `SUB.__init__` initializes SUB, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(SUB, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `SUB.forward` executes the concrete runtime path for SUB, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, *inputs) -> Tensor:
        """
        减法函数的C后端实现，在Python层处理广播
        """
        if len(inputs) != 2:
            raise ValueError(f"SUB operator expects 2 inputs, but got {len(inputs)}")
        
        out_tensor = self._execute_binary(inputs[0], inputs[1], "sub_forward")
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values
        
    # Egor Izmaylov: Function `SUB.forward_` performs shape-only inference for SUB, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, *inputs) -> Tensor_:
        """
        Sub函数的Python实现，不使用真实数据进行计算 (用于图推断)
        """
        if len(inputs) != 2:
            raise ValueError(f"SUB operator expects 2 inputs (Tensor_), but got {len(inputs)}")
        
        a = inputs[0]
        b = inputs[1]

        # 1. 计算广播后的形状
        temp_a = np.empty(a.size, dtype=np.uint8)
        temp_b = np.empty(b.size, dtype=np.uint8)
        try:
            output_shape = np.broadcast(temp_a, temp_b).shape
        except ValueError as e:
            print(f"Error during broadcasting shapes {a.size} and {b.size}")
            raise e

        # 2. 计算类型提升
        dtype_a = nn.DTYPE_TO_NUMPY[a.dtype]
        dtype_b = nn.DTYPE_TO_NUMPY[b.dtype]
        output_dtype_np = np.result_type(dtype_a, dtype_b)
        
        if output_dtype_np.type in nn.NUMPY_TO_DTYPE:
             output_dtype_str = nn.NUMPY_TO_DTYPE[output_dtype_np.type]
        elif 'float' in str(output_dtype_np):
             output_dtype_str = "float64"
        elif 'int' in str(output_dtype_np):
             output_dtype_str = "int64"
        else:
             output_dtype_str = "float32"
        
        # 3. 创建输出的 Tensor_
        output_tensor = Tensor_(*output_shape, dtype=output_dtype_str)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

class MUL(Ops):
    """乘法操作类 (A * B)，支持广播和混合精度"""

    # Egor Izmaylov: Function `MUL.__init__` initializes MUL, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(MUL, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `MUL.forward` executes the concrete runtime path for MUL, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, *inputs) -> Tensor:
        """
        乘法函数的C后端实现，在Python层处理广播
        """
        if len(inputs) != 2:
            raise ValueError(f"MUL operator expects 2 inputs, but got {len(inputs)}")
        
        out_tensor = self._execute_binary(inputs[0], inputs[1], "mul_forward")
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # Egor Izmaylov: Function `MUL.forward_` performs shape-only inference for MUL, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, *inputs) -> Tensor_:
        """
        Mul函数的Python实现，不使用真实数据进行计算 (用于图推断)
        """
        if len(inputs) != 2:
            raise ValueError(f"MUL operator expects 2 inputs (Tensor_), but got {len(inputs)}")
        
        a = inputs[0]
        b = inputs[1]

        # 1. 计算广播后的形状
        temp_a = np.empty(a.size, dtype=np.uint8)
        temp_b = np.empty(b.size, dtype=np.uint8)
        try:
            output_shape = np.broadcast(temp_a, temp_b).shape
        except ValueError as e:
            print(f"Error during broadcasting shapes {a.size} and {b.size}")
            raise e

        # 2. 计算类型提升
        dtype_a = nn.DTYPE_TO_NUMPY[a.dtype]
        dtype_b = nn.DTYPE_TO_NUMPY[b.dtype]
        output_dtype_np = np.result_type(dtype_a, dtype_b)
        
        if output_dtype_np.type in nn.NUMPY_TO_DTYPE:
             output_dtype_str = nn.NUMPY_TO_DTYPE[output_dtype_np.type]
        elif 'float' in str(output_dtype_np):
             output_dtype_str = "float64"
        elif 'int' in str(output_dtype_np):
             output_dtype_str = "int64"
        else:
             output_dtype_str = "float32"
        
        # 3. 创建输出的 Tensor_
        output_tensor = Tensor_(*output_shape, dtype=output_dtype_str)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

class DIV(Ops):
    """除法操作类 (A / B)，支持广播和混合精度"""

    # Egor Izmaylov: Function `DIV.__init__` initializes DIV, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(DIV, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `DIV.forward` executes the concrete runtime path for DIV, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, *inputs) -> Tensor:
        """
        除法函数的C后端实现，在Python层处理广播
        """
        if len(inputs) != 2:
            raise ValueError(f"DIV operator expects 2 inputs, but got {len(inputs)}")
        
        out_tensor = self._execute_binary(inputs[0], inputs[1], "div_forward")
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # Egor Izmaylov: Function `DIV.forward_` performs shape-only inference for DIV, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, *inputs) -> Tensor_:
        """
        Div函数的Python实现，不使用真实数据进行计算 (用于图推断)
        """
        if len(inputs) != 2:
            raise ValueError(f"DIV operator expects 2 inputs (Tensor_), but got {len(inputs)}")
        
        a = inputs[0]
        b = inputs[1]

        # 1. 计算广播后的形状
        temp_a = np.empty(a.size, dtype=np.uint8)
        temp_b = np.empty(b.size, dtype=np.uint8)
        try:
            output_shape = np.broadcast(temp_a, temp_b).shape
        except ValueError as e:
            print(f"Error during broadcasting shapes {a.size} and {b.size}")
            raise e

        # 2. 计算类型提升
        dtype_a = nn.DTYPE_TO_NUMPY[a.dtype]
        dtype_b = nn.DTYPE_TO_NUMPY[b.dtype]
        output_dtype_np = np.result_type(dtype_a, dtype_b)
        
        if output_dtype_np.type in nn.NUMPY_TO_DTYPE:
             output_dtype_str = nn.NUMPY_TO_DTYPE[output_dtype_np.type]
        elif 'float' in str(output_dtype_np):
             output_dtype_str = "float64"
        elif 'int' in str(output_dtype_np):
             output_dtype_str = "int64"
        else:
             output_dtype_str = "float32"
        
        # 3. 创建输出的 Tensor_
        output_tensor = Tensor_(*output_shape, dtype=output_dtype_str)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values
    
class QuantizeLinear(Ops):
    # Egor Izmaylov: Function `QuantizeLinear.__init__` initializes QuantizeLinear, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axis=1, dtype=None, version="17"):
        super(QuantizeLinear, self).__init__(inputs, outputs)
        self.dtype = dtype or "uint8"
        self.axis = axis # 保存 axis
        self.version = version

    # Egor Izmaylov: Function `QuantizeLinear._default_zero_point` centralizes the default zero point helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _default_zero_point(self):
        return Tensor(1, dtype=self.dtype, data=np.zeros((1,), dtype=nn.DTYPE_TO_NUMPY[self.dtype]))

    # Egor Izmaylov: Function `QuantizeLinear.forward` executes the concrete runtime path for QuantizeLinear, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, y_scale, y_zero_point=None) -> Tensor:
        scale_tensor = y_scale
        zp_tensor = y_zero_point if y_zero_point is not None else self._default_zero_point()

        # 检查是否需要广播处理 (Scale 是 1D 但 Input 是 ND)
        if y_scale.data.ndim == 1 and x.data.ndim > 1:
            new_shape = [1] * x.data.ndim
            safe_axis = self.axis if self.axis >= 0 else self.axis + x.data.ndim
            if safe_axis < x.data.ndim:
                new_shape[safe_axis] = y_scale.data.size
            scale_tensor = Tensor(*new_shape, dtype=y_scale.dtype, data=y_scale.data.reshape(new_shape))
            if zp_tensor.data.size == y_scale.data.size:
                zp_tensor = Tensor(*new_shape, dtype=zp_tensor.dtype, data=zp_tensor.data.reshape(new_shape))

        out_tensor = self._execute_ternary(x, scale_tensor, zp_tensor, "quantize_linear_forward")
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # Egor Izmaylov: Function `QuantizeLinear.forward_` performs shape-only inference for QuantizeLinear, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, y_scale, y_zero_point=None) -> Tensor_:
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None, "graph": None}

class DequantizeLinear(Ops):
    # Egor Izmaylov: Function `DequantizeLinear.__init__` initializes DequantizeLinear, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype=None, axis=1, version="17"):
        super(DequantizeLinear, self).__init__(inputs, outputs)
        self.dtype = dtype or "float32" # 通常为 float32
        self.axis = axis
        self.version = version

    # Egor Izmaylov: Function `DequantizeLinear.forward` executes the concrete runtime path for DequantizeLinear, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, x_scale, x_zero_point=None) -> Tensor:
        scale_tensor = x_scale
        zp_tensor = x_zero_point
        if zp_tensor is None:
            zp_tensor = Tensor(1, dtype=x.dtype, data=np.zeros((1,), dtype=nn.DTYPE_TO_NUMPY[x.dtype]))
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
            x_bc, scale_bc, zp_bc = np.broadcast_arrays(x.data, scale_tensor.data, zp_tensor.data)
            out_data = (x_bc.astype(np.float64) - zp_bc.astype(np.float64)) * scale_bc.astype(np.float64)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, np.float32))
            out_tensor = Tensor(*out_data.shape, dtype=self.dtype, data=out_data)
            values = {"tensor": out_tensor, "parameters": None, "graph": None}
            self.parameters = {"values": values}
            return values
        out_tensor = self._execute_ternary(x, scale_tensor, zp_tensor, "dequantize_linear_forward")
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # Egor Izmaylov: Function `DequantizeLinear.forward_` performs shape-only inference for DequantizeLinear, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, x_scale, x_zero_point=None) -> Tensor_:
        output_tensor = Tensor_(*x.size, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values
    
# Egor Izmaylov: Function `_conv_attr` centralizes the conv attr helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _conv_attr(values, spatial_rank, default):
    if values is None:
        return [default] * spatial_rank
    values = list(values)
    if len(values) != spatial_rank:
        raise ValueError(f"Convolution attribute rank {len(values)} does not match spatial rank {spatial_rank}")
    return values


# Egor Izmaylov: Function `_conv_effective_kernel` centralizes the conv effective kernel helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _conv_effective_kernel(kernel_shape, dilations):
    return [dilations[i] * (kernel_shape[i] - 1) + 1 for i in range(len(kernel_shape))]


# Egor Izmaylov: Function `_conv_resolve_pads` centralizes the conv resolve pads helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _conv_resolve_pads(input_spatial, kernel_shape, pads, strides, dilations, auto_pad="NOTSET"):
    spatial_rank = len(input_spatial)
    auto_pad = auto_pad or "NOTSET"
    if auto_pad == "VALID":
        return [0] * (2 * spatial_rank)
    if auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
        effective = _conv_effective_kernel(kernel_shape, dilations)
        resolved = []
        end_pads = []
        for dim in range(spatial_rank):
            out_dim = int(np.ceil(float(input_spatial[dim]) / float(strides[dim])))
            total = max((out_dim - 1) * strides[dim] + effective[dim] - input_spatial[dim], 0)
            if auto_pad == "SAME_UPPER":
                begin = total // 2
            else:
                begin = total - total // 2
            resolved.append(begin)
            end_pads.append(total - begin)
        return resolved + end_pads
    if pads is None:
        return [0] * (2 * spatial_rank)
    pads = list(pads)
    if len(pads) != 2 * spatial_rank:
        raise ValueError(f"Convolution pads must contain {2 * spatial_rank} values")
    return pads


# Egor Izmaylov: Function `_conv_output_spatial` centralizes the conv output spatial helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _conv_output_spatial(input_spatial, kernel_shape, pads, strides, dilations):
    effective = _conv_effective_kernel(kernel_shape, dilations)
    spatial_rank = len(input_spatial)
    return tuple(
        (input_spatial[i] + pads[i] + pads[spatial_rank + i] - effective[i]) // strides[i] + 1
        for i in range(spatial_rank)
    )


# Egor Izmaylov: Function `_conv_nd_numpy` centralizes the conv nd numpy helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _conv_nd_numpy(x, w, bias=None, pads=None, strides=None, dilations=None, group=1, auto_pad="NOTSET", acc_dtype=np.float64):
    x = np.asarray(x)
    w = np.asarray(w)
    spatial_rank = x.ndim - 2
    kernel_shape = list(w.shape[2:])
    strides = _conv_attr(strides, spatial_rank, 1)
    dilations = _conv_attr(dilations, spatial_rank, 1)
    pads = _conv_resolve_pads(list(x.shape[2:]), kernel_shape, pads, strides, dilations, auto_pad)

    n_batches, in_channels = x.shape[:2]
    out_channels, channels_per_group = w.shape[:2]
    if group <= 0 or in_channels % group != 0 or out_channels % group != 0:
        raise ValueError(f"Invalid convolution group={group} for input channels={in_channels}, output channels={out_channels}")
    if channels_per_group != in_channels // group:
        raise ValueError(f"Weight channel dimension {channels_per_group} does not match input channels/group {in_channels // group}")

    out_spatial = _conv_output_spatial(list(x.shape[2:]), kernel_shape, pads, strides, dilations)
    pad_width = [(0, 0), (0, 0)] + [(pads[i], pads[spatial_rank + i]) for i in range(spatial_rank)]
    x_pad = np.pad(x.astype(acc_dtype, copy=False), pad_width, mode="constant")
    w_acc = w.astype(acc_dtype, copy=False)
    out = np.zeros((n_batches, out_channels) + out_spatial, dtype=acc_dtype)
    out_per_group = out_channels // group
    in_per_group = in_channels // group

    for n in range(n_batches):
        for g in range(group):
            for oc_local in range(out_per_group):
                oc = g * out_per_group + oc_local
                for out_index in np.ndindex(*out_spatial):
                    total = 0
                    for ic_local in range(in_per_group):
                        ic = g * in_per_group + ic_local
                        for kernel_index in np.ndindex(*kernel_shape):
                            in_index = tuple(out_index[d] * strides[d] + kernel_index[d] * dilations[d] for d in range(spatial_rank))
                            total += x_pad[(n, ic) + in_index] * w_acc[(oc, ic_local) + kernel_index]
                    out[(n, oc) + out_index] = total

    if bias is not None:
        bias_arr = np.asarray(bias, dtype=acc_dtype).reshape((1, out_channels) + (1,) * spatial_rank)
        out = out + bias_arr
    return out


# Egor Izmaylov: Function `_reshape_channel_param` centralizes the reshape channel param helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _reshape_channel_param(param, target, axis, dtype):
    if param is None:
        return np.array(0, dtype=dtype)
    arr = np.asarray(param.data, dtype=dtype)
    if arr.ndim == 0 or arr.size == 1:
        return arr.reshape(())
    if arr.ndim == 1 and arr.shape[0] == target.shape[axis]:
        shape = [1] * target.ndim
        shape[axis] = arr.shape[0]
        return arr.reshape(shape)
    return arr


# Egor Izmaylov: Function `_broadcast_conv_zero_point` centralizes the broadcast conv zero point helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _broadcast_conv_zero_point(param, target_shape, dtype, axis=None):
    np_dtype = nn.DTYPE_TO_NUMPY[dtype]
    if param is None:
        return np.zeros(target_shape, dtype=np_dtype)
    arr = np.asarray(param.data, dtype=np_dtype)
    if arr.shape == target_shape:
        return np.ascontiguousarray(arr)
    if arr.shape == () or arr.size == 1:
        return np.broadcast_to(arr.reshape(()), target_shape).copy()
    if axis is not None and arr.ndim == 1 and arr.shape[0] == target_shape[axis]:
        shape = [1] * len(target_shape)
        shape[axis] = arr.shape[0]
        return np.broadcast_to(arr.reshape(shape), target_shape).copy()
    return np.broadcast_to(arr, target_shape).copy()


# Egor Izmaylov: Function `_broadcast_conv_param` centralizes the broadcast conv param helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _broadcast_conv_param(param, target_shape, dtype, axis=None):
    np_dtype = nn.DTYPE_TO_NUMPY[dtype]
    arr = np.asarray(param.data, dtype=np_dtype)
    if arr.shape == target_shape:
        return np.ascontiguousarray(arr)
    if arr.shape == () or arr.size == 1:
        return np.broadcast_to(arr.reshape(()), target_shape).copy()
    if axis is not None and arr.ndim == 1 and arr.shape[0] == target_shape[axis]:
        shape = [1] * len(target_shape)
        shape[axis] = arr.shape[0]
        return np.broadcast_to(arr.reshape(shape), target_shape).copy()
    return np.broadcast_to(arr, target_shape).copy()


# Egor Izmaylov: Function `_reshape_output_channel_param` centralizes the reshape output channel param helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _reshape_output_channel_param(param, out_channels, spatial_rank, dtype):
    arr = np.asarray(param.data, dtype=dtype)
    if arr.ndim == 0 or arr.size == 1:
        return arr.reshape(())
    if arr.ndim == 1 and arr.shape[0] == out_channels:
        return arr.reshape((1, out_channels) + (1,) * spatial_rank)
    return arr


# Egor Izmaylov: Function `_dtype_bounds` centralizes the dtype bounds helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _dtype_bounds(dtype):
    np_dtype = nn.DTYPE_TO_NUMPY.get(dtype, np.uint8)
    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        return info.min, info.max
    return None, None

class Conv(Ops):
    # Egor Izmaylov: Function `Conv.__init__` initializes Conv, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, pads, strides, dilations, group, dtype, kernel_shape=None, auto_pad="NOTSET", version="17"):
        super(Conv, self).__init__(inputs, outputs)
        # 必须完整保存所有参数
        self.pads = list(pads) if pads is not None else None
        self.strides = list(strides) if strides is not None else None
        self.dilations = list(dilations) if dilations is not None else None
        self.group = group
        self.kernel_shape = list(kernel_shape) if kernel_shape is not None else None
        self.auto_pad = auto_pad
        self.dtype = dtype
        self.version = version

        # 注册 C 函数参数类型
        if self.lib:
            self.lib.conv2d_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), 
                ctypes.POINTER(CTensor), ctypes.POINTER(CConvParams)
            ]

    # Egor Izmaylov: Function `Conv._params` centralizes the params helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _params(self, x_shape, w_shape):
        spatial_rank = len(x_shape) - 2
        kernel_shape = self.kernel_shape if self.kernel_shape is not None else list(w_shape[2:])
        if list(kernel_shape) != list(w_shape[2:]):
            raise ValueError(f"Conv kernel_shape {kernel_shape} does not match weight spatial shape {w_shape[2:]}")
        strides = _conv_attr(self.strides, spatial_rank, 1)
        dilations = _conv_attr(self.dilations, spatial_rank, 1)
        pads = _conv_resolve_pads(list(x_shape[2:]), kernel_shape, self.pads, strides, dilations, self.auto_pad)
        out_spatial = _conv_output_spatial(list(x_shape[2:]), kernel_shape, pads, strides, dilations)
        return kernel_shape, strides, dilations, pads, tuple(x_shape[:1]) + tuple(w_shape[:1]) + out_spatial

    # Egor Izmaylov: Function `Conv.forward` executes the concrete runtime path for Conv, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x: Tensor, w: Tensor, b: Tensor = None) -> dict:
        _, strides, dilations, pads, out_shape = self._params(x.size, w.size)

        if self.lib is None or len(x.size) != 4:
            out_data = _conv_nd_numpy(
                x.data,
                w.data,
                None if b is None else b.data,
                pads=pads,
                strides=strides,
                dilations=dilations,
                group=self.group,
                auto_pad="NOTSET",
                acc_dtype=np.float64,
            )
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
            out_tensor = Tensor(*out_data.shape, dtype=self.dtype, data=out_data)
            values = {"tensor": out_tensor, "parameters": None, "graph": None}
            self.parameters = {"values": values}
            return values
        
        # 2. 准备 C 参数
        pads_arr = (ctypes.c_int * 4)(*pads)
        strides_arr = (ctypes.c_int * 2)(*strides)
        dilations_arr = (ctypes.c_int * 2)(*dilations)
        
        c_params = CConvParams()
        c_params.pads = ctypes.cast(pads_arr, ctypes.POINTER(ctypes.c_int))
        c_params.strides = ctypes.cast(strides_arr, ctypes.POINTER(ctypes.c_int))
        c_params.dilations = ctypes.cast(dilations_arr, ctypes.POINTER(ctypes.c_int))
        c_params.group = self.group

        # 3. 准备 Tensor
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        w_c = self._numpy_to_ctensor(w.data, w.dtype)
        b_c = self._numpy_to_ctensor(b.data, b.dtype) if b is not None else ctypes.POINTER(CTensor)()
        
        # 创建输出 Tensor
        output_shape_c = (ctypes.c_int * 4)(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, 4, DTYPE_MAP[self.dtype])
        
        # 4. 执行计算
        self.lib.conv2d_forward(x_c, w_c, b_c, output_c, ctypes.byref(c_params))
        
        # 5. 回收与返回
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(x_c)
        self.lib.free_tensor(w_c)
        self.lib.free_tensor(output_c)
        if b is not None: self.lib.free_tensor(b_c)

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # Egor Izmaylov: Function `Conv.forward_` performs shape-only inference for Conv, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x: Tensor_, w: Tensor_, b: Tensor_ = None) -> dict:
        # 仅做形状推断
        _, _, _, _, out_shape = self._params(x.size, w.size)
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

class ConvTranspose(Ops):
    # Egor Izmaylov: Function `ConvTranspose.__init__` initializes ConvTranspose, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(
        self,
        inputs,
        outputs,
        pads=None,
        strides=None,
        dilations=None,
        group=1,
        kernel_shape=None,
        output_padding=None,
        output_shape=None,
        auto_pad="NOTSET",
        dtype="float32",
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.pads = list(pads) if pads is not None else None
        self.strides = list(strides) if strides is not None else None
        self.dilations = list(dilations) if dilations is not None else None
        self.group = group
        self.kernel_shape = list(kernel_shape) if kernel_shape is not None else None
        self.output_padding = list(output_padding) if output_padding is not None else None
        self.output_shape = tuple(output_shape) if output_shape is not None else None
        self.auto_pad = auto_pad
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.conv_transpose2d_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CConvParams)
            ]

    # Egor Izmaylov: Function `ConvTranspose._params` centralizes the params helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _params(self, x_shape, w_shape):
        spatial_rank = len(x_shape) - 2
        kernel_shape = self.kernel_shape if self.kernel_shape is not None else list(w_shape[2:])
        strides = _conv_attr(self.strides, spatial_rank, 1)
        dilations = _conv_attr(self.dilations, spatial_rank, 1)
        output_padding = _conv_attr(self.output_padding, spatial_rank, 0)
        pads = [0] * (2 * spatial_rank) if self.pads is None else list(self.pads)
        if len(pads) != 2 * spatial_rank:
            raise ValueError(f"ConvTranspose pads must contain {2 * spatial_rank} values")
        effective = _conv_effective_kernel(kernel_shape, dilations)
        if self.output_shape is not None:
            out_spatial = tuple(self.output_shape)
            if len(out_spatial) != spatial_rank:
                raise ValueError(f"ConvTranspose output_shape rank {len(out_spatial)} does not match spatial rank {spatial_rank}")
            if self.pads is None or self.auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
                begin_pads, end_pads = [], []
                for dim in range(spatial_rank):
                    total = strides[dim] * (x_shape[dim + 2] - 1) + output_padding[dim] + effective[dim] - out_spatial[dim]
                    if self.auto_pad == "SAME_LOWER":
                        begin = total - total // 2
                    else:
                        begin = total // 2
                    begin_pads.append(begin)
                    end_pads.append(total - begin)
                pads = begin_pads + end_pads
        elif self.auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
            out_spatial = tuple(x_shape[dim + 2] * strides[dim] for dim in range(spatial_rank))
            begin_pads, end_pads = [], []
            for dim in range(spatial_rank):
                total = strides[dim] * (x_shape[dim + 2] - 1) + output_padding[dim] + effective[dim] - out_spatial[dim]
                if self.auto_pad == "SAME_LOWER":
                    begin = total - total // 2
                else:
                    begin = total // 2
                begin_pads.append(begin)
                end_pads.append(total - begin)
            pads = begin_pads + end_pads
        elif self.auto_pad == "VALID":
            pads = [0] * (2 * spatial_rank)
            out_spatial = tuple(
                strides[dim] * (x_shape[dim + 2] - 1)
                + output_padding[dim]
                + effective[dim]
                for dim in range(spatial_rank)
            )
        else:
            out_spatial = tuple(
                strides[dim] * (x_shape[dim + 2] - 1)
                + output_padding[dim]
                + effective[dim]
                - pads[dim]
                - pads[spatial_rank + dim]
                for dim in range(spatial_rank)
            )
        return kernel_shape, pads, strides, dilations, output_padding, out_spatial

    # Egor Izmaylov: Function `ConvTranspose.forward` executes the concrete runtime path for ConvTranspose, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, w, b=None):
        x_data = np.asarray(x.data, dtype=np.float64)
        w_data = np.asarray(w.data, dtype=np.float64)
        n_batches, in_channels = x_data.shape[:2]
        if w_data.shape[0] != in_channels:
            raise ValueError(f"ConvTranspose weight input channels {w_data.shape[0]} != input channels {in_channels}")
        if self.group <= 0 or in_channels % self.group != 0:
            raise ValueError(f"Invalid ConvTranspose group={self.group} for input channels={in_channels}")
        m_per_group = w_data.shape[1]
        out_channels = m_per_group * self.group
        in_per_group = in_channels // self.group
        kernel_shape, pads, strides, dilations, _output_padding, out_spatial = self._params(x_data.shape, w_data.shape)
        out_shape = (n_batches, out_channels) + out_spatial
        if (
            self.lib is not None
            and x.data.ndim == 4
            and w.data.ndim == 4
            and x.dtype in nn.DTYPE_MAP
            and w.dtype in nn.DTYPE_MAP
            and (b is None or b.dtype in nn.DTYPE_MAP)
        ):
            pads_c = (ctypes.c_int * len(pads))(*pads)
            strides_c = (ctypes.c_int * len(strides))(*strides)
            dilations_c = (ctypes.c_int * len(dilations))(*dilations)
            c_params = CConvParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.group = self.group

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data.astype(nn.DTYPE_TO_NUMPY[x.dtype], copy=False)), x.dtype)
            w_c = self._numpy_to_ctensor(np.ascontiguousarray(w.data.astype(nn.DTYPE_TO_NUMPY[w.dtype], copy=False)), w.dtype)
            b_c = self._numpy_to_ctensor(np.ascontiguousarray(b.data), b.dtype) if b is not None else ctypes.POINTER(CTensor)()
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            self.lib.conv_transpose2d_forward(x_c, w_c, b_c, out_c, ctypes.byref(c_params))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(w_c)
            self.lib.free_tensor(out_c)
            if b is not None:
                self.lib.free_tensor(b_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        out = np.zeros((n_batches, out_channels) + out_spatial, dtype=np.float64)
        spatial_rank = len(out_spatial)

        for n in range(n_batches):
            for ic in range(in_channels):
                group_idx = ic // in_per_group
                for in_index in np.ndindex(*x_data.shape[2:]):
                    x_value = x_data[(n, ic) + in_index]
                    for oc_local in range(m_per_group):
                        oc = group_idx * m_per_group + oc_local
                        for kernel_index in np.ndindex(*kernel_shape):
                            out_index = tuple(
                                in_index[dim] * strides[dim] + kernel_index[dim] * dilations[dim] - pads[dim]
                                for dim in range(spatial_rank)
                            )
                            if all(0 <= out_index[dim] < out_spatial[dim] for dim in range(spatial_rank)):
                                out[(n, oc) + out_index] += x_value * w_data[(ic, oc_local) + kernel_index]

        if b is not None:
            out += np.asarray(b.data, dtype=np.float64).reshape((1, out_channels) + (1,) * spatial_rank)
        out_data = out.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, np.float32), copy=False)
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `ConvTranspose.forward_` performs shape-only inference for ConvTranspose, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, w, b=None):
        _kernel_shape, _pads, _strides, _dilations, _output_padding, out_spatial = self._params(x.size, w.size)
        out_channels = w.size[1] * self.group
        return {"tensor": Tensor_(x.size[0], out_channels, *out_spatial, dtype=self.dtype), "parameters": None}

class ConvInteger(Ops):
    # Egor Izmaylov: Function `ConvInteger.__init__` initializes ConvInteger, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(
        self,
        inputs,
        outputs,
        pads=None,
        strides=None,
        dilations=None,
        group=1,
        kernel_shape=None,
        auto_pad="NOTSET",
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.pads = list(pads) if pads is not None else None
        self.strides = list(strides) if strides is not None else None
        self.dilations = list(dilations) if dilations is not None else None
        self.group = group
        self.kernel_shape = list(kernel_shape) if kernel_shape is not None else None
        self.auto_pad = auto_pad
        self.dtype = "int32"
        self.version = version
        if self.lib:
            self.lib.conv_integer_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CConvParams)
            ]

    # Egor Izmaylov: Function `ConvInteger._shape` centralizes the shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _shape(self, x_shape, w_shape):
        spatial_rank = len(x_shape) - 2
        kernel_shape = self.kernel_shape if self.kernel_shape is not None else list(w_shape[2:])
        strides = _conv_attr(self.strides, spatial_rank, 1)
        dilations = _conv_attr(self.dilations, spatial_rank, 1)
        pads = _conv_resolve_pads(list(x_shape[2:]), kernel_shape, self.pads, strides, dilations, self.auto_pad)
        return (x_shape[0], w_shape[0]) + _conv_output_spatial(list(x_shape[2:]), kernel_shape, pads, strides, dilations)

    # Egor Izmaylov: Function `ConvInteger.forward` executes the concrete runtime path for ConvInteger, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, w, x_zero_point=None, w_zero_point=None):
        spatial_rank = x.data.ndim - 2
        if (
            self.lib is not None
            and spatial_rank == 2
            and x.dtype in nn.DTYPE_MAP
            and w.dtype in nn.DTYPE_MAP
            and (x_zero_point is None or x_zero_point.dtype in nn.DTYPE_MAP)
            and (w_zero_point is None or w_zero_point.dtype in nn.DTYPE_MAP)
        ):
            kernel_shape = self.kernel_shape if self.kernel_shape is not None else list(w.size[2:])
            strides = _conv_attr(self.strides, spatial_rank, 1)
            dilations = _conv_attr(self.dilations, spatial_rank, 1)
            pads = _conv_resolve_pads(list(x.size[2:]), kernel_shape, self.pads, strides, dilations, self.auto_pad)
            out_shape = (x.size[0], w.size[0]) + _conv_output_spatial(list(x.size[2:]), kernel_shape, pads, strides, dilations)

            x_zp = _broadcast_conv_zero_point(x_zero_point, x.data.shape, x.dtype)
            w_zp = _broadcast_conv_zero_point(w_zero_point, w.data.shape, w.dtype, axis=0)

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data.astype(nn.DTYPE_TO_NUMPY[x.dtype], copy=False)), x.dtype)
            w_c = self._numpy_to_ctensor(np.ascontiguousarray(w.data.astype(nn.DTYPE_TO_NUMPY[w.dtype], copy=False)), w.dtype)
            x_zp_c = self._numpy_to_ctensor(x_zp, x.dtype)
            w_zp_c = self._numpy_to_ctensor(w_zp, w.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            pads_c = (ctypes.c_int * len(pads))(*pads)
            strides_c = (ctypes.c_int * len(strides))(*strides)
            dilations_c = (ctypes.c_int * len(dilations))(*dilations)
            c_params = CConvParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.group = self.group

            self.lib.conv_integer_forward(x_c, w_c, x_zp_c, w_zp_c, out_c, ctypes.byref(c_params))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            for c_tensor in (x_c, w_c, x_zp_c, w_zp_c, out_c):
                self.lib.free_tensor(c_tensor)
            return {"tensor": Tensor(*out_shape, dtype="int32", data=out_data), "parameters": None}

        x_i = x.data.astype(np.int32) - _reshape_channel_param(x_zero_point, x.data, 1, np.int32)
        w_i = w.data.astype(np.int32) - _reshape_channel_param(w_zero_point, w.data, 0, np.int32)
        strides = _conv_attr(self.strides, spatial_rank, 1)
        dilations = _conv_attr(self.dilations, spatial_rank, 1)
        out = _conv_nd_numpy(
            x_i,
            w_i,
            pads=self.pads,
            strides=strides,
            dilations=dilations,
            group=self.group,
            auto_pad=self.auto_pad,
            acc_dtype=np.int64,
        ).astype(np.int32)
        return {"tensor": Tensor(*out.shape, dtype="int32", data=out), "parameters": None}

    # Egor Izmaylov: Function `ConvInteger.forward_` performs shape-only inference for ConvInteger, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, w, x_zero_point=None, w_zero_point=None):
        return {"tensor": Tensor_(*self._shape(x.size, w.size), dtype="int32"), "parameters": None}

class QLinearConv(Ops):
    # Egor Izmaylov: Function `QLinearConv.__init__` initializes QLinearConv, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(
        self,
        inputs,
        outputs,
        pads=None,
        strides=None,
        dilations=None,
        group=1,
        kernel_shape=None,
        auto_pad="NOTSET",
        dtype="uint8",
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.pads = list(pads) if pads is not None else None
        self.strides = list(strides) if strides is not None else None
        self.dilations = list(dilations) if dilations is not None else None
        self.group = group
        self.kernel_shape = list(kernel_shape) if kernel_shape is not None else None
        self.auto_pad = auto_pad
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.qlinear_conv_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CConvParams)
            ]

    # Egor Izmaylov: Function `QLinearConv._shape` centralizes the shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _shape(self, x_shape, w_shape):
        spatial_rank = len(x_shape) - 2
        kernel_shape = self.kernel_shape if self.kernel_shape is not None else list(w_shape[2:])
        strides = _conv_attr(self.strides, spatial_rank, 1)
        dilations = _conv_attr(self.dilations, spatial_rank, 1)
        pads = _conv_resolve_pads(list(x_shape[2:]), kernel_shape, self.pads, strides, dilations, self.auto_pad)
        return (x_shape[0], w_shape[0]) + _conv_output_spatial(list(x_shape[2:]), kernel_shape, pads, strides, dilations)

    # Egor Izmaylov: Function `QLinearConv.forward` executes the concrete runtime path for QLinearConv, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, b=None):
        spatial_rank = x.data.ndim - 2
        out_channels = w.data.shape[0]
        if (
            self.lib is not None
            and spatial_rank == 2
            and x.dtype in nn.DTYPE_MAP
            and w.dtype in nn.DTYPE_MAP
            and x_scale.dtype in nn.DTYPE_MAP
            and w_scale.dtype in nn.DTYPE_MAP
            and y_scale.dtype in nn.DTYPE_MAP
            and y_zero_point.dtype in nn.DTYPE_MAP
            and (x_zero_point is None or x_zero_point.dtype in nn.DTYPE_MAP)
            and (w_zero_point is None or w_zero_point.dtype in nn.DTYPE_MAP)
            and (b is None or b.dtype in nn.DTYPE_MAP)
        ):
            kernel_shape = self.kernel_shape if self.kernel_shape is not None else list(w.size[2:])
            strides = _conv_attr(self.strides, spatial_rank, 1)
            dilations = _conv_attr(self.dilations, spatial_rank, 1)
            pads = _conv_resolve_pads(list(x.size[2:]), kernel_shape, self.pads, strides, dilations, self.auto_pad)
            out_shape = (x.size[0], out_channels) + _conv_output_spatial(list(x.size[2:]), kernel_shape, pads, strides, dilations)

            x_zp = _broadcast_conv_zero_point(x_zero_point, x.data.shape, x.dtype)
            w_zp = _broadcast_conv_zero_point(w_zero_point, w.data.shape, w.dtype, axis=0)
            x_s = _broadcast_conv_param(x_scale, x.data.shape, x_scale.dtype)
            w_s = _broadcast_conv_param(w_scale, w.data.shape, w_scale.dtype, axis=0)
            y_s = _broadcast_conv_param(y_scale, out_shape, y_scale.dtype)
            y_zp = _broadcast_conv_param(y_zero_point, out_shape, self.dtype)

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data.astype(nn.DTYPE_TO_NUMPY[x.dtype], copy=False)), x.dtype)
            x_s_c = self._numpy_to_ctensor(x_s, x_scale.dtype)
            x_zp_c = self._numpy_to_ctensor(x_zp, x.dtype)
            w_c = self._numpy_to_ctensor(np.ascontiguousarray(w.data.astype(nn.DTYPE_TO_NUMPY[w.dtype], copy=False)), w.dtype)
            w_s_c = self._numpy_to_ctensor(w_s, w_scale.dtype)
            w_zp_c = self._numpy_to_ctensor(w_zp, w.dtype)
            y_s_c = self._numpy_to_ctensor(y_s, y_scale.dtype)
            y_zp_c = self._numpy_to_ctensor(y_zp, self.dtype)
            b_c = self._numpy_to_ctensor(np.ascontiguousarray(b.data), b.dtype) if b is not None else ctypes.POINTER(CTensor)()
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            pads_c = (ctypes.c_int * len(pads))(*pads)
            strides_c = (ctypes.c_int * len(strides))(*strides)
            dilations_c = (ctypes.c_int * len(dilations))(*dilations)
            c_params = CConvParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.group = self.group

            self.lib.qlinear_conv_forward(
                x_c, x_s_c, x_zp_c,
                w_c, w_s_c, w_zp_c,
                y_s_c, y_zp_c, b_c,
                out_c, ctypes.byref(c_params)
            )
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            for c_tensor in (x_c, x_s_c, x_zp_c, w_c, w_s_c, w_zp_c, y_s_c, y_zp_c, out_c):
                self.lib.free_tensor(c_tensor)
            if b is not None:
                self.lib.free_tensor(b_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        x_zp = _reshape_channel_param(x_zero_point, x.data, 1, np.int32)
        w_zp = _reshape_channel_param(w_zero_point, w.data, 0, np.int32)
        x_s = _reshape_channel_param(x_scale, x.data, 1, np.float64)
        w_s = _reshape_channel_param(w_scale, w.data, 0, np.float64)
        x_real = (x.data.astype(np.int32) - x_zp).astype(np.float64) * x_s
        w_real = (w.data.astype(np.int32) - w_zp).astype(np.float64) * w_s

        bias = None
        if b is not None:
            raw_x_scale = np.asarray(x_scale.data, dtype=np.float64)
            raw_w_scale = np.asarray(w_scale.data, dtype=np.float64)
            if raw_w_scale.ndim == 0 or raw_w_scale.size == 1:
                bias_scale = raw_x_scale.reshape(-1)[0] * raw_w_scale.reshape(-1)[0]
            else:
                bias_scale = raw_x_scale.reshape(-1)[0] * raw_w_scale.reshape(-1)
            bias = b.data.astype(np.float64) * bias_scale

        strides = _conv_attr(self.strides, spatial_rank, 1)
        dilations = _conv_attr(self.dilations, spatial_rank, 1)
        conv = _conv_nd_numpy(
            x_real,
            w_real,
            bias=bias,
            pads=self.pads,
            strides=strides,
            dilations=dilations,
            group=self.group,
            auto_pad=self.auto_pad,
            acc_dtype=np.float64,
        )
        y_s = _reshape_output_channel_param(y_scale, out_channels, spatial_rank, np.float64)
        y_zp = _reshape_output_channel_param(y_zero_point, out_channels, spatial_rank, np.float64)
        quantized = np.rint(conv / y_s + y_zp)
        low, high = _dtype_bounds(self.dtype)
        if low is not None:
            quantized = np.clip(quantized, low, high)
        out = quantized.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, np.uint8))
        return {"tensor": Tensor(*out.shape, dtype=self.dtype, data=out), "parameters": None}

    # Egor Izmaylov: Function `QLinearConv.forward_` performs shape-only inference for QLinearConv, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point, b=None):
        return {"tensor": Tensor_(*self._shape(x.size, w.size), dtype=self.dtype), "parameters": None}

# Egor Izmaylov: Function `_normalize_pool_params` centralizes the normalize pool params helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _normalize_pool_params(input_shape, kernel_shape, pads, strides, dilations, auto_pad="NOTSET"):
    spatial_rank = len(input_shape) - 2
    if spatial_rank < 1:
        raise ValueError("Pool operators expect input rank >= 3")
    if len(kernel_shape) != spatial_rank:
        raise ValueError(f"kernel_shape rank {len(kernel_shape)} does not match input spatial rank {spatial_rank}")

    strides = list(strides) if strides else [1] * spatial_rank
    dilations = list(dilations) if dilations else [1] * spatial_rank
    pads = _conv_resolve_pads(list(input_shape[2:]), list(kernel_shape), pads, strides, dilations, auto_pad)
    if len(pads) != 2 * spatial_rank:
        raise ValueError(f"pads must contain {2 * spatial_rank} values for spatial rank {spatial_rank}")
    if len(strides) != spatial_rank:
        raise ValueError(f"strides must contain {spatial_rank} values")
    if len(dilations) != spatial_rank:
        raise ValueError(f"dilations must contain {spatial_rank} values")
    return spatial_rank, pads, strides, dilations


# Egor Izmaylov: Function `_pool_output_shape` centralizes the pool output shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _pool_output_shape(input_shape, kernel_shape, pads, strides, dilations, ceil_mode=0, auto_pad="NOTSET"):
    spatial_rank, pads, strides, dilations = _normalize_pool_params(input_shape, kernel_shape, pads, strides, dilations, auto_pad)
    out_spatial = []
    for axis in range(spatial_rank):
        input_dim = input_shape[axis + 2]
        kernel_extent = dilations[axis] * (kernel_shape[axis] - 1) + 1
        numerator = input_dim + pads[axis] + pads[axis + spatial_rank] - kernel_extent
        if ceil_mode:
            out_dim = int(np.ceil(numerator / strides[axis])) + 1
            if (out_dim - 1) * strides[axis] >= input_dim + pads[axis]:
                out_dim -= 1
        else:
            out_dim = numerator // strides[axis] + 1
        out_spatial.append(max(0, int(out_dim)))
    return tuple(input_shape[:2]) + tuple(out_spatial)


# Egor Izmaylov: Function `_pool_window_slices` centralizes the pool window slices helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _pool_window_slices(out_index, kernel_shape, pads, strides, dilations):
    spatial_rank = len(kernel_shape)
    slices = []
    for axis in range(spatial_rank):
        start = out_index[axis] * strides[axis]
        stop = start + dilations[axis] * (kernel_shape[axis] - 1) + 1
        slices.append(slice(start, stop, dilations[axis]))
    return tuple(slices)


# Egor Izmaylov: Function `_pool_flat_index` centralizes the pool flat index helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _pool_flat_index(input_shape, n, c, spatial_coords, storage_order=0):
    coords = (n, c, *spatial_coords)
    order = "F" if storage_order else "C"
    return int(np.ravel_multi_index(coords, input_shape, order=order))


# Egor Izmaylov: Function `_max_pool_nd` centralizes the max pool nd helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _max_pool_nd(data, kernel_shape, pads, strides, dilations, ceil_mode=0, storage_order=0, auto_pad="NOTSET"):
    spatial_rank, pads, strides, dilations = _normalize_pool_params(data.shape, kernel_shape, pads, strides, dilations, auto_pad)
    out_shape = _pool_output_shape(data.shape, kernel_shape, pads, strides, dilations, ceil_mode)
    pad_width = [(0, 0), (0, 0)] + [(pads[i], pads[i + spatial_rank]) for i in range(spatial_rank)]
    work = np.pad(data, pad_width, mode="constant", constant_values=-np.inf)
    out = np.empty(out_shape, dtype=data.dtype)
    indices = np.zeros(out_shape, dtype=np.int64)

    for prefix in np.ndindex(data.shape[0], data.shape[1]):
        for out_spatial in np.ndindex(*out_shape[2:]):
            window = work[prefix + _pool_window_slices(out_spatial, kernel_shape, pads, strides, dilations)]
            flat = int(np.argmax(window))
            local = np.unravel_index(flat, window.shape)
            value = window[local]
            out[prefix + out_spatial] = value
            input_spatial = tuple(
                out_spatial[i] * strides[i] + local[i] * dilations[i] - pads[i]
                for i in range(spatial_rank)
            )
            if all(0 <= input_spatial[i] < data.shape[i + 2] for i in range(spatial_rank)):
                indices[prefix + out_spatial] = _pool_flat_index(data.shape, prefix[0], prefix[1], input_spatial, storage_order)
    return out, indices


# Egor Izmaylov: Function `_average_pool_nd` centralizes the average pool nd helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _average_pool_nd(data, kernel_shape, pads, strides, dilations, count_include_pad=0, ceil_mode=0, auto_pad="NOTSET"):
    spatial_rank, pads, strides, dilations = _normalize_pool_params(data.shape, kernel_shape, pads, strides, dilations, auto_pad)
    out_shape = _pool_output_shape(data.shape, kernel_shape, pads, strides, dilations, ceil_mode)
    pad_width = [(0, 0), (0, 0)] + [(pads[i], pads[i + spatial_rank]) for i in range(spatial_rank)]
    work = np.pad(data, pad_width, mode="constant", constant_values=0)
    valid = np.pad(np.ones(data.shape, dtype=np.float32), pad_width, mode="constant", constant_values=0)
    out = np.empty(out_shape, dtype=np.float64)
    full_count = np.prod(kernel_shape)

    for prefix in np.ndindex(data.shape[0], data.shape[1]):
        for out_spatial in np.ndindex(*out_shape[2:]):
            slices = _pool_window_slices(out_spatial, kernel_shape, pads, strides, dilations)
            window = work[prefix + slices]
            if count_include_pad:
                denom = full_count
            else:
                denom = np.sum(valid[prefix + slices])
            out[prefix + out_spatial] = 0.0 if denom == 0 else np.sum(window) / denom
    return out


# Egor Izmaylov: Function `_lp_pool_nd` centralizes the lp pool nd helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _lp_pool_nd(data, kernel_shape, pads, strides, dilations, p=2, ceil_mode=0, auto_pad="NOTSET"):
    spatial_rank, pads, strides, dilations = _normalize_pool_params(data.shape, kernel_shape, pads, strides, dilations, auto_pad)
    out_shape = _pool_output_shape(data.shape, kernel_shape, pads, strides, dilations, ceil_mode)
    pad_width = [(0, 0), (0, 0)] + [(pads[i], pads[i + spatial_rank]) for i in range(spatial_rank)]
    work = np.pad(data, pad_width, mode="constant", constant_values=0)
    out = np.empty(out_shape, dtype=np.float64)

    for prefix in np.ndindex(data.shape[0], data.shape[1]):
        for out_spatial in np.ndindex(*out_shape[2:]):
            window = work[prefix + _pool_window_slices(out_spatial, kernel_shape, pads, strides, dilations)]
            out[prefix + out_spatial] = np.sum(np.abs(window) ** p) ** (1.0 / p)
    return out


# Egor Izmaylov: Function `_grid_denormalize` centralizes the grid denormalize helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _grid_denormalize(coord, length, align_corners):
    if align_corners:
        return (coord + 1.0) * (length - 1) / 2.0
    return ((coord + 1.0) * length - 1.0) / 2.0


# Egor Izmaylov: Function `_reflect_coordinate` centralizes the reflect coordinate helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _reflect_coordinate(coord, low, high):
    if high <= low:
        return low
    span = high - low
    coord = abs((coord - low) % (2.0 * span))
    if coord > span:
        coord = 2.0 * span - coord
    return coord + low


# Egor Izmaylov: Function `_sample_coordinate` centralizes the sample coordinate helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _sample_coordinate(coord, length, padding_mode, align_corners):
    if padding_mode == "border":
        return min(max(coord, 0.0), length - 1.0)
    if padding_mode == "reflection":
        low, high = (0.0, length - 1.0) if align_corners else (-0.5, length - 0.5)
        reflected = _reflect_coordinate(coord, low, high)
        return min(max(reflected, 0.0), length - 1.0)
    return coord


# Egor Izmaylov: Function `_get_pixel_2d` centralizes the get pixel 2d helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _get_pixel_2d(data, y, x, padding_mode, align_corners):
    height, width = data.shape
    if padding_mode in {"border", "reflection"}:
        y = _sample_coordinate(y, height, padding_mode, align_corners)
        x = _sample_coordinate(x, width, padding_mode, align_corners)
    yi, xi = int(y), int(x)
    if yi < 0 or yi >= height or xi < 0 or xi >= width:
        return 0.0
    return data[yi, xi]


# Egor Izmaylov: Function `_bilinear_sample_2d` centralizes the bilinear sample 2d helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _bilinear_sample_2d(data, y, x, padding_mode="zeros", align_corners=False):
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = y0 + 1
    x1 = x0 + 1
    ly = y - y0
    lx = x - x0
    hy = 1.0 - ly
    hx = 1.0 - lx
    return (
        _get_pixel_2d(data, y0, x0, padding_mode, align_corners) * hy * hx
        + _get_pixel_2d(data, y0, x1, padding_mode, align_corners) * hy * lx
        + _get_pixel_2d(data, y1, x0, padding_mode, align_corners) * ly * hx
        + _get_pixel_2d(data, y1, x1, padding_mode, align_corners) * ly * lx
    )


# Egor Izmaylov: Function `_roi_align_weighted_terms` centralizes the roi align weighted terms helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _roi_align_weighted_terms(data, y, x):
    height, width = data.shape
    if y < -1.0 or y > height or x < -1.0 or x > width:
        return (0.0, 0.0, 0.0, 0.0)
    y = max(y, 0.0)
    x = max(x, 0.0)
    y_low = int(y)
    x_low = int(x)
    if y_low >= height - 1:
        y_high = y_low = height - 1
        y = float(y_low)
    else:
        y_high = y_low + 1
    if x_low >= width - 1:
        x_high = x_low = width - 1
        x = float(x_low)
    else:
        x_high = x_low + 1
    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx
    return (
        data[y_low, x_low] * hy * hx,
        data[y_low, x_high] * hy * lx,
        data[y_high, x_low] * ly * hx,
        data[y_high, x_high] * ly * lx,
    )


# Egor Izmaylov: Function `_cubic_coefficients` centralizes the cubic coefficients helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _cubic_coefficients(t):
    alpha = -0.75
    x = abs(t)
    return np.array([
        ((alpha * (x + 1) - 5 * alpha) * (x + 1) + 8 * alpha) * (x + 1) - 4 * alpha,
        ((alpha + 2) * x - (alpha + 3)) * x * x + 1,
        ((alpha + 2) * (1 - x) - (alpha + 3)) * (1 - x) * (1 - x) + 1,
        ((alpha * (2 - x) - 5 * alpha) * (2 - x) + 8 * alpha) * (2 - x) - 4 * alpha,
    ])


# Egor Izmaylov: Function `_bicubic_sample_2d` centralizes the bicubic sample 2d helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _bicubic_sample_2d(data, y, x, padding_mode="zeros", align_corners=False):
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    cy = _cubic_coefficients(y - y0)
    cx = _cubic_coefficients(x - x0)
    total = 0.0
    for iy in range(4):
        for ix in range(4):
            total += cy[iy] * cx[ix] * _get_pixel_2d(data, y0 - 1 + iy, x0 - 1 + ix, padding_mode, align_corners)
    return total


class GridSample(Ops):
    # Egor Izmaylov: Function `GridSample.__init__` initializes GridSample, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, mode="bilinear", padding_mode="zeros", align_corners=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = bool(align_corners)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.grid_sample_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]

    # Egor Izmaylov: Function `GridSample._mode_code` centralizes the mode code helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _mode_code(self):
        mode = {"linear": "bilinear", "cubic": "bicubic"}.get(self.mode, self.mode)
        if mode == "bilinear":
            return 0
        if mode == "nearest":
            return 1
        if mode == "bicubic":
            return 2
        raise ValueError(f"Unsupported GridSample mode {self.mode!r}")

    # Egor Izmaylov: Function `GridSample._padding_mode_code` centralizes the padding mode code helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _padding_mode_code(self):
        if self.padding_mode == "zeros":
            return 0
        if self.padding_mode == "border":
            return 1
        if self.padding_mode == "reflection":
            return 2
        raise ValueError(f"Unsupported GridSample padding_mode {self.padding_mode!r}")

    # Egor Izmaylov: Function `GridSample.forward` executes the concrete runtime path for GridSample, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, grid):
        data = np.asarray(x.data)
        grid_data = np.asarray(grid.data)
        if data.ndim != 4 or grid_data.ndim != 4 or grid_data.shape[-1] != 2:
            raise ValueError(f"GridSample expects X [N,C,H,W] and grid [N,Hout,Wout,2], got {data.shape}, {grid_data.shape}")
        n_batches, channels, height, width = data.shape
        if grid_data.shape[0] != n_batches:
            raise ValueError("GridSample batch dimensions must match")
        h_out, w_out = grid_data.shape[1], grid_data.shape[2]
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and grid.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            out_shape = (n_batches, channels, h_out, w_out)
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            grid_c = self._numpy_to_ctensor(np.ascontiguousarray(grid.data), grid.dtype)
            output_shape_c = (ctypes.c_int * 4)(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, 4, nn.DTYPE_MAP[self.dtype])
            self.lib.grid_sample_forward(
                x_c,
                grid_c,
                output_c,
                ctypes.c_int(self._mode_code()),
                ctypes.c_int(self._padding_mode_code()),
                ctypes.c_int(int(self.align_corners)),
            )
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(grid_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        out = np.empty((n_batches, channels, h_out, w_out), dtype=np.float64)
        for n in range(n_batches):
            for oy in range(h_out):
                for ox in range(w_out):
                    x_norm, y_norm = grid_data[n, oy, ox]
                    in_x = _grid_denormalize(float(x_norm), width, self.align_corners)
                    in_y = _grid_denormalize(float(y_norm), height, self.align_corners)
                    mode = {"linear": "bilinear", "cubic": "bicubic"}.get(self.mode, self.mode)
                    if mode == "nearest":
                        sample_y = int(np.rint(_sample_coordinate(in_y, height, self.padding_mode, self.align_corners)))
                        sample_x = int(np.rint(_sample_coordinate(in_x, width, self.padding_mode, self.align_corners)))
                        for c in range(channels):
                            out[n, c, oy, ox] = _get_pixel_2d(data[n, c], sample_y, sample_x, self.padding_mode, self.align_corners)
                    elif mode == "bilinear":
                        for c in range(channels):
                            out[n, c, oy, ox] = _bilinear_sample_2d(data[n, c], in_y, in_x, self.padding_mode, self.align_corners)
                    elif mode == "bicubic":
                        for c in range(channels):
                            out[n, c, oy, ox] = _bicubic_sample_2d(data[n, c], in_y, in_x, self.padding_mode, self.align_corners)
                    else:
                        raise ValueError(f"Unsupported GridSample mode {self.mode!r}")
        out_data = out.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, data.dtype), copy=False)
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `GridSample.forward_` performs shape-only inference for GridSample, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, grid):
        return {"tensor": Tensor_(x.size[0], x.size[1], grid.size[1], grid.size[2], dtype=self.dtype), "parameters": None}


class MaxPool(Ops):
    # Egor Izmaylov: Function `MaxPool.__init__` initializes MaxPool, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, kernel_shape, pads, strides, dtype, dilations=[1, 1], ceil_mode=0, storage_order=0, auto_pad="NOTSET", version="17"):
        super(MaxPool, self).__init__(inputs, outputs)
        self.kernel_shape = kernel_shape
        self.pads = pads
        self.strides = strides
        self.dilations = dilations
        self.ceil_mode = ceil_mode
        self.storage_order = storage_order
        self.auto_pad = auto_pad
        self.dtype = dtype
        self.version = version

        if self.lib:
            self.lib.max_pool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CPoolParams)
            ]

    # Egor Izmaylov: Function `MaxPool.forward` executes the concrete runtime path for MaxPool, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x: Tensor) -> dict:
        if (
            self.lib is not None
            and x.data.ndim == 4
            and len(self.kernel_shape) == 2
            and not (len(self.outputs) > 1 and self.outputs[1])
            and x.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
        ):
            _rank, pads, strides, dilations = _normalize_pool_params(
                x.size, self.kernel_shape, self.pads, self.strides, self.dilations, self.auto_pad
            )
            out_shape = _pool_output_shape(
                x.size, self.kernel_shape, pads, strides, dilations, self.ceil_mode, "NOTSET"
            )

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            pads_c = (ctypes.c_int * len(pads))(*pads)
            strides_c = (ctypes.c_int * len(strides))(*strides)
            dilations_c = (ctypes.c_int * len(dilations))(*dilations)
            kernel_c = (ctypes.c_int * len(self.kernel_shape))(*self.kernel_shape)
            c_params = CPoolParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.kernel_shape = ctypes.cast(kernel_c, ctypes.POINTER(ctypes.c_int))

            self.lib.max_pool_forward(x_c, out_c, ctypes.byref(c_params))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
            out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
            values = {"tensor": out_tensor, "parameters": None, "graph": None}
            self.parameters = {"values": values}
            return values

        out_data, indices_data = _max_pool_nd(
            x.data, self.kernel_shape, self.pads, self.strides, self.dilations, self.ceil_mode, self.storage_order, self.auto_pad
        )
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        out_shape = out_data.shape
        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        if len(self.outputs) > 1 and self.outputs[1]:
            indices_tensor = Tensor(*indices_data.shape, dtype="int64", data=indices_data)
            values = {"tensor": (out_tensor, indices_tensor), "parameters": None, "graph": None}
            self.parameters = {"values": values}
            return values

        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # Egor Izmaylov: Function `MaxPool.forward_` performs shape-only inference for MaxPool, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x: Tensor_) -> dict:
        out_shape = _pool_output_shape(x.size, self.kernel_shape, self.pads, self.strides, self.dilations, self.ceil_mode, self.auto_pad)
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        if len(self.outputs) > 1 and self.outputs[1]:
            values = {"tensor": (output_tensor, Tensor_(*out_shape, dtype="int64")), "parameters": None, "graph": None}
            self.parameters = {"values": values}
            return values
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

class MaxUnpool(Ops):
    # Egor Izmaylov: Function `MaxUnpool.__init__` initializes MaxUnpool, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, kernel_shape, pads=None, strides=None, dtype="float32", version="17"):
        super(MaxUnpool, self).__init__(inputs, outputs)
        self.kernel_shape = list(kernel_shape)
        spatial_rank = len(self.kernel_shape)
        self.pads = list(pads) if pads is not None else [0] * (2 * spatial_rank)
        self.strides = list(strides) if strides is not None else [1] * spatial_rank
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.max_unpool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CPoolParams)
            ]

    # Egor Izmaylov: Function `MaxUnpool._inferred_shape` centralizes the inferred shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _inferred_shape(self, x_shape):
        spatial_rank = len(x_shape) - 2
        if spatial_rank != len(self.kernel_shape):
            raise ValueError(f"MaxUnpool kernel rank {len(self.kernel_shape)} does not match input spatial rank {spatial_rank}")
        if len(self.pads) != 2 * spatial_rank:
            raise ValueError(f"MaxUnpool pads must contain {2 * spatial_rank} values")
        if len(self.strides) != spatial_rank:
            raise ValueError(f"MaxUnpool strides must contain {spatial_rank} values")
        out_shape = [x_shape[0], x_shape[1]]
        for dim in range(spatial_rank):
            out_shape.append(
                (x_shape[dim + 2] - 1) * self.strides[dim]
                - self.pads[dim]
                - self.pads[spatial_rank + dim]
                + self.kernel_shape[dim]
            )
        return tuple(out_shape)

    # Egor Izmaylov: Function `MaxUnpool.forward` executes the concrete runtime path for MaxUnpool, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, indices, output_shape=None):
        inferred_shape = self._inferred_shape(x.data.shape)
        shape = tuple(np.asarray(output_shape.data, dtype=np.int64).tolist()) if output_shape is not None else inferred_shape
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and indices.dtype in nn.DTYPE_MAP:
            spatial_rank = x.data.ndim - 2
            pads_c = (ctypes.c_int * len(self.pads))(*self.pads)
            strides_c = (ctypes.c_int * len(self.strides))(*self.strides)
            dilations_c = (ctypes.c_int * spatial_rank)(*([1] * spatial_rank))
            kernel_c = (ctypes.c_int * len(self.kernel_shape))(*self.kernel_shape)
            c_params = CPoolParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.kernel_shape = ctypes.cast(kernel_c, ctypes.POINTER(ctypes.c_int))

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data.astype(nn.DTYPE_TO_NUMPY[x.dtype], copy=False)), x.dtype)
            indices_c = self._numpy_to_ctensor(np.ascontiguousarray(indices.data.astype(nn.DTYPE_TO_NUMPY[indices.dtype], copy=False)), indices.dtype)
            output_shape_c = (ctypes.c_int * len(shape))(*shape)
            out_c = self.lib.create_tensor(output_shape_c, len(shape), nn.DTYPE_MAP[self.dtype])

            self.lib.max_unpool_forward(x_c, indices_c, out_c, ctypes.byref(c_params))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(indices_c)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*shape, dtype=self.dtype, data=out_data), "parameters": None}

        flat = np.zeros((int(np.prod(inferred_shape)),), dtype=x.data.dtype)
        x_flat = x.data.reshape(-1)
        idx_flat = indices.data.reshape(-1).astype(np.int64)
        for pos, value in zip(idx_flat, x_flat):
            flat[pos] = value
        inferred = flat.reshape(inferred_shape)
        out_data = np.zeros(shape, dtype=x.data.dtype)
        slices = tuple(slice(0, dim) for dim in inferred_shape)
        out_data[slices] = inferred
        out_data = out_data.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype), copy=False)
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `MaxUnpool.forward_` performs shape-only inference for MaxUnpool, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, indices, output_shape=None):
        if isinstance(output_shape, Tensor):
            shape = tuple(np.asarray(output_shape.data, dtype=np.int64).tolist())
        else:
            shape = self._inferred_shape(x.size)
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class MaxRoiPool(Ops):
    # Egor Izmaylov: Function `MaxRoiPool.__init__` initializes MaxRoiPool, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, pooled_shape, spatial_scale=1.0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        if len(pooled_shape) != 2:
            raise ValueError("MaxRoiPool pooled_shape must contain [height, width]")
        self.pooled_shape = tuple(int(v) for v in pooled_shape)
        self.spatial_scale = float(spatial_scale)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.max_roi_pool_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
            ]

    # Egor Izmaylov: Function `MaxRoiPool.forward` executes the concrete runtime path for MaxRoiPool, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, rois):
        data = np.asarray(x.data)
        roi_data = np.asarray(rois.data)
        if data.ndim != 4 or roi_data.ndim != 2 or roi_data.shape[1] != 5:
            raise ValueError(f"MaxRoiPool expects X [N,C,H,W] and rois [num_rois,5], got {data.shape}, {roi_data.shape}")
        pooled_h, pooled_w = self.pooled_shape
        num_rois, channels = roi_data.shape[0], data.shape[1]
        out_shape = (num_rois, channels, pooled_h, pooled_w)
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and rois.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            rois_c = self._numpy_to_ctensor(np.ascontiguousarray(rois.data), rois.dtype)
            output_shape_c = (ctypes.c_int * 4)(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, 4, nn.DTYPE_MAP[self.dtype])
            self.lib.max_roi_pool_forward(
                x_c,
                rois_c,
                output_c,
                ctypes.c_int(pooled_h),
                ctypes.c_int(pooled_w),
                ctypes.c_float(self.spatial_scale),
            )
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(rois_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        out = np.zeros((num_rois, channels, pooled_h, pooled_w), dtype=data.dtype)
        height, width = data.shape[2], data.shape[3]
        for roi_idx, roi in enumerate(roi_data):
            batch = int(roi[0])
            x1 = int(round(float(roi[1]) * self.spatial_scale))
            y1 = int(round(float(roi[2]) * self.spatial_scale))
            x2 = int(round(float(roi[3]) * self.spatial_scale))
            y2 = int(round(float(roi[4]) * self.spatial_scale))
            roi_w = max(x2 - x1 + 1, 1)
            roi_h = max(y2 - y1 + 1, 1)
            bin_h = float(roi_h) / float(pooled_h)
            bin_w = float(roi_w) / float(pooled_w)
            for ph in range(pooled_h):
                for pw in range(pooled_w):
                    hstart = int(np.floor(ph * bin_h)) + y1
                    hend = int(np.ceil((ph + 1) * bin_h)) + y1
                    wstart = int(np.floor(pw * bin_w)) + x1
                    wend = int(np.ceil((pw + 1) * bin_w)) + x1
                    hstart, hend = min(max(hstart, 0), height), min(max(hend, 0), height)
                    wstart, wend = min(max(wstart, 0), width), min(max(wend, 0), width)
                    if hend <= hstart or wend <= wstart:
                        out[roi_idx, :, ph, pw] = 0
                    else:
                        out[roi_idx, :, ph, pw] = np.max(data[batch, :, hstart:hend, wstart:wend], axis=(1, 2))
        out = out.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, out.dtype), copy=False)
        return {"tensor": Tensor(*out.shape, dtype=self.dtype, data=out), "parameters": None}

    # Egor Izmaylov: Function `MaxRoiPool.forward_` performs shape-only inference for MaxRoiPool, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, rois):
        return {"tensor": Tensor_(rois.size[0], x.size[1], self.pooled_shape[0], self.pooled_shape[1], dtype=self.dtype), "parameters": None}


class RoiAlign(Ops):
    # Egor Izmaylov: Function `RoiAlign.__init__` initializes RoiAlign, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(
        self,
        inputs,
        outputs,
        output_height=1,
        output_width=1,
        spatial_scale=1.0,
        sampling_ratio=0,
        mode="avg",
        coordinate_transformation_mode="half_pixel",
        dtype="float32",
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.mode = mode
        self.coordinate_transformation_mode = coordinate_transformation_mode
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.roi_align_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
            ]

    # Egor Izmaylov: Function `RoiAlign._mode_code` centralizes the mode code helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _mode_code(self):
        mode = self.mode.lower()
        if mode == "avg":
            return 0
        if mode == "max":
            return 1
        raise ValueError(f"Unsupported RoiAlign mode {self.mode!r}")

    # Egor Izmaylov: Function `RoiAlign._coordinate_transformation_mode_code` centralizes the coordinate transformation mode code helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _coordinate_transformation_mode_code(self):
        mode = self.coordinate_transformation_mode.lower()
        if mode == "half_pixel":
            return 0
        if mode == "output_half_pixel":
            return 1
        raise ValueError(f"Unsupported RoiAlign coordinate_transformation_mode {self.coordinate_transformation_mode!r}")

    # Egor Izmaylov: Function `RoiAlign.forward` executes the concrete runtime path for RoiAlign, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, rois, batch_indices):
        data = np.asarray(x.data)
        roi_data = np.asarray(rois.data)
        batches = np.asarray(batch_indices.data, dtype=np.int64).reshape(-1)
        if data.ndim != 4 or roi_data.ndim != 2 or roi_data.shape[1] != 4:
            raise ValueError(f"RoiAlign expects X [N,C,H,W] and rois [num_rois,4], got {data.shape}, {roi_data.shape}")
        if len(batches) != roi_data.shape[0]:
            raise ValueError("RoiAlign batch_indices length must match number of rois")
        num_rois, channels = roi_data.shape[0], data.shape[1]
        out_shape = (num_rois, channels, self.output_height, self.output_width)
        if (
            self.lib is not None
            and x.dtype in nn.DTYPE_MAP
            and rois.dtype in nn.DTYPE_MAP
            and batch_indices.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
        ):
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            rois_c = self._numpy_to_ctensor(np.ascontiguousarray(rois.data), rois.dtype)
            batch_c = self._numpy_to_ctensor(np.ascontiguousarray(batch_indices.data), batch_indices.dtype)
            output_shape_c = (ctypes.c_int * 4)(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, 4, nn.DTYPE_MAP[self.dtype])
            self.lib.roi_align_forward(
                x_c,
                rois_c,
                batch_c,
                output_c,
                ctypes.c_int(self.output_height),
                ctypes.c_int(self.output_width),
                ctypes.c_int(self.sampling_ratio),
                ctypes.c_float(self.spatial_scale),
                ctypes.c_int(self._mode_code()),
                ctypes.c_int(self._coordinate_transformation_mode_code()),
            )
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(rois_c)
            self.lib.free_tensor(batch_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        height, width = data.shape[2], data.shape[3]
        out = np.empty((num_rois, channels, self.output_height, self.output_width), dtype=np.float64)
        half_pixel = self.coordinate_transformation_mode.lower() == "half_pixel"
        offset = 0.5 if half_pixel else 0.0
        for roi_idx, roi in enumerate(roi_data):
            batch = int(batches[roi_idx])
            roi_start_w = float(roi[0]) * self.spatial_scale - offset
            roi_start_h = float(roi[1]) * self.spatial_scale - offset
            roi_end_w = float(roi[2]) * self.spatial_scale - offset
            roi_end_h = float(roi[3]) * self.spatial_scale - offset
            roi_w = roi_end_w - roi_start_w
            roi_h = roi_end_h - roi_start_h
            if not half_pixel:
                roi_w = max(roi_w, 1.0)
                roi_h = max(roi_h, 1.0)
            bin_h = roi_h / self.output_height
            bin_w = roi_w / self.output_width
            grid_h = self.sampling_ratio if self.sampling_ratio > 0 else int(np.ceil(roi_h / self.output_height))
            grid_w = self.sampling_ratio if self.sampling_ratio > 0 else int(np.ceil(roi_w / self.output_width))
            grid_h, grid_w = max(grid_h, 1), max(grid_w, 1)
            count = grid_h * grid_w
            for c in range(channels):
                image = data[batch, c]
                for ph in range(self.output_height):
                    for pw in range(self.output_width):
                        values = []
                        for iy in range(grid_h):
                            yy = roi_start_h + ph * bin_h + (iy + 0.5) * bin_h / grid_h
                            for ix in range(grid_w):
                                xx = roi_start_w + pw * bin_w + (ix + 0.5) * bin_w / grid_w
                                if self.mode.lower() == "max":
                                    values.append(max(_roi_align_weighted_terms(image, yy, xx)))
                                else:
                                    values.append(sum(_roi_align_weighted_terms(image, yy, xx)))
                        if self.mode.lower() == "max":
                            out[roi_idx, c, ph, pw] = max(values) if values else 0.0
                        elif self.mode.lower() == "avg":
                            out[roi_idx, c, ph, pw] = sum(values) / count
                        else:
                            raise ValueError(f"Unsupported RoiAlign mode {self.mode!r}")
        out = out.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, data.dtype), copy=False)
        return {"tensor": Tensor(*out.shape, dtype=self.dtype, data=out), "parameters": None}

    # Egor Izmaylov: Function `RoiAlign.forward_` performs shape-only inference for RoiAlign, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, rois, batch_indices):
        return {"tensor": Tensor_(rois.size[0], x.size[1], self.output_height, self.output_width, dtype=self.dtype), "parameters": None}

# Egor Izmaylov: Function `_num_directions` centralizes the num directions helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _num_directions(direction):
    if direction in ("forward", "reverse"):
        return 1
    if direction == "bidirectional":
        return 2
    raise ValueError(f"Unsupported recurrent direction {direction!r}")


# Egor Izmaylov: Function `_recurrent_time_major` centralizes the recurrent time major helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _recurrent_time_major(x, layout):
    return np.swapaxes(x, 0, 1) if layout == 1 else x


# Egor Izmaylov: Function `_recurrent_output_layout` centralizes the recurrent output layout helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _recurrent_output_layout(y, layout):
    return np.transpose(y, (2, 0, 1, 3)) if layout == 1 else y


# Egor Izmaylov: Function `_activation_function` centralizes the activation function helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _activation_function(name, alpha=None, beta=None):
    name = name.decode("utf-8") if isinstance(name, bytes) else name
    if name in (None, "Tanh", "tanh"):
        return np.tanh
    if name in ("Sigmoid", "sigmoid"):
        return lambda x: 1.0 / (1.0 + np.exp(-x))
    if name in ("Relu", "relu"):
        return lambda x: np.maximum(x, 0)
    if name in ("Affine", "affine"):
        a = 1.0 if alpha is None else alpha
        b = 0.0 if beta is None else beta
        return lambda x: a * x + b
    if name in ("LeakyRelu", "leakyrelu"):
        a = 0.01 if alpha is None else alpha
        return lambda x: np.where(x >= 0, x, a * x)
    if name in ("ThresholdedRelu", "thresholdedrelu"):
        a = 1.0 if alpha is None else alpha
        return lambda x: np.where(x >= a, x, 0)
    if name in ("ScaledTanh", "scaledtanh"):
        a = 1.0 if alpha is None else alpha
        b = 1.0 if beta is None else beta
        return lambda x: a * np.tanh(b * x)
    if name in ("HardSigmoid", "hardsigmoid"):
        a = 0.2 if alpha is None else alpha
        b = 0.5 if beta is None else beta
        return lambda x: np.clip(a * x + b, 0, 1)
    if name in ("Elu", "elu"):
        a = 1.0 if alpha is None else alpha
        return lambda x: np.where(x >= 0, x, a * (np.exp(x) - 1))
    if name in ("Softsign", "softsign"):
        return lambda x: x / (1 + np.abs(x))
    if name in ("Softplus", "softplus"):
        return lambda x: np.log1p(np.exp(x))
    raise ValueError(f"Unsupported recurrent activation {name!r}")


# Egor Izmaylov: Function `_activation_at` centralizes the activation at helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _activation_at(activations, alphas, betas, index, default):
    name = activations[index] if index < len(activations) else default
    alpha = alphas[index] if index < len(alphas) else None
    beta = betas[index] if index < len(betas) else None
    return _activation_function(name, alpha, beta)


_ACTIVATION_CODES = {
    "tanh": 0,
    "sigmoid": 1,
    "relu": 2,
    "affine": 3,
    "leakyrelu": 4,
    "thresholdedrelu": 5,
    "scaledtanh": 6,
    "hardsigmoid": 7,
    "elu": 8,
    "softsign": 9,
    "softplus": 10,
}


# Egor Izmaylov: Function `_clip_if_needed` centralizes the clip if needed helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _clip_if_needed(x, clip):
    return np.clip(x, -clip, clip) if clip is not None else x


# Egor Izmaylov: Function `_sequence_mask` centralizes the sequence mask helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _sequence_mask(sequence_lens, t, batch_size):
    if sequence_lens is None:
        return np.ones((batch_size, 1), dtype=bool)
    return (np.asarray(sequence_lens.data).reshape(-1) > t).reshape(batch_size, 1)


class RNN(Ops):
    # Egor Izmaylov: Function `RNN.__init__` initializes RNN, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(
        self,
        inputs,
        outputs,
        hidden_size=None,
        direction="forward",
        activations=None,
        activation_alpha=None,
        activation_beta=None,
        clip=None,
        layout=0,
        dtype="float32",
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.hidden_size = hidden_size
        self.direction = direction
        self.activations = list(activations or ["Tanh", "Tanh"])
        self.activation_alpha = list(activation_alpha or [])
        self.activation_beta = list(activation_beta or [])
        self.clip = clip
        self.layout = layout
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.rnn_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
            ]

    # Egor Izmaylov: Function `RNN._direction_code` centralizes the direction code helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _direction_code(self):
        return {"forward": 0, "reverse": 1, "bidirectional": 2}[self.direction]

    # Egor Izmaylov: Function `RNN._activation_buffers` centralizes the activation buffers helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _activation_buffers(self):
        codes = []
        for activation in self.activations:
            name = activation.decode("utf-8") if isinstance(activation, bytes) else activation
            key = str(name).lower()
            if key not in _ACTIVATION_CODES:
                raise ValueError(f"Unsupported recurrent activation {activation!r}")
            codes.append(_ACTIVATION_CODES[key])
        if not codes:
            codes = [0]
        alphas = [np.nan] * len(codes)
        betas = [np.nan] * len(codes)
        for idx, value in enumerate(self.activation_alpha[: len(codes)]):
            alphas[idx] = float(value)
        for idx, value in enumerate(self.activation_beta[: len(codes)]):
            betas[idx] = float(value)
        return (
            (ctypes.c_int * len(codes))(*codes),
            (ctypes.c_float * len(codes))(*alphas),
            (ctypes.c_float * len(codes))(*betas),
            len(codes),
        )

    # Egor Izmaylov: Function `RNN._optional_ctensor` centralizes the optional ctensor helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _optional_ctensor(self, tensor):
        if tensor is None:
            return None
        return self._numpy_to_ctensor(np.ascontiguousarray(tensor.data), tensor.dtype)

    # Egor Izmaylov: Function `RNN._c_supported` centralizes the c supported helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _c_supported(self, *tensors):
        return self.lib is not None and self.dtype in nn.DTYPE_MAP and all(
            tensor is None or tensor.dtype in nn.DTYPE_MAP for tensor in tensors
        )

    # Egor Izmaylov: Function `RNN.forward` executes the concrete runtime path for RNN, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, w, r, b=None, sequence_lens=None, initial_h=None):
        x_time = _recurrent_time_major(np.asarray(x.data), self.layout)
        seq_len, batch_size = x_time.shape[0], x_time.shape[1]
        num_dirs = w.data.shape[0]
        hidden = self.hidden_size or r.data.shape[-1]
        if self._c_supported(x, w, r, b, sequence_lens, initial_h):
            y_shape = (batch_size, seq_len, num_dirs, hidden) if self.layout == 1 else (seq_len, num_dirs, batch_size, hidden)
            y_h_shape = (num_dirs, batch_size, hidden)
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            w_c = self._numpy_to_ctensor(np.ascontiguousarray(w.data), w.dtype)
            r_c = self._numpy_to_ctensor(np.ascontiguousarray(r.data), r.dtype)
            b_c = self._optional_ctensor(b)
            seq_c = self._optional_ctensor(sequence_lens)
            init_c = self._optional_ctensor(initial_h)
            y_shape_c = (ctypes.c_int * 4)(*y_shape)
            yh_shape_c = (ctypes.c_int * 3)(*y_h_shape)
            y_c = self.lib.create_tensor(y_shape_c, 4, nn.DTYPE_MAP[self.dtype])
            yh_c = self.lib.create_tensor(yh_shape_c, 3, nn.DTYPE_MAP[self.dtype])
            act_codes, act_alpha, act_beta, act_count = self._activation_buffers()
            self.lib.rnn_forward(
                x_c, w_c, r_c, b_c, seq_c, init_c, y_c, yh_c,
                ctypes.c_int(hidden), ctypes.c_int(self._direction_code()), ctypes.c_int(self.layout),
                act_codes, act_alpha, act_beta, ctypes.c_int(act_count),
                ctypes.c_float(0.0 if self.clip is None else self.clip), ctypes.c_int(self.clip is not None),
            )
            y_data = self._ctensor_to_numpy(y_c, self.dtype)
            yh_data = self._ctensor_to_numpy(yh_c, self.dtype)
            for c_tensor in (x_c, w_c, r_c, b_c, seq_c, init_c, y_c, yh_c):
                if c_tensor is not None:
                    self.lib.free_tensor(c_tensor)
            outputs = (Tensor(*y_shape, dtype=self.dtype, data=y_data), Tensor(*y_h_shape, dtype=self.dtype, data=yh_data))
            return {"tensor": outputs[0] if len(self.outputs) == 1 else outputs, "parameters": None}

        bias = b.data if b is not None else np.zeros((num_dirs, 2 * hidden), dtype=x_time.dtype)
        h_prev = initial_h.data.copy() if initial_h is not None else np.zeros((num_dirs, batch_size, hidden), dtype=x_time.dtype)
        y = np.zeros((seq_len, num_dirs, batch_size, hidden), dtype=x_time.dtype)
        for direction_index in range(num_dirs):
            reverse = self.direction == "reverse" or (self.direction == "bidirectional" and direction_index == 1)
            time_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
            act = _activation_at(self.activations, self.activation_alpha, self.activation_beta, direction_index, "Tanh")
            wb, rb = np.split(bias[direction_index], 2)
            h_t = h_prev[direction_index]
            for t in time_indices:
                pre = x_time[t] @ w.data[direction_index].T + h_t @ r.data[direction_index].T + wb + rb
                h_new = act(_clip_if_needed(pre, self.clip))
                active = _sequence_mask(sequence_lens, t, batch_size)
                h_t = np.where(active, h_new, h_t)
                y[t, direction_index] = h_t
            h_prev[direction_index] = h_t
        y = _recurrent_output_layout(y, self.layout).astype(nn.DTYPE_TO_NUMPY.get(self.dtype, x_time.dtype), copy=False)
        y_h = h_prev.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, x_time.dtype), copy=False)
        outputs = (Tensor(*y.shape, dtype=self.dtype, data=y), Tensor(*y_h.shape, dtype=self.dtype, data=y_h))
        return {"tensor": outputs[0] if len(self.outputs) == 1 else outputs, "parameters": None}

    # Egor Izmaylov: Function `RNN.forward_` performs shape-only inference for RNN, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, w, r, b=None, sequence_lens=None, initial_h=None):
        num_dirs = _num_directions(self.direction)
        hidden = self.hidden_size or r.size[-1]
        if self.layout == 1:
            batch_size, seq_len = x.size[0], x.size[1]
            y_shape = (batch_size, seq_len, num_dirs, hidden)
        else:
            seq_len, batch_size = x.size[0], x.size[1]
            y_shape = (seq_len, num_dirs, batch_size, hidden)
        y = Tensor_(*y_shape, dtype=self.dtype)
        y_h = Tensor_(num_dirs, batch_size, hidden, dtype=self.dtype)
        return {"tensor": y if len(self.outputs) == 1 else (y, y_h), "parameters": None}


class GRU(RNN):
    # Egor Izmaylov: Function `GRU.__init__` initializes GRU, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(
        self,
        inputs,
        outputs,
        hidden_size=None,
        direction="forward",
        activations=None,
        activation_alpha=None,
        activation_beta=None,
        clip=None,
        layout=0,
        linear_before_reset=0,
        dtype="float32",
        version="17",
    ):
        super().__init__(inputs, outputs, hidden_size, direction, activations or ["Sigmoid", "Tanh"] * _num_directions(direction), activation_alpha, activation_beta, clip, layout, dtype, version)
        self.linear_before_reset = linear_before_reset
        if self.lib:
            self.lib.gru_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
            ]

    # Egor Izmaylov: Function `GRU.forward` executes the concrete runtime path for GRU, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, w, r, b=None, sequence_lens=None, initial_h=None):
        x_time = _recurrent_time_major(np.asarray(x.data), self.layout)
        seq_len, batch_size = x_time.shape[0], x_time.shape[1]
        num_dirs = w.data.shape[0]
        hidden = self.hidden_size or r.data.shape[-1]
        if self._c_supported(x, w, r, b, sequence_lens, initial_h):
            y_shape = (batch_size, seq_len, num_dirs, hidden) if self.layout == 1 else (seq_len, num_dirs, batch_size, hidden)
            y_h_shape = (num_dirs, batch_size, hidden)
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            w_c = self._numpy_to_ctensor(np.ascontiguousarray(w.data), w.dtype)
            r_c = self._numpy_to_ctensor(np.ascontiguousarray(r.data), r.dtype)
            b_c = self._optional_ctensor(b)
            seq_c = self._optional_ctensor(sequence_lens)
            init_c = self._optional_ctensor(initial_h)
            y_shape_c = (ctypes.c_int * 4)(*y_shape)
            yh_shape_c = (ctypes.c_int * 3)(*y_h_shape)
            y_c = self.lib.create_tensor(y_shape_c, 4, nn.DTYPE_MAP[self.dtype])
            yh_c = self.lib.create_tensor(yh_shape_c, 3, nn.DTYPE_MAP[self.dtype])
            act_codes, act_alpha, act_beta, act_count = self._activation_buffers()
            self.lib.gru_forward(
                x_c, w_c, r_c, b_c, seq_c, init_c, y_c, yh_c,
                ctypes.c_int(hidden), ctypes.c_int(self._direction_code()), ctypes.c_int(self.layout),
                ctypes.c_int(self.linear_before_reset), act_codes, act_alpha, act_beta,
                ctypes.c_int(act_count), ctypes.c_float(0.0 if self.clip is None else self.clip),
                ctypes.c_int(self.clip is not None),
            )
            y_data = self._ctensor_to_numpy(y_c, self.dtype)
            yh_data = self._ctensor_to_numpy(yh_c, self.dtype)
            for c_tensor in (x_c, w_c, r_c, b_c, seq_c, init_c, y_c, yh_c):
                if c_tensor is not None:
                    self.lib.free_tensor(c_tensor)
            outputs = (Tensor(*y_shape, dtype=self.dtype, data=y_data), Tensor(*y_h_shape, dtype=self.dtype, data=yh_data))
            return {"tensor": outputs[0] if len(self.outputs) == 1 else outputs, "parameters": None}

        bias = b.data if b is not None else np.zeros((num_dirs, 6 * hidden), dtype=x_time.dtype)
        h_prev = initial_h.data.copy() if initial_h is not None else np.zeros((num_dirs, batch_size, hidden), dtype=x_time.dtype)
        y = np.zeros((seq_len, num_dirs, batch_size, hidden), dtype=x_time.dtype)
        for direction_index in range(num_dirs):
            reverse = self.direction == "reverse" or (self.direction == "bidirectional" and direction_index == 1)
            time_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
            f = _activation_at(self.activations, self.activation_alpha, self.activation_beta, direction_index * 2, "Sigmoid")
            g = _activation_at(self.activations, self.activation_alpha, self.activation_beta, direction_index * 2 + 1, "Tanh")
            wz, wr, wh = np.split(w.data[direction_index], 3)
            rz, rr, rh = np.split(r.data[direction_index], 3)
            wbz, wbr, wbh, rbz, rbr, rbh = np.split(bias[direction_index], 6)
            h_t = h_prev[direction_index]
            for t in time_indices:
                z = f(_clip_if_needed(x_time[t] @ wz.T + h_t @ rz.T + wbz + rbz, self.clip))
                reset = f(_clip_if_needed(x_time[t] @ wr.T + h_t @ rr.T + wbr + rbr, self.clip))
                if self.linear_before_reset:
                    h_candidate = g(_clip_if_needed(x_time[t] @ wh.T + reset * (h_t @ rh.T + rbh) + wbh, self.clip))
                else:
                    h_candidate = g(_clip_if_needed(x_time[t] @ wh.T + (reset * h_t) @ rh.T + wbh + rbh, self.clip))
                h_new = (1.0 - z) * h_candidate + z * h_t
                active = _sequence_mask(sequence_lens, t, batch_size)
                h_t = np.where(active, h_new, h_t)
                y[t, direction_index] = h_t
            h_prev[direction_index] = h_t
        y = _recurrent_output_layout(y, self.layout).astype(nn.DTYPE_TO_NUMPY.get(self.dtype, x_time.dtype), copy=False)
        y_h = h_prev.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, x_time.dtype), copy=False)
        outputs = (Tensor(*y.shape, dtype=self.dtype, data=y), Tensor(*y_h.shape, dtype=self.dtype, data=y_h))
        return {"tensor": outputs[0] if len(self.outputs) == 1 else outputs, "parameters": None}


class LSTM(RNN):
    # Egor Izmaylov: Function `LSTM.__init__` initializes LSTM, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(
        self,
        inputs,
        outputs,
        hidden_size=None,
        direction="forward",
        activations=None,
        activation_alpha=None,
        activation_beta=None,
        clip=None,
        layout=0,
        input_forget=0,
        dtype="float32",
        version="17",
    ):
        super().__init__(inputs, outputs, hidden_size, direction, activations or ["Sigmoid", "Tanh", "Tanh"] * _num_directions(direction), activation_alpha, activation_beta, clip, layout, dtype, version)
        self.input_forget = input_forget
        if self.lib:
            self.lib.lstm_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
            ]

    # Egor Izmaylov: Function `LSTM.forward` executes the concrete runtime path for LSTM, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, w, r, b=None, sequence_lens=None, initial_h=None, initial_c=None, p=None):
        x_time = _recurrent_time_major(np.asarray(x.data), self.layout)
        seq_len, batch_size = x_time.shape[0], x_time.shape[1]
        num_dirs = w.data.shape[0]
        hidden = self.hidden_size or r.data.shape[-1]
        if self._c_supported(x, w, r, b, sequence_lens, initial_h, initial_c, p):
            y_shape = (batch_size, seq_len, num_dirs, hidden) if self.layout == 1 else (seq_len, num_dirs, batch_size, hidden)
            y_h_shape = (num_dirs, batch_size, hidden)
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            w_c = self._numpy_to_ctensor(np.ascontiguousarray(w.data), w.dtype)
            r_c = self._numpy_to_ctensor(np.ascontiguousarray(r.data), r.dtype)
            b_c = self._optional_ctensor(b)
            seq_c = self._optional_ctensor(sequence_lens)
            init_h_c = self._optional_ctensor(initial_h)
            init_c_c = self._optional_ctensor(initial_c)
            p_c = self._optional_ctensor(p)
            y_shape_c = (ctypes.c_int * 4)(*y_shape)
            yh_shape_c = (ctypes.c_int * 3)(*y_h_shape)
            y_c = self.lib.create_tensor(y_shape_c, 4, nn.DTYPE_MAP[self.dtype])
            yh_c = self.lib.create_tensor(yh_shape_c, 3, nn.DTYPE_MAP[self.dtype])
            yc_c = self.lib.create_tensor(yh_shape_c, 3, nn.DTYPE_MAP[self.dtype])
            act_codes, act_alpha, act_beta, act_count = self._activation_buffers()
            self.lib.lstm_forward(
                x_c, w_c, r_c, b_c, seq_c, init_h_c, init_c_c, p_c,
                y_c, yh_c, yc_c, ctypes.c_int(hidden), ctypes.c_int(self._direction_code()),
                ctypes.c_int(self.layout), ctypes.c_int(self.input_forget), act_codes,
                act_alpha, act_beta, ctypes.c_int(act_count),
                ctypes.c_float(0.0 if self.clip is None else self.clip), ctypes.c_int(self.clip is not None),
            )
            y_data = self._ctensor_to_numpy(y_c, self.dtype)
            yh_data = self._ctensor_to_numpy(yh_c, self.dtype)
            yc_data = self._ctensor_to_numpy(yc_c, self.dtype)
            for c_tensor in (x_c, w_c, r_c, b_c, seq_c, init_h_c, init_c_c, p_c, y_c, yh_c, yc_c):
                if c_tensor is not None:
                    self.lib.free_tensor(c_tensor)
            outputs = (
                Tensor(*y_shape, dtype=self.dtype, data=y_data),
                Tensor(*y_h_shape, dtype=self.dtype, data=yh_data),
                Tensor(*y_h_shape, dtype=self.dtype, data=yc_data),
            )
            selected = tuple(value for name, value in zip(self.outputs, outputs) if name)
            return {"tensor": selected[0] if len(selected) == 1 else selected, "parameters": None}

        bias = b.data if b is not None else np.zeros((num_dirs, 8 * hidden), dtype=x_time.dtype)
        peepholes = p.data if p is not None else np.zeros((num_dirs, 3 * hidden), dtype=x_time.dtype)
        h_prev = initial_h.data.copy() if initial_h is not None else np.zeros((num_dirs, batch_size, hidden), dtype=x_time.dtype)
        c_prev = initial_c.data.copy() if initial_c is not None else np.zeros((num_dirs, batch_size, hidden), dtype=x_time.dtype)
        y = np.zeros((seq_len, num_dirs, batch_size, hidden), dtype=x_time.dtype)
        for direction_index in range(num_dirs):
            reverse = self.direction == "reverse" or (self.direction == "bidirectional" and direction_index == 1)
            time_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
            f_act = _activation_at(self.activations, self.activation_alpha, self.activation_beta, direction_index * 3, "Sigmoid")
            g_act = _activation_at(self.activations, self.activation_alpha, self.activation_beta, direction_index * 3 + 1, "Tanh")
            h_act = _activation_at(self.activations, self.activation_alpha, self.activation_beta, direction_index * 3 + 2, "Tanh")
            p_i, p_o, p_f = np.split(peepholes[direction_index], 3)
            h_t = h_prev[direction_index]
            c_t = c_prev[direction_index]
            bias_sum = np.add(*np.split(bias[direction_index], 2))
            for t in time_indices:
                i, o, forget, c = np.split(x_time[t] @ w.data[direction_index].T + h_t @ r.data[direction_index].T + bias_sum, 4, axis=-1)
                i = f_act(_clip_if_needed(i + p_i * c_t, self.clip))
                forget = 1.0 - i if self.input_forget else f_act(_clip_if_needed(forget + p_f * c_t, self.clip))
                c_bar = g_act(_clip_if_needed(c, self.clip))
                c_new = forget * c_t + i * c_bar
                o = f_act(_clip_if_needed(o + p_o * c_new, self.clip))
                h_new = o * h_act(c_new)
                active = _sequence_mask(sequence_lens, t, batch_size)
                h_t = np.where(active, h_new, h_t)
                c_t = np.where(active, c_new, c_t)
                y[t, direction_index] = h_t
            h_prev[direction_index] = h_t
            c_prev[direction_index] = c_t
        y = _recurrent_output_layout(y, self.layout).astype(nn.DTYPE_TO_NUMPY.get(self.dtype, x_time.dtype), copy=False)
        y_h = h_prev.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, x_time.dtype), copy=False)
        y_c = c_prev.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, x_time.dtype), copy=False)
        outputs = (
            Tensor(*y.shape, dtype=self.dtype, data=y),
            Tensor(*y_h.shape, dtype=self.dtype, data=y_h),
            Tensor(*y_c.shape, dtype=self.dtype, data=y_c),
        )
        selected = tuple(value for name, value in zip(self.outputs, outputs) if name)
        return {"tensor": selected[0] if len(selected) == 1 else selected, "parameters": None}

    # Egor Izmaylov: Function `LSTM.forward_` performs shape-only inference for LSTM, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, w, r, b=None, sequence_lens=None, initial_h=None, initial_c=None, p=None):
        num_dirs = _num_directions(self.direction)
        hidden = self.hidden_size or r.size[-1]
        if self.layout == 1:
            batch_size, seq_len = x.size[0], x.size[1]
            y_shape = (batch_size, seq_len, num_dirs, hidden)
        else:
            seq_len, batch_size = x.size[0], x.size[1]
            y_shape = (seq_len, num_dirs, batch_size, hidden)
        y = Tensor_(*y_shape, dtype=self.dtype)
        y_h = Tensor_(num_dirs, batch_size, hidden, dtype=self.dtype)
        y_c = Tensor_(num_dirs, batch_size, hidden, dtype=self.dtype)
        selected = tuple(value for name, value in zip(self.outputs, (y, y_h, y_c)) if name)
        return {"tensor": selected[0] if len(selected) == 1 else selected, "parameters": None}

class Gemm(Ops):
    # Egor Izmaylov: Function `Gemm.__init__` initializes Gemm, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `Gemm.forward` executes the concrete runtime path for Gemm, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Gemm.forward_` performs shape-only inference for Gemm, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, A: Tensor_, B: Tensor_, C: Tensor_ = None) -> dict:
        M = A.size[0] if self.transA == 0 else A.size[1]
        N = B.size[1] if self.transB == 0 else B.size[0]
        output_tensor = Tensor_(M, N, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

class Softmax(Ops):
    # Egor Izmaylov: Function `Softmax.__init__` initializes Softmax, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axis, dtype, version="17"):
        super(Softmax, self).__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        
        if self.lib:
            self.lib.softmax_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int
            ]

    # Egor Izmaylov: Function `Softmax.forward` executes the concrete runtime path for Softmax, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Softmax.forward_` performs shape-only inference for Softmax, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values
    
class EXP(Ops):
    # Egor Izmaylov: Function `EXP.__init__` initializes EXP, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(EXP, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `EXP.forward` executes the concrete runtime path for EXP, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "exp_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # Egor Izmaylov: Function `EXP.forward_` performs shape-only inference for EXP, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class LOG(Ops):
    # Egor Izmaylov: Function `LOG.__init__` initializes LOG, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(LOG, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `LOG.forward` executes the concrete runtime path for LOG, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "log_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # Egor Izmaylov: Function `LOG.forward_` performs shape-only inference for LOG, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class SQRT(Ops):
    # Egor Izmaylov: Function `SQRT.__init__` initializes SQRT, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(SQRT, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `SQRT.forward` executes the concrete runtime path for SQRT, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "sqrt_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # Egor Izmaylov: Function `SQRT.forward_` performs shape-only inference for SQRT, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class SIGMOID(Ops):
    # Egor Izmaylov: Function `SIGMOID.__init__` initializes SIGMOID, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(SIGMOID, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `SIGMOID.forward` executes the concrete runtime path for SIGMOID, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "sigmoid_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # Egor Izmaylov: Function `SIGMOID.forward_` performs shape-only inference for SIGMOID, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class TANH(Ops):
    # Egor Izmaylov: Function `TANH.__init__` initializes TANH, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(TANH, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `TANH.forward` executes the concrete runtime path for TANH, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "tanh_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # Egor Izmaylov: Function `TANH.forward_` performs shape-only inference for TANH, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}
    
class Flatten(Ops):
    # Egor Izmaylov: Function `Flatten.__init__` initializes Flatten, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axis=1, dtype="float32", version="17"):
        super(Flatten, self).__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `Flatten._calc_shape` centralizes the calc shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
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

    # Egor Izmaylov: Function `Flatten.forward` executes the concrete runtime path for Flatten, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Flatten.forward_` performs shape-only inference for Flatten, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        out_shape = self._calc_shape(input.size)
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

class Reshape(Ops):
    # Egor Izmaylov: Function `Reshape.__init__` initializes Reshape, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17", allowzero=0):
        super(Reshape, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        self.allowzero = allowzero

    # Egor Izmaylov: Function `Reshape._resolve_shape` centralizes the resolve shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
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

    # Egor Izmaylov: Function `Reshape.forward` executes the concrete runtime path for Reshape, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Reshape.forward_` performs shape-only inference for Reshape, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
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
    # Egor Izmaylov: Function `Transpose.__init__` initializes Transpose, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, perm=None, dtype="float32", version="17"):
        super(Transpose, self).__init__(inputs, outputs)
        self.perm = None if perm is None else list(perm)
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.transpose_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor), ctypes.POINTER(ctypes.c_int)
            ]

    # Egor Izmaylov: Function `Transpose._resolve_perm` centralizes the resolve perm helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
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

    # Egor Izmaylov: Function `Transpose.forward` executes the concrete runtime path for Transpose, consuming real tensor values and returning the graph-runner value contract.
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
    # Egor Izmaylov: Function `Transpose.forward_` performs shape-only inference for Transpose, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        perm = self._resolve_perm(len(input.size))
        out_shape = [input.size[i] for i in perm]

        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values
    
class Pow(Ops):
    # Egor Izmaylov: Function `Pow.__init__` initializes Pow, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Pow, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `Pow.forward` executes the concrete runtime path for Pow, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input_a: Tensor, input_b: Tensor) -> dict:
        out_tensor = self._execute_binary(input_a, input_b, "pow_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # Egor Izmaylov: Function `Pow.forward_` performs shape-only inference for Pow, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input_a: Tensor_, input_b: Tensor_) -> dict:
        # 简单广播推断
        try:
            bcast = np.broadcast_shapes(input_a.size, input_b.size)
        except:
            bcast = input_a.size
        output_tensor = Tensor_(*bcast, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Max(Ops):
    # Egor Izmaylov: Function `Max.__init__` initializes Max, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Max, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `Max.forward` executes the concrete runtime path for Max, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, *inputs: Tensor) -> dict:
        if not inputs:
            raise ValueError("Max requires at least one input")
        if self.lib is None:
            arrays = np.broadcast_arrays(*(x.data for x in inputs))
            out_data = arrays[0]
            for arr in arrays[1:]:
                out_data = np.maximum(out_data, arr)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
            return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}
        out_tensor = inputs[0]
        for next_tensor in inputs[1:]:
            out_tensor = self._execute_binary(out_tensor, next_tensor, "max_forward")
        if self.dtype and out_tensor.dtype != self.dtype:
            out_data = out_tensor.data.astype(nn.DTYPE_TO_NUMPY[self.dtype], copy=False)
            out_tensor = Tensor(*out_data.shape, dtype=self.dtype, data=out_data)
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # Egor Izmaylov: Function `Max.forward_` performs shape-only inference for Max, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, *inputs: Tensor_) -> dict:
        if not inputs:
            raise ValueError("Max requires at least one input")
        bcast = np.broadcast_shapes(*(x.size for x in inputs))
        output_tensor = Tensor_(*bcast, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Min(Ops):
    # Egor Izmaylov: Function `Min.__init__` initializes Min, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Min, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `Min.forward` executes the concrete runtime path for Min, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, *inputs: Tensor) -> dict:
        if not inputs:
            raise ValueError("Min requires at least one input")
        if self.lib is None:
            arrays = np.broadcast_arrays(*(x.data for x in inputs))
            out_data = arrays[0]
            for arr in arrays[1:]:
                out_data = np.minimum(out_data, arr)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
            return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}
        out_tensor = inputs[0]
        for next_tensor in inputs[1:]:
            out_tensor = self._execute_binary(out_tensor, next_tensor, "min_forward")
        if self.dtype and out_tensor.dtype != self.dtype:
            out_data = out_tensor.data.astype(nn.DTYPE_TO_NUMPY[self.dtype], copy=False)
            out_tensor = Tensor(*out_data.shape, dtype=self.dtype, data=out_data)
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # Egor Izmaylov: Function `Min.forward_` performs shape-only inference for Min, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, *inputs: Tensor_) -> dict:
        if not inputs:
            raise ValueError("Min requires at least one input")
        bcast = np.broadcast_shapes(*(x.size for x in inputs))
        output_tensor = Tensor_(*bcast, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Squeeze(Ops):
    # Egor Izmaylov: Function `Squeeze.__init__` initializes Squeeze, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axes=None, dtype="float32", version="17"):
        super(Squeeze, self).__init__(inputs, outputs)
        self.axes = axes
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `Squeeze._calc_shape` centralizes the calc shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
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

    # Egor Izmaylov: Function `Squeeze.forward` executes the concrete runtime path for Squeeze, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Squeeze.forward_` performs shape-only inference for Squeeze, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
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
    # Egor Izmaylov: Function `Unsqueeze.__init__` initializes Unsqueeze, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axes=None, dtype="float32", version="17"):
        super(Unsqueeze, self).__init__(inputs, outputs)
        self.axes = axes
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `Unsqueeze._calc_shape` centralizes the calc shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
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

    # Egor Izmaylov: Function `Unsqueeze.forward` executes the concrete runtime path for Unsqueeze, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Unsqueeze.forward_` performs shape-only inference for Unsqueeze, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
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
    # Egor Izmaylov: Function `Concat.__init__` initializes Concat, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `Concat._calc_shape` centralizes the calc shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
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

    # Egor Izmaylov: Function `Concat.forward` executes the concrete runtime path for Concat, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Concat.forward_` performs shape-only inference for Concat, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, *inputs) -> dict:
        input_list = list(inputs)
        out_shape, _ = self._calc_shape(input_list)
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

class Slice(Ops):
    # Egor Izmaylov: Function `Slice.__init__` initializes Slice, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super(Slice, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.slice_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor), 
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
            ]

    # Egor Izmaylov: Function `Slice.forward` executes the concrete runtime path for Slice, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Slice.forward_` performs shape-only inference for Slice, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
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
    
class Neg(Ops):
    # Egor Izmaylov: Function `Neg.__init__` initializes Neg, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Neg, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Neg.forward` executes the concrete runtime path for Neg, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "neg_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}
    # Egor Izmaylov: Function `Neg.forward_` performs shape-only inference for Neg, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Reciprocal(Ops):
    # Egor Izmaylov: Function `Reciprocal.__init__` initializes Reciprocal, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Reciprocal, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Reciprocal.forward` executes the concrete runtime path for Reciprocal, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "reciprocal_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}
    # Egor Izmaylov: Function `Reciprocal.forward_` performs shape-only inference for Reciprocal, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Ceil(Ops):
    # Egor Izmaylov: Function `Ceil.__init__` initializes Ceil, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Ceil, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Ceil.forward` executes the concrete runtime path for Ceil, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "ceil_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}
    # Egor Izmaylov: Function `Ceil.forward_` performs shape-only inference for Ceil, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Floor(Ops):
    # Egor Izmaylov: Function `Floor.__init__` initializes Floor, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Floor, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Floor.forward` executes the concrete runtime path for Floor, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "floor_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}
    # Egor Izmaylov: Function `Floor.forward_` performs shape-only inference for Floor, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Cast(Ops):
    # Egor Izmaylov: Function `Cast.__init__` initializes Cast, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Cast, self).__init__(inputs, outputs)
        self.dtype = dtype # 这里的 dtype 就是目标类型
        self.version = version
        if self.lib:
            self.lib.cast_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

    # Egor Izmaylov: Function `Cast.forward` executes the concrete runtime path for Cast, consuming real tensor values and returning the graph-runner value contract.
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
    # Egor Izmaylov: Function `Cast.forward_` performs shape-only inference for Cast, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class CastLike(Ops):
    # Egor Izmaylov: Function `CastLike.__init__` initializes CastLike, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype=None, version="17"):
        super(CastLike, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.cast_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

    # Egor Izmaylov: Function `CastLike.forward` executes the concrete runtime path for CastLike, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `CastLike.forward_` performs shape-only inference for CastLike, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_, target_type: Tensor_) -> dict:
        out_dtype = self.dtype or target_type.dtype
        return {"tensor": Tensor_(*input.size, dtype=out_dtype), "parameters": None, "graph": None}

class Sum(Ops):
    # Egor Izmaylov: Function `Sum.__init__` initializes Sum, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super(Sum, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.sum_forward.argtypes = [
                ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.POINTER(CTensor)
            ]

    # Egor Izmaylov: Function `Sum.forward` executes the concrete runtime path for Sum, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Sum.forward_` performs shape-only inference for Sum, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, *inputs: Tensor_) -> dict:
        if not inputs:
            raise ValueError("Sum requires at least one input")
        out_shape = np.broadcast_shapes(*(x.size for x in inputs))
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None, "graph": None}

class PRelu(Ops):
    # Egor Izmaylov: Function `PRelu.__init__` initializes PRelu, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super(PRelu, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.prelu_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)
            ]

    # Egor Izmaylov: Function `PRelu.forward` executes the concrete runtime path for PRelu, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `PRelu.forward_` performs shape-only inference for PRelu, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x: Tensor_, slope: Tensor_) -> dict:
        out_shape = np.broadcast_shapes(x.size, slope.size)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None, "graph": None}

class Det(Ops):
    # Egor Izmaylov: Function `Det.__init__` initializes Det, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super(Det, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.det_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]

    # Egor Izmaylov: Function `Det.forward` executes the concrete runtime path for Det, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Det.forward_` performs shape-only inference for Det, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x: Tensor_) -> dict:
        if len(x.size) < 2 or x.size[-1] != x.size[-2]:
            raise ValueError(f"Det expects input shape [..., M, M], got {x.size}")
        return {"tensor": Tensor_(*x.size[:-2], dtype=self.dtype), "parameters": None, "graph": None}

# Egor Izmaylov: Function `_matmul_output_shape` centralizes the matmul output shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _matmul_output_shape(shape_a, shape_b):
    shape_a = list(shape_a)
    shape_b = list(shape_b)
    if len(shape_a) == 0 or len(shape_b) == 0:
        raise ValueError("MatMul inputs must have rank >= 1")
    is_a_1d = len(shape_a) == 1
    is_b_1d = len(shape_b) == 1
    if is_a_1d:
        shape_a = [1] + shape_a
    if is_b_1d:
        shape_b = shape_b + [1]
    if shape_a[-1] != shape_b[-2]:
        raise ValueError(f"MatMul shape mismatch: {shape_a[-1]} != {shape_b[-2]}")
    batch = np.broadcast_shapes(tuple(shape_a[:-2]), tuple(shape_b[:-2]))
    out_shape = list(batch) + [shape_a[-2], shape_b[-1]]
    if is_b_1d:
        out_shape.pop(-1)
    if is_a_1d:
        out_shape.pop(-1 if is_b_1d else -2)
    return tuple(out_shape)


# Egor Izmaylov: Function `_prepare_matmul_c_shapes` centralizes the prepare matmul c shapes helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _prepare_matmul_c_shapes(input_a: Tensor, input_b: Tensor):
    data_a = np.asarray(input_a.data)
    data_b = np.asarray(input_b.data)
    is_a_1d = data_a.ndim == 1
    is_b_1d = data_b.ndim == 1

    if is_a_1d:
        data_a = data_a[np.newaxis, :]
    if is_b_1d:
        data_b = data_b[:, np.newaxis]

    shape_a = list(data_a.shape)
    shape_b = list(data_b.shape)
    if shape_a[-1] != shape_b[-2]:
        raise ValueError(f"MatMul shape mismatch: {shape_a[-1]} != {shape_b[-2]}")
    batch_out = np.broadcast_shapes(tuple(shape_a[:-2]), tuple(shape_b[:-2]))
    out_shape_for_c = tuple(list(batch_out) + [shape_a[-2], shape_b[-1]])

    final_shape = list(out_shape_for_c)
    if is_b_1d:
        final_shape.pop(-1)
    if is_a_1d:
        final_shape.pop(-1 if is_b_1d else -2)
    return data_a, data_b, out_shape_for_c, tuple(final_shape)


# Egor Izmaylov: Function `_broadcast_matmul_param` centralizes the broadcast matmul param helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _broadcast_matmul_param(param, target_shape, dtype, role):
    np_dtype = nn.DTYPE_TO_NUMPY[dtype]
    if param is None:
        return np.zeros(target_shape, dtype=np_dtype)

    arr = np.asarray(param.data, dtype=np_dtype)
    if arr.shape == target_shape:
        return np.ascontiguousarray(arr)
    if arr.shape == ():
        return np.broadcast_to(arr, target_shape).copy()
    if arr.ndim == 1 and len(target_shape) >= 2:
        axis_len = target_shape[-2] if role == "row" else target_shape[-1]
        if arr.shape[0] == axis_len:
            shape = (1,) * (len(target_shape) - 2)
            shape = shape + ((axis_len, 1) if role == "row" else (1, axis_len))
            return np.broadcast_to(arr.reshape(shape), target_shape).copy()
    return np.broadcast_to(arr, target_shape).copy()


# Egor Izmaylov: Function `_broadcast_output_param` centralizes the broadcast output param helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _broadcast_output_param(param, target_shape, dtype):
    arr = np.asarray(param.data, dtype=nn.DTYPE_TO_NUMPY[dtype])
    if arr.shape == target_shape:
        return np.ascontiguousarray(arr)
    return np.broadcast_to(arr, target_shape).copy()


class MatMulInteger(Ops):
    # Egor Izmaylov: Function `MatMulInteger.__init__` initializes MatMulInteger, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="int32", version="17"):
        super(MatMulInteger, self).__init__(inputs, outputs)
        self.dtype = "int32"
        self.version = version
        if self.lib:
            self.lib.matmul_integer_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)
            ]

    # Egor Izmaylov: Function `MatMulInteger._zero_point_data` centralizes the zero point data helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _zero_point_data(zero_point, dtype):
        if zero_point is None:
            return np.array(0, dtype=nn.DTYPE_TO_NUMPY[dtype])
        return zero_point.data

    # Egor Izmaylov: Function `MatMulInteger.forward` executes the concrete runtime path for MatMulInteger, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `MatMulInteger.forward_` performs shape-only inference for MatMulInteger, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, A: Tensor_, B: Tensor_, a_zero_point: Tensor_ = None, b_zero_point: Tensor_ = None) -> dict:
        return {"tensor": Tensor_(*_matmul_output_shape(A.size, B.size), dtype=self.dtype), "parameters": None, "graph": None}

class QLinearMatMul(Ops):
    # Egor Izmaylov: Function `QLinearMatMul.__init__` initializes QLinearMatMul, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `QLinearMatMul.forward` executes the concrete runtime path for QLinearMatMul, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `QLinearMatMul.forward_` performs shape-only inference for QLinearMatMul, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point):
        out_dtype = y_zero_point.dtype if y_zero_point is not None else self.dtype
        return {"tensor": Tensor_(*_matmul_output_shape(a.size, b.size), dtype=out_dtype), "parameters": None, "graph": None}

class LRN(Ops):
    # Egor Izmaylov: Function `LRN.__init__` initializes LRN, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `LRN.forward` executes the concrete runtime path for LRN, consuming real tensor values and returning the graph-runner value contract.
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
        data = x.data.astype(np.float32, copy=False)
        square_sum = np.zeros_like(data, dtype=np.float32)
        channels = data.shape[1]
        lower = (self.size - 1) // 2
        upper = self.size - 1 - lower
        for c in range(channels):
            begin = max(0, c - lower)
            end = min(channels, c + upper + 1)
            square_sum[:, c, ...] = np.sum(data[:, begin:end, ...] ** 2, axis=1)
        out_data = data / ((self.bias + (self.alpha / self.size) * square_sum) ** self.beta)
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*x.size, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # Egor Izmaylov: Function `LRN.forward_` performs shape-only inference for LRN, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x: Tensor_) -> dict:
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None, "graph": None}

class MeanVarianceNormalization(Ops):
    # Egor Izmaylov: Function `MeanVarianceNormalization.__init__` initializes MeanVarianceNormalization, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axes=None, dtype="float32", version="17"):
        super(MeanVarianceNormalization, self).__init__(inputs, outputs)
        self.axes = list(axes) if axes is not None else [0, 2, 3]
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.mean_variance_normalization_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CReduceParams)
            ]

    # Egor Izmaylov: Function `MeanVarianceNormalization._axes_for_rank` centralizes the axes for rank helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _axes_for_rank(self, rank):
        axes = []
        for ax in self.axes:
            axis = ax + rank if ax < 0 else ax
            if axis < 0 or axis >= rank:
                raise ValueError(f"MeanVarianceNormalization axis {ax} is out of bounds for rank {rank}")
            axes.append(axis)
        return tuple(sorted(set(axes)))

    # Egor Izmaylov: Function `MeanVarianceNormalization.forward` executes the concrete runtime path for MeanVarianceNormalization, consuming real tensor values and returning the graph-runner value contract.
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
        data = x.data.astype(np.float32, copy=False)
        mean = np.mean(data, axis=axes, keepdims=True)
        variance = np.mean((data - mean) ** 2, axis=axes, keepdims=True)
        out_data = (data - mean) / np.sqrt(variance)
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*x.size, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # Egor Izmaylov: Function `MeanVarianceNormalization.forward_` performs shape-only inference for MeanVarianceNormalization, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x: Tensor_) -> dict:
        self._axes_for_rank(len(x.size))
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None, "graph": None}

class EyeLike(Ops):
    # Egor Izmaylov: Function `EyeLike.__init__` initializes EyeLike, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, k=0, dtype=None, version="17"):
        super(EyeLike, self).__init__(inputs, outputs)
        self.k = k
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.eye_like_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.c_int]

    # Egor Izmaylov: Function `EyeLike.forward` executes the concrete runtime path for EyeLike, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `EyeLike.forward_` performs shape-only inference for EyeLike, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        if len(input.size) != 2:
            raise ValueError(f"EyeLike expects a 2-D input, got shape {input.size}")
        return {"tensor": Tensor_(*input.size, dtype=self.dtype or input.dtype), "parameters": None, "graph": None}
    
class Clip(Ops):
    # Egor Izmaylov: Function `Clip.__init__` initializes Clip, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Clip, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        
        if self.lib:
             self.lib.clip_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor), 
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor)
            ]

    # Egor Izmaylov: Function `Clip.forward` executes the concrete runtime path for Clip, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input: Tensor, min_val: Tensor = None, max_val: Tensor = None) -> dict:
        # 1. 准备广播列表
        # 注意：ONNX 中 min/max 是可选的，可能为 None
        broadcast_list = [input.data]
        if min_val is not None: broadcast_list.append(min_val.data)
        if max_val is not None: broadcast_list.append(max_val.data)

        # 2. 执行广播 (numpy 会自动处理标量与张量的广播)
        try:
            broadcasted = np.broadcast_arrays(*broadcast_list)
        except ValueError:
             raise ValueError(f"Clip inputs shape mismatch: input={input.size}, "
                              f"min={min_val.size if min_val else 'None'}, "
                              f"max={max_val.size if max_val else 'None'}")

        # 3. 提取广播后的数据
        # broadcasted[0] 对应 input
        input_data = np.ascontiguousarray(broadcasted[0])
        
        idx = 1
        min_data = None
        if min_val is not None:
            min_data = np.ascontiguousarray(broadcasted[idx])
            idx += 1
            
        max_data = None
        if max_val is not None:
            max_data = np.ascontiguousarray(broadcasted[idx])
        
        # 4. 准备 C Tensor
        # 此时 input_data, min_data, max_data 的 shape 应该完全一致
        out_shape = input_data.shape
        if self.lib is None or self.dtype not in nn.DTYPE_MAP:
            out_data = input_data
            if min_data is not None:
                out_data = np.maximum(out_data, min_data)
            if max_data is not None:
                out_data = np.minimum(out_data, max_data)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}
        
        input_c = self._numpy_to_ctensor(input_data, self.dtype) # 使用广播后的数据
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
        
        min_c = self._numpy_to_ctensor(min_data, min_val.dtype if min_val else "float32") if min_data is not None else ctypes.POINTER(nn.CTensor)()
        max_c = self._numpy_to_ctensor(max_data, max_val.dtype if max_val else "float32") if max_data is not None else ctypes.POINTER(nn.CTensor)()
        
        # 5. 调用 C 函数
        self.lib.clip_forward(input_c, output_c, min_c, max_c)
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        
        # 6. 资源释放
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)
        if min_data is not None: self.lib.free_tensor(min_c)
        if max_data is not None: self.lib.free_tensor(max_c)

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # Egor Izmaylov: Function `Clip.forward_` performs shape-only inference for Clip, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_, min_val: Tensor_ = None, max_val: Tensor_ = None) -> dict:
        # 图推断模式：简单假设输出形状与输入一致（实际应考虑广播）
        # 假设 min/max 是标量或可广播，输出 shape 由 input 主导
        try:
            shapes = [input.size]
            if min_val: shapes.append(min_val.size)
            if max_val: shapes.append(max_val.size)
            out_shape = np.broadcast_shapes(*shapes)
        except:
            out_shape = input.size
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}
    
class MatMul(Ops):
    # Egor Izmaylov: Function `MatMul.__init__` initializes MatMul, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(MatMul, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `MatMul.forward` executes the concrete runtime path for MatMul, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `MatMul.forward_` performs shape-only inference for MatMul, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
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
    
class Gather(Ops):
    # Egor Izmaylov: Function `Gather.__init__` initializes Gather, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `Gather.forward` executes the concrete runtime path for Gather, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Gather.forward_` performs shape-only inference for Gather, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
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
    
class Expand(Ops):
    # Egor Izmaylov: Function `Expand.__init__` initializes Expand, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super(Expand, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.expand_forward.argtypes = [ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor)]

    # Egor Izmaylov: Function `Expand.forward` executes the concrete runtime path for Expand, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Expand.forward_` performs shape-only inference for Expand, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
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
    
class Shape(Ops):
    # Egor Izmaylov: Function `Shape.__init__` initializes Shape, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, start=0, end=None, dtype="int64", version="17"):
        super(Shape, self).__init__(inputs, outputs)
        self.start = start
        self.end = end
        self.dtype = "int64" # Shape 输出永远是 int64
        self.version = version

    # Egor Izmaylov: Function `Shape.forward` executes the concrete runtime path for Shape, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input: Tensor) -> dict:
        dims = list(input.size)
        # 处理 start/end
        end = len(dims) if self.end is None else self.end
        sliced_dims = dims[self.start : end]
        
        out_data = np.array(sliced_dims, dtype=np.int64)
        out_tensor = Tensor(len(sliced_dims), dtype="int64", data=out_data)
        
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # Egor Izmaylov: Function `Shape.forward_` performs shape-only inference for Shape, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input: Tensor_) -> dict:
        # Shape 的输出形状取决于 input 的 rank
        dims = list(input.size)
        end = len(dims) if self.end is None else self.end
        out_len = len(dims[self.start : end])
        return {"tensor": Tensor_(out_len, dtype="int64"), "parameters": None, "graph": None}
    
class Constant(Ops):
    # Egor Izmaylov: Function `Constant.__init__` initializes Constant, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, value=None, dtype="float32", version="17"):
        super(Constant, self).__init__(inputs, outputs)
        self.value = value
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `Constant._value_array` centralizes the value array helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _value_array(self):
        if isinstance(self.value, Tensor):
            return np.asarray(self.value.data).copy(), tuple(self.value.size)
        if isinstance(self.value, np.ndarray):
            return np.asarray(self.value).copy(), self.value.shape
        arr = np.asarray(self.value, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, None))
        return arr.copy(), arr.shape

    # Egor Izmaylov: Function `Constant.forward` executes the concrete runtime path for Constant, consuming real tensor values and returning the graph-runner value contract.
    def forward(self) -> dict:
        out_data, val_shape = self._value_array()
        return {"tensor": Tensor(*val_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # Egor Izmaylov: Function `Constant.forward_` performs shape-only inference for Constant, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self) -> dict:
        # [Fix] 为了支持 Shape 推断，Constant 需要返回真实数据
        out_data, val_shape = self._value_array()
        return {"tensor": Tensor(*val_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}
    
class Equal(Ops):
    # Egor Izmaylov: Function `Equal.__init__` initializes Equal, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Equal.forward` executes the concrete runtime path for Equal, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "equal_forward"), "parameters": None}
    # Egor Izmaylov: Function `Equal.forward_` performs shape-only inference for Equal, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class Greater(Ops):
    # Egor Izmaylov: Function `Greater.__init__` initializes Greater, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Greater.forward` executes the concrete runtime path for Greater, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "greater_forward"), "parameters": None}
    # Egor Izmaylov: Function `Greater.forward_` performs shape-only inference for Greater, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class Less(Ops):
    # Egor Izmaylov: Function `Less.__init__` initializes Less, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Less.forward` executes the concrete runtime path for Less, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "less_forward"), "parameters": None}
    # Egor Izmaylov: Function `Less.forward_` performs shape-only inference for Less, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class GreaterOrEqual(Ops):
    # Egor Izmaylov: Function `GreaterOrEqual.__init__` initializes GreaterOrEqual, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `GreaterOrEqual.forward` executes the concrete runtime path for GreaterOrEqual, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "greater_or_equal_forward"), "parameters": None}
    # Egor Izmaylov: Function `GreaterOrEqual.forward_` performs shape-only inference for GreaterOrEqual, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class LessOrEqual(Ops):
    # Egor Izmaylov: Function `LessOrEqual.__init__` initializes LessOrEqual, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `LessOrEqual.forward` executes the concrete runtime path for LessOrEqual, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "less_or_equal_forward"), "parameters": None}
    # Egor Izmaylov: Function `LessOrEqual.forward_` performs shape-only inference for LessOrEqual, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class Not(Ops):
    # Egor Izmaylov: Function `Not.__init__` initializes Not, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Not.forward` executes the concrete runtime path for Not, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "not_forward"), "parameters": None}
    # Egor Izmaylov: Function `Not.forward_` performs shape-only inference for Not, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class And(Ops):
    # Egor Izmaylov: Function `And.__init__` initializes And, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `And.forward` executes the concrete runtime path for And, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "and_forward"), "parameters": None}
    # Egor Izmaylov: Function `And.forward_` performs shape-only inference for And, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class Or(Ops):
    # Egor Izmaylov: Function `Or.__init__` initializes Or, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Or.forward` executes the concrete runtime path for Or, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "or_forward"), "parameters": None}
    # Egor Izmaylov: Function `Or.forward_` performs shape-only inference for Or, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class Xor(Ops):
    # Egor Izmaylov: Function `Xor.__init__` initializes Xor, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Xor.forward` executes the concrete runtime path for Xor, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "xor_forward"), "parameters": None}
    # Egor Izmaylov: Function `Xor.forward_` performs shape-only inference for Xor, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class IsNaN(Ops):
    # Egor Izmaylov: Function `IsNaN.__init__` initializes IsNaN, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `IsNaN.forward` executes the concrete runtime path for IsNaN, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "isnan_forward"), "parameters": None}
    # Egor Izmaylov: Function `IsNaN.forward_` performs shape-only inference for IsNaN, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Sin(Ops):
    # Egor Izmaylov: Function `Sin.__init__` initializes Sin, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Sin.forward` executes the concrete runtime path for Sin, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "sin_forward"), "parameters": None}
    # Egor Izmaylov: Function `Sin.forward_` performs shape-only inference for Sin, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Tan(Ops):
    # Egor Izmaylov: Function `Tan.__init__` initializes Tan, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Tan.forward` executes the concrete runtime path for Tan, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "tan_forward"), "parameters": None}
    # Egor Izmaylov: Function `Tan.forward_` performs shape-only inference for Tan, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Atan(Ops):
    # Egor Izmaylov: Function `Atan.__init__` initializes Atan, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Atan.forward` executes the concrete runtime path for Atan, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "atan_forward"), "parameters": None}
    # Egor Izmaylov: Function `Atan.forward_` performs shape-only inference for Atan, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Sign(Ops):
    # Egor Izmaylov: Function `Sign.__init__` initializes Sign, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Sign.forward` executes the concrete runtime path for Sign, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "sign_forward"), "parameters": None}
    # Egor Izmaylov: Function `Sign.forward_` performs shape-only inference for Sign, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}
        
class Identity(Ops):
    # Egor Izmaylov: Function `Identity.__init__` initializes Identity, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.identity_forward.argtypes = [ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor)]
    # Egor Izmaylov: Function `Identity.forward` executes the concrete runtime path for Identity, consuming real tensor values and returning the graph-runner value contract.
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
    # Egor Izmaylov: Function `Identity.forward_` performs shape-only inference for Identity, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        if isinstance(x, Tensor_):
            return {"tensor": Tensor_(*x.size, dtype=self.dtype or x.dtype), "parameters": None}
        return {"tensor": x, "parameters": None}

class Mod(Ops):
    # Egor Izmaylov: Function `Mod.__init__` initializes Mod, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `Mod.forward` executes the concrete runtime path for Mod, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Mod.forward_` performs shape-only inference for Mod, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, a, b):
        shape = np.broadcast_shapes(a.size, b.size)
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class Where(Ops):
    # Egor Izmaylov: Function `Where.__init__` initializes Where, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.where_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor),
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor)
            ]
    # Egor Izmaylov: Function `Where.forward` executes the concrete runtime path for Where, consuming real tensor values and returning the graph-runner value contract.
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
    # Egor Izmaylov: Function `Where.forward_` performs shape-only inference for Where, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, cond, x, y):
        try: shape = np.broadcast_shapes(cond.size, x.size, y.size)
        except: shape = x.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}
    
class ConstantOfShape(Ops):
    # Egor Izmaylov: Function `ConstantOfShape.__init__` initializes ConstantOfShape, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `ConstantOfShape.forward` executes the concrete runtime path for ConstantOfShape, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `ConstantOfShape.forward_` performs shape-only inference for ConstantOfShape, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, shape_tensor):
        if hasattr(shape_tensor, "data") and shape_tensor.data is not None:
            target_shape = tuple(shape_tensor.data.astype(np.int64).flatten().tolist())
            return {"tensor": Tensor_(*target_shape, dtype=self.dtype), "parameters": None}
        return {"tensor": Tensor_(1, dtype=self.dtype), "parameters": None}

class Range(Ops):
    # Egor Izmaylov: Function `Range.__init__` initializes Range, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `Range.forward` executes the concrete runtime path for Range, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Range.forward_` performs shape-only inference for Range, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, start, limit, delta):
        if all(hasattr(t, "data") and t.data is not None for t in (start, limit, delta)):
            s = start.data.item()
            l = limit.data.item()
            d = delta.data.item()
            length = max(int(np.ceil((l - s) / d)), 0)
            return {"tensor": Tensor_(length, dtype=self.dtype), "parameters": None}
        return {"tensor": Tensor_(1, dtype=self.dtype), "parameters": None}

class Tile(Ops):
    # Egor Izmaylov: Function `Tile.__init__` initializes Tile, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.tile_forward.argtypes = [ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor)]

    # Egor Izmaylov: Function `Tile.forward` executes the concrete runtime path for Tile, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Tile.forward_` performs shape-only inference for Tile, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input, repeats):
        if hasattr(repeats, "data") and repeats.data is not None:
            rep = repeats.data.astype(np.int64).flatten()
            in_shape = np.array(input.size)
            if len(rep) == len(in_shape):
                out_shape = tuple((in_shape * rep).tolist())
                return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}
        return {"tensor": Tensor_(*input.size, dtype=self.dtype), "parameters": None}

class Pad(Ops):
    # Egor Izmaylov: Function `Pad.__init__` initializes Pad, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `Pad._calc_shape` centralizes the calc shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
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

    # Egor Izmaylov: Function `Pad.forward` executes the concrete runtime path for Pad, consuming real tensor values and returning the graph-runner value contract.
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
    
    # Egor Izmaylov: Function `Pad.forward_` performs shape-only inference for Pad, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, data, pads, constant_value=None):
        if hasattr(pads, "data") and pads.data is not None:
            out_shape = self._calc_shape(data.size, pads.data)
        else:
            out_shape = data.size
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}

class Split(Ops):
    # Egor Izmaylov: Function `Split.__init__` initializes Split, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `Split.forward` executes the concrete runtime path for Split, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Split.forward_` performs shape-only inference for Split, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
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
    
# Egor Izmaylov: Function `_sequence_position` centralizes the sequence position helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _sequence_position(position, length, default=None, allow_end=False):
    if position is None:
        if default is None:
            raise ValueError("Sequence position is required")
        pos = default
    else:
        pos = int(position.data.item())
    if pos < 0:
        pos += length
    upper = length if allow_end else length - 1
    if pos < 0 or pos > upper:
        raise IndexError(f"Sequence position {pos} is out of bounds for length {length}")
    return pos


class SequenceEmpty(Ops):
    # Egor Izmaylov: Function `SequenceEmpty.__init__` initializes SequenceEmpty, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `SequenceEmpty.forward` executes the concrete runtime path for SequenceEmpty, consuming real tensor values and returning the graph-runner value contract.
    def forward(self):
        return {"tensor": [], "parameters": None}

    # Egor Izmaylov: Function `SequenceEmpty.forward_` performs shape-only inference for SequenceEmpty, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self):
        return {"tensor": [], "parameters": None}


class SequenceConstruct(Ops):
    # Egor Izmaylov: Function `SequenceConstruct.__init__` initializes SequenceConstruct, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `SequenceConstruct.forward` executes the concrete runtime path for SequenceConstruct, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, *inputs):
        return {"tensor": list(inputs), "parameters": None}

    # Egor Izmaylov: Function `SequenceConstruct.forward_` performs shape-only inference for SequenceConstruct, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, *inputs):
        return {"tensor": list(inputs), "parameters": None}


class SequenceAt(Ops):
    # Egor Izmaylov: Function `SequenceAt.__init__` initializes SequenceAt, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `SequenceAt.forward` executes the concrete runtime path for SequenceAt, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input_sequence, position):
        return {"tensor": input_sequence[_sequence_position(position, len(input_sequence))], "parameters": None}

    # Egor Izmaylov: Function `SequenceAt.forward_` performs shape-only inference for SequenceAt, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input_sequence, position):
        return self.forward(input_sequence, position)


class SequenceInsert(Ops):
    # Egor Izmaylov: Function `SequenceInsert.__init__` initializes SequenceInsert, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `SequenceInsert.forward` executes the concrete runtime path for SequenceInsert, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input_sequence, tensor, position=None):
        output = list(input_sequence)
        pos = _sequence_position(position, len(output), default=len(output), allow_end=True)
        output.insert(pos, tensor)
        return {"tensor": output, "parameters": None}

    # Egor Izmaylov: Function `SequenceInsert.forward_` performs shape-only inference for SequenceInsert, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input_sequence, tensor, position=None):
        return self.forward(input_sequence, tensor, position)


class SequenceErase(Ops):
    # Egor Izmaylov: Function `SequenceErase.__init__` initializes SequenceErase, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `SequenceErase.forward` executes the concrete runtime path for SequenceErase, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input_sequence, position=None):
        output = list(input_sequence)
        pos = _sequence_position(position, len(output), default=len(output) - 1)
        del output[pos]
        return {"tensor": output, "parameters": None}

    # Egor Izmaylov: Function `SequenceErase.forward_` performs shape-only inference for SequenceErase, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input_sequence, position=None):
        return self.forward(input_sequence, position)


class SequenceLength(Ops):
    # Egor Izmaylov: Function `SequenceLength.__init__` initializes SequenceLength, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="int64", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = "int64"
        self.version = version

    # Egor Izmaylov: Function `SequenceLength.forward` executes the concrete runtime path for SequenceLength, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input_sequence):
        return {"tensor": Tensor(dtype=self.dtype, data=np.array(len(input_sequence), dtype=np.int64)), "parameters": None}

    # Egor Izmaylov: Function `SequenceLength.forward_` performs shape-only inference for SequenceLength, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input_sequence):
        if isinstance(input_sequence, list):
            return self.forward(input_sequence)
        return {"tensor": Tensor_(dtype=self.dtype), "parameters": None}


class ConcatFromSequence(Ops):
    # Egor Izmaylov: Function `ConcatFromSequence.__init__` initializes ConcatFromSequence, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axis=0, new_axis=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.new_axis = new_axis
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `ConcatFromSequence.forward` executes the concrete runtime path for ConcatFromSequence, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input_sequence):
        if not input_sequence:
            raise ValueError("ConcatFromSequence requires a non-empty sequence")
        arrays = [tensor.data for tensor in input_sequence]
        if self.new_axis:
            out_data = np.stack(arrays, axis=self.axis)
        else:
            out_data = np.concatenate(arrays, axis=self.axis)
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `ConcatFromSequence.forward_` performs shape-only inference for ConcatFromSequence, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input_sequence):
        if not input_sequence:
            return {"tensor": Tensor_(0, dtype=self.dtype), "parameters": None}
        shapes = [tuple(tensor.size) for tensor in input_sequence]
        if self.new_axis:
            axis = self.axis if self.axis >= 0 else self.axis + len(shapes[0]) + 1
            out_shape = list(shapes[0])
            out_shape.insert(axis, len(shapes))
        else:
            axis = self.axis if self.axis >= 0 else self.axis + len(shapes[0])
            out_shape = list(shapes[0])
            out_shape[axis] = sum(shape[axis] for shape in shapes)
        return {"tensor": Tensor_(*tuple(out_shape), dtype=self.dtype), "parameters": None}


class SplitToSequence(Ops):
    # Egor Izmaylov: Function `SplitToSequence.__init__` initializes SplitToSequence, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axis=0, keepdims=1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.keepdims = keepdims
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `SplitToSequence._split_sizes` centralizes the split sizes helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _split_sizes(self, axis_dim, split=None):
        if split is None:
            step = 1
            return [1] * axis_dim
        values = np.asarray(split.data).astype(np.int64).reshape(-1)
        if values.size == 1:
            step = int(values[0])
            if step <= 0:
                raise ValueError("SplitToSequence split values must be positive")
            return [min(step, axis_dim - start) for start in range(0, axis_dim, step)]
        sizes = values.astype(int).tolist()
        if any(size <= 0 for size in sizes) or sum(sizes) != axis_dim:
            raise ValueError("SplitToSequence 1-D split must contain positive sizes that sum to the axis dimension")
        return sizes

    # Egor Izmaylov: Function `SplitToSequence.forward` executes the concrete runtime path for SplitToSequence, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input, split=None):
        axis = self.axis if self.axis >= 0 else self.axis + len(input.size)
        sizes = self._split_sizes(input.size[axis], split)
        result = []
        start = 0
        for size in sizes:
            slc = [slice(None)] * len(input.size)
            slc[axis] = slice(start, start + size)
            data = input.data[tuple(slc)]
            if split is None and not self.keepdims:
                data = np.squeeze(data, axis=axis)
            result.append(Tensor(*data.shape, dtype=self.dtype, data=data.copy()))
            start += size
        return {"tensor": result, "parameters": None}

    # Egor Izmaylov: Function `SplitToSequence.forward_` performs shape-only inference for SplitToSequence, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input, split=None):
        axis = self.axis if self.axis >= 0 else self.axis + len(input.size)
        sizes = self._split_sizes(input.size[axis], split) if split is not None and hasattr(split, "data") and split.data is not None else [1] * input.size[axis]
        result = []
        for size in sizes:
            shape = list(input.size)
            shape[axis] = size
            if split is None and not self.keepdims:
                shape.pop(axis)
            result.append(Tensor_(*tuple(shape), dtype=self.dtype))
        return {"tensor": result, "parameters": None}

class Optional(Ops):
    # Egor Izmaylov: Function `Optional.__init__` initializes Optional, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `Optional.forward` executes the concrete runtime path for Optional, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input=None):
        return {"tensor": input, "parameters": None}

    # Egor Izmaylov: Function `Optional.forward_` performs shape-only inference for Optional, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input=None):
        return {"tensor": input, "parameters": None}


class OptionalGetElement(Ops):
    # Egor Izmaylov: Function `OptionalGetElement.__init__` initializes OptionalGetElement, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `OptionalGetElement.forward` executes the concrete runtime path for OptionalGetElement, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input):
        if input is None:
            raise ValueError("OptionalGetElement cannot read an empty optional")
        return {"tensor": input, "parameters": None}

    # Egor Izmaylov: Function `OptionalGetElement.forward_` performs shape-only inference for OptionalGetElement, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input):
        return self.forward(input)


class OptionalHasElement(Ops):
    # Egor Izmaylov: Function `OptionalHasElement.__init__` initializes OptionalHasElement, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = "bool"
        self.version = version

    # Egor Izmaylov: Function `OptionalHasElement.forward` executes the concrete runtime path for OptionalHasElement, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input):
        return {"tensor": Tensor(dtype=self.dtype, data=np.array(input is not None, dtype=np.bool_)), "parameters": None}

    # Egor Izmaylov: Function `OptionalHasElement.forward_` performs shape-only inference for OptionalHasElement, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input):
        if input is None:
            return self.forward(input)
        return {"tensor": Tensor_(dtype=self.dtype), "parameters": None}

# Reduce 基类，复用 Shape 计算逻辑
class ReduceBase(Ops):
    # Egor Izmaylov: Function `ReduceBase.__init__` initializes ReduceBase, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `ReduceBase._get_c_func_name` centralizes the get c func name helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _get_c_func_name(self):
        raise NotImplementedError

    # Egor Izmaylov: Function `ReduceBase._prepare_axes` centralizes the prepare axes helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
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

    # Egor Izmaylov: Function `ReduceBase._numpy_reduce` centralizes the numpy reduce helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
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

    # Egor Izmaylov: Function `ReduceBase._calc_out_shape` centralizes the calc out shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
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

    # Egor Izmaylov: Function `ReduceBase.forward` executes the concrete runtime path for ReduceBase, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `ReduceBase.forward_` performs shape-only inference for ReduceBase, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
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
    # Egor Izmaylov: Function `ReduceMean._get_c_func_name` centralizes the get c func name helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _get_c_func_name(self): return "reduce_mean_forward"

class ReduceSum(ReduceBase):
    # Egor Izmaylov: Function `ReduceSum._get_c_func_name` centralizes the get c func name helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _get_c_func_name(self): return "reduce_sum_forward"

class ReduceMax(ReduceBase):
    # Egor Izmaylov: Function `ReduceMax._get_c_func_name` centralizes the get c func name helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _get_c_func_name(self): return "reduce_max_forward"

class ReduceMin(ReduceBase):
    # Egor Izmaylov: Function `ReduceMin._get_c_func_name` centralizes the get c func name helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _get_c_func_name(self): return "reduce_min_forward"

class ReduceProd(ReduceBase):
    # Egor Izmaylov: Function `ReduceProd._get_c_func_name` centralizes the get c func name helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _get_c_func_name(self): return "reduce_prod_forward"
    
class ArgBase(Ops):
    # Egor Izmaylov: Function `ArgBase.__init__` initializes ArgBase, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `ArgBase._get_c_func_name` centralizes the get c func name helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _get_c_func_name(self): raise NotImplementedError

    # Egor Izmaylov: Function `ArgBase._arg_numpy` centralizes the arg numpy helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
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

    # Egor Izmaylov: Function `ArgBase.forward` executes the concrete runtime path for ArgBase, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `ArgBase.forward_` performs shape-only inference for ArgBase, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, data):
        ndim = len(data.size)
        axis = self.axis if self.axis >= 0 else self.axis + ndim
        out_shape = list(data.size)
        if self.keepdims: out_shape[axis] = 1
        else: out_shape.pop(axis)
        return {"tensor": Tensor_(*tuple(out_shape), dtype=self.dtype), "parameters": None}

class ArgMax(ArgBase):
    # Egor Izmaylov: Function `ArgMax._get_c_func_name` centralizes the get c func name helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _get_c_func_name(self): return "argmax_forward"

class ArgMin(ArgBase):
    # Egor Izmaylov: Function `ArgMin._get_c_func_name` centralizes the get c func name helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _get_c_func_name(self): return "argmin_forward"
    
class ScatterND(Ops):
    # Egor Izmaylov: Function `ScatterND.__init__` initializes ScatterND, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, reduction="none", dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.reduction = {"none": 0, "add": 1, "mul": 2}.get(reduction, 0)
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `ScatterND.forward` executes the concrete runtime path for ScatterND, consuming real tensor values and returning the graph-runner value contract.
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
    
    # Egor Izmaylov: Function `ScatterND.forward_` performs shape-only inference for ScatterND, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, data, indices, updates):
        return {"tensor": Tensor_(*data.size, dtype=self.dtype), "parameters": None}

class GatherND(Ops):
    # Egor Izmaylov: Function `GatherND.__init__` initializes GatherND, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, batch_dims=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.batch_dims = batch_dims
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `GatherND.forward` executes the concrete runtime path for GatherND, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `GatherND.forward_` performs shape-only inference for GatherND, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, data, indices):
        idx_shape = list(indices.size)
        data_shape = list(data.size)
        k = idx_shape[-1]
        out_shape = idx_shape[:-1] + data_shape[k + self.batch_dims:]
        return {"tensor": Tensor_(*tuple(out_shape), dtype=self.dtype), "parameters": None}

class GatherElements(Ops):
    # Egor Izmaylov: Function `GatherElements.__init__` initializes GatherElements, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axis=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `GatherElements.forward` executes the concrete runtime path for GatherElements, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `GatherElements.forward_` performs shape-only inference for GatherElements, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, data, indices):
        return {"tensor": Tensor_(*indices.size, dtype=self.dtype), "parameters": None}

class NonZero(Ops):
    # Egor Izmaylov: Function `NonZero.__init__` initializes NonZero, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="int64", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = "int64" # NonZero 必须返回 int64
        self.version = version

    # Egor Izmaylov: Function `NonZero.forward` executes the concrete runtime path for NonZero, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `NonZero.forward_` performs shape-only inference for NonZero, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input):
        count = int(np.count_nonzero(input.data)) if hasattr(input, "data") and input.data is not None else 1
        return {"tensor": Tensor_(len(input.size), count, dtype=self.dtype), "parameters": None}

class Resize(Ops):
    # Egor Izmaylov: Function `Resize.__init__` initializes Resize, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `Resize._should_use_reference` centralizes the should use reference helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
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

    # Egor Izmaylov: Function `Resize._run_reference_resize` centralizes the run reference resize helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
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
            feeds["scales"] = scales.data.astype(np.float32, copy=False)
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

    # Egor Izmaylov: Function `Resize.forward` executes the concrete runtime path for Resize, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, roi=None, scales=None, sizes=None):
        in_shape = np.array(x.size)
        
        # 参数解析逻辑
        if scales is not None and scales.data.size > 0:
            s = scales.data.flatten()
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

    # Egor Izmaylov: Function `Resize.forward_` performs shape-only inference for Resize, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, roi=None, scales=None, sizes=None):
        in_shape = np.array(x.size, dtype=np.int64)
        if sizes is not None and hasattr(sizes, "data") and sizes.data is not None and sizes.data.size > 0:
            out_shape = tuple(sizes.data.astype(np.int64).flatten().tolist())
            return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}
        if scales is not None and hasattr(scales, "data") and scales.data is not None and scales.data.size > 0:
            out_shape = tuple((in_shape * scales.data.astype(np.float64).flatten()).astype(np.int64).tolist())
            return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}
    
class TopK(Ops):
    # Egor Izmaylov: Function `TopK.__init__` initializes TopK, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `TopK.forward` executes the concrete runtime path for TopK, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, k_tensor):
        K = int(k_tensor.data.item())
        axis = self.axis if self.axis >= 0 else self.axis + len(x.size)
        
        out_shape = list(x.size)
        out_shape[axis] = K
        out_shape = tuple(out_shape)
        
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

    # Egor Izmaylov: Function `TopK.forward_` performs shape-only inference for TopK, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
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
    # Egor Izmaylov: Function `CumSum.__init__` initializes CumSum, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, exclusive=0, reverse=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.exclusive = exclusive
        self.reverse = reverse
        self.dtype = dtype
        self.version = version

    # Egor Izmaylov: Function `CumSum.forward` executes the concrete runtime path for CumSum, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `CumSum.forward_` performs shape-only inference for CumSum, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, axis_tensor):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class RandomUniformLike(Ops):
    # Egor Izmaylov: Function `RandomUniformLike.__init__` initializes RandomUniformLike, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, high=1.0, low=0.0, seed=0.0, dtype=None, version="17"):
        super().__init__(inputs, outputs)
        self.high = high
        self.low = low
        self.seed = seed
        self.dtype = dtype # None means infer from input, matching ONNX Like-op semantics.
        self.version = version

    # Egor Izmaylov: Function `RandomUniformLike.forward` executes the concrete runtime path for RandomUniformLike, consuming real tensor values and returning the graph-runner value contract.
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
            out_data = rng.uniform(self.low, self.high, size=input.size).astype(nn.DTYPE_TO_NUMPY.get(target_dtype, np.float32))
        out_tensor.data = out_data
        
        return {"tensor": out_tensor, "parameters": None}

    # Egor Izmaylov: Function `RandomUniformLike.forward_` performs shape-only inference for RandomUniformLike, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input):
        target_dtype = self.dtype if self.dtype else input.dtype
        return {"tensor": Tensor_(*input.size, dtype=target_dtype), "parameters": None}

class RandomUniform(Ops):
    # Egor Izmaylov: Function `RandomUniform.__init__` initializes RandomUniform, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, high=1.0, low=0.0, seed=0.0, dtype=1, shape=None, version="17"):
        super().__init__(inputs, outputs)
        self.high = high
        self.low = low
        self.seed = seed
        self.dtype = nn.onnx_dtype_mapping.get(dtype, "float32")
        self.shape_val = shape
        self.version = version

    # Egor Izmaylov: Function `RandomUniform.forward` executes the concrete runtime path for RandomUniform, consuming real tensor values and returning the graph-runner value contract.
    def forward(self):
        if self.shape_val is None:
            raise ValueError("RandomUniform requires 'shape' attribute")
        out_shape = tuple(self.shape_val)
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), DTYPE_MAP[self.dtype])
        self.lib.random_uniform_like_forward(output_c, ctypes.c_float(self.low), ctypes.c_float(self.high), ctypes.c_float(self.seed))
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(output_c)
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `RandomUniform.forward_` performs shape-only inference for RandomUniform, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self):
        out_shape = tuple(self.shape_val) if self.shape_val is not None else (1,)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}

class Multinomial(Ops):
    # Egor Izmaylov: Function `Multinomial.__init__` initializes Multinomial, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype=6, sample_size=1, seed=None, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = nn.onnx_dtype_mapping.get(dtype, "int32") if isinstance(dtype, int) else dtype
        self.sample_size = int(sample_size)
        self.seed = seed
        self.version = version
        if self.lib:
            self.lib.multinomial_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_uint32
            ]

    # Egor Izmaylov: Function `Multinomial.forward` executes the concrete runtime path for Multinomial, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input):
        probs = np.asarray(input.data, dtype=np.float64)
        if probs.ndim != 2:
            raise ValueError(f"Multinomial expects rank-2 input, got shape {input.size}")
        if self.sample_size < 0:
            raise ValueError(f"Multinomial sample_size must be non-negative, got {self.sample_size}")
        if np.any(probs < 0):
            raise ValueError("Multinomial probabilities must be non-negative")
        if np.any(probs.sum(axis=1) <= 0):
            raise ValueError("Multinomial probabilities must have a positive sum")

        if self.lib is not None and input.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            out_shape = (probs.shape[0], self.sample_size)
            output_shape_c = (ctypes.c_int * 2)(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, 2, nn.DTYPE_MAP[self.dtype])
            seed = 0 if self.seed is None else int(self.seed)
            self.lib.multinomial_forward(input_c, output_c, ctypes.c_int(self.sample_size), ctypes.c_uint32(seed))
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        rng = np.random.default_rng(None if self.seed is None else int(self.seed))
        class_count = probs.shape[1]
        out = np.empty((probs.shape[0], self.sample_size), dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, np.int32))
        for row_idx, row in enumerate(probs):
            total = row.sum()
            out[row_idx] = rng.choice(class_count, size=self.sample_size, replace=True, p=row / total)
        return {"tensor": Tensor(*out.shape, dtype=self.dtype, data=out), "parameters": None}

    # Egor Izmaylov: Function `Multinomial.forward_` performs shape-only inference for Multinomial, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input):
        if len(input.size) != 2:
            raise ValueError(f"Multinomial expects rank-2 input, got shape {input.size}")
        return {"tensor": Tensor_(input.size[0], self.sample_size, dtype=self.dtype), "parameters": None}

class NegativeLogLikelihoodLoss(Ops):
    # Egor Izmaylov: Function `NegativeLogLikelihoodLoss.__init__` initializes NegativeLogLikelihoodLoss, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, reduction="mean", ignore_index=None, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.negative_log_likelihood_loss_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int64,
            ]

    # Egor Izmaylov: Function `NegativeLogLikelihoodLoss._reduction_code` centralizes the reduction code helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _reduction_code(self):
        if self.reduction == "none":
            return 0
        if self.reduction == "mean":
            return 1
        if self.reduction == "sum":
            return 2
        raise ValueError(f"Unsupported loss reduction {self.reduction!r}")

    # Egor Izmaylov: Function `NegativeLogLikelihoodLoss._target_shape` centralizes the target shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _target_shape(input_shape):
        if len(input_shape) < 2:
            raise ValueError(f"Loss input expects rank >= 2, got shape {input_shape}")
        return (input_shape[0],) + tuple(input_shape[2:])

    # Egor Izmaylov: Function `NegativeLogLikelihoodLoss._gather_negative_scores` centralizes the gather negative scores helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _gather_negative_scores(log_probs, target, ignore_index):
        input_shape = log_probs.shape
        target_shape = target.shape
        n, c = input_shape[0], input_shape[1]
        reshaped = log_probs.reshape((n, c, -1))
        target_2d = target.reshape((n, -1))
        loss_2d = np.zeros((n, target_2d.shape[1]), dtype=log_probs.dtype)
        for i in range(n):
            for j in range(target_2d.shape[1]):
                cls = int(target_2d[i, j])
                if ignore_index is not None and cls == ignore_index:
                    continue
                if cls < 0 or cls >= c:
                    raise ValueError(f"Target class {cls} is out of range [0, {c})")
                loss_2d[i, j] = -reshaped[i, cls, j]
        return loss_2d.reshape(target_shape)

    # Egor Izmaylov: Function `NegativeLogLikelihoodLoss._reduce` centralizes the reduce helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _reduce(self, loss, target, weight=None):
        gather_weight = None
        if weight is not None:
            target_i = np.asarray(target, dtype=np.int64)
            clipped = np.clip(target_i, 0, len(weight.data) - 1)
            gather_weight = np.take(weight.data, clipped).astype(loss.dtype, copy=False)
            if self.ignore_index is not None:
                gather_weight = np.where(target_i == self.ignore_index, 0, gather_weight).astype(loss.dtype, copy=False)
        elif self.ignore_index is not None:
            gather_weight = np.where(target == self.ignore_index, 0, 1).astype(loss.dtype, copy=False)

        if gather_weight is not None:
            loss = loss * gather_weight
            if self.reduction == "mean":
                denom = gather_weight.sum()
                return loss.sum() / denom

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        raise ValueError(f"Unsupported loss reduction {self.reduction!r}")

    # Egor Izmaylov: Function `NegativeLogLikelihoodLoss.forward` executes the concrete runtime path for NegativeLogLikelihoodLoss, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input, target, weight=None):
        data = np.asarray(input.data)
        labels = np.asarray(target.data)
        expected_target_shape = self._target_shape(data.shape)
        if labels.shape != expected_target_shape:
            raise ValueError(f"Target shape {labels.shape} does not match expected {expected_target_shape}")
        invalid = (labels < 0) | (labels >= data.shape[1])
        if self.ignore_index is not None:
            invalid = invalid & (labels != self.ignore_index)
        if np.any(invalid):
            raise ValueError(f"Target class is out of range [0, {data.shape[1]})")

        if (
            self.lib is not None
            and input.dtype in nn.DTYPE_MAP
            and target.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
            and (weight is None or weight.dtype in nn.DTYPE_MAP)
        ):
            out_shape = target.size if self.reduction == "none" else ()
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            target_c = self._numpy_to_ctensor(np.ascontiguousarray(target.data), target.dtype)
            weight_c = self._numpy_to_ctensor(np.ascontiguousarray(weight.data), weight.dtype) if weight is not None else None
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.negative_log_likelihood_loss_forward(
                input_c,
                target_c,
                weight_c,
                output_c,
                ctypes.c_int(self._reduction_code()),
                ctypes.c_int(1 if self.ignore_index is not None else 0),
                ctypes.c_int64(0 if self.ignore_index is None else int(self.ignore_index)),
            )
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(target_c)
            if weight_c is not None:
                self.lib.free_tensor(weight_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        loss = self._gather_negative_scores(data, labels, self.ignore_index)
        reduced = self._reduce(loss, labels, weight)
        out_data = np.asarray(reduced, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `NegativeLogLikelihoodLoss.forward_` performs shape-only inference for NegativeLogLikelihoodLoss, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input, target, weight=None):
        out_shape = target.size if self.reduction == "none" else ()
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}

class SoftmaxCrossEntropyLoss(NegativeLogLikelihoodLoss):
    # Egor Izmaylov: Function `SoftmaxCrossEntropyLoss.__init__` initializes SoftmaxCrossEntropyLoss, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, reduction="mean", ignore_index=None, dtype="float32", version="17"):
        super().__init__(inputs, outputs, reduction=reduction, ignore_index=ignore_index, dtype=dtype, version=version)
        if self.lib:
            self.lib.softmax_cross_entropy_loss_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int64,
            ]

    # Egor Izmaylov: Function `SoftmaxCrossEntropyLoss._log_softmax` centralizes the log softmax helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _log_softmax(scores):
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

    # Egor Izmaylov: Function `SoftmaxCrossEntropyLoss.forward` executes the concrete runtime path for SoftmaxCrossEntropyLoss, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, scores, labels, weights=None):
        data = np.asarray(scores.data)
        target = np.asarray(labels.data)
        expected_target_shape = self._target_shape(data.shape)
        if target.shape != expected_target_shape:
            raise ValueError(f"Label shape {target.shape} does not match expected {expected_target_shape}")
        invalid = (target < 0) | (target >= data.shape[1])
        if self.ignore_index is not None:
            invalid = invalid & (target != self.ignore_index)
        if np.any(invalid):
            raise ValueError(f"Target class is out of range [0, {data.shape[1]})")
        out_dtype = nn.DTYPE_TO_NUMPY.get(self.dtype, data.dtype)
        if (
            self.lib is not None
            and scores.dtype in nn.DTYPE_MAP
            and labels.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
            and (weights is None or weights.dtype in nn.DTYPE_MAP)
        ):
            loss_shape = labels.size if self.reduction == "none" else ()
            scores_c = self._numpy_to_ctensor(np.ascontiguousarray(scores.data), scores.dtype)
            labels_c = self._numpy_to_ctensor(np.ascontiguousarray(labels.data), labels.dtype)
            weights_c = self._numpy_to_ctensor(np.ascontiguousarray(weights.data), weights.dtype) if weights is not None else None
            loss_shape_c = (ctypes.c_int * len(loss_shape))(*loss_shape)
            loss_c = self.lib.create_tensor(loss_shape_c, len(loss_shape), nn.DTYPE_MAP[self.dtype])
            log_c = None
            want_log_prob = len(self.outputs) > 1 and self.outputs[1]
            if want_log_prob:
                log_shape_c = (ctypes.c_int * len(scores.size))(*scores.size)
                log_c = self.lib.create_tensor(log_shape_c, len(scores.size), nn.DTYPE_MAP[self.dtype])
            self.lib.softmax_cross_entropy_loss_forward(
                scores_c,
                labels_c,
                weights_c,
                loss_c,
                log_c,
                ctypes.c_int(self._reduction_code()),
                ctypes.c_int(1 if self.ignore_index is not None else 0),
                ctypes.c_int64(0 if self.ignore_index is None else int(self.ignore_index)),
            )
            loss_data = self._ctensor_to_numpy(loss_c, self.dtype)
            loss_tensor = Tensor(*loss_shape, dtype=self.dtype, data=loss_data)
            log_tensor = None
            if want_log_prob:
                log_data = self._ctensor_to_numpy(log_c, self.dtype)
                log_tensor = Tensor(*scores.size, dtype=self.dtype, data=log_data)
            self.lib.free_tensor(scores_c)
            self.lib.free_tensor(labels_c)
            if weights_c is not None:
                self.lib.free_tensor(weights_c)
            self.lib.free_tensor(loss_c)
            if log_c is not None:
                self.lib.free_tensor(log_c)
            if want_log_prob:
                return {"tensor": (loss_tensor, log_tensor), "parameters": None}
            return {"tensor": loss_tensor, "parameters": None}

        log_prob = self._log_softmax(data)
        loss = self._gather_negative_scores(log_prob, target, self.ignore_index)
        reduced = self._reduce(loss, target, weights)
        loss_tensor = Tensor(*np.asarray(reduced).shape, dtype=self.dtype, data=np.asarray(reduced, dtype=out_dtype))
        if len(self.outputs) > 1 and self.outputs[1]:
            log_tensor = Tensor(*log_prob.shape, dtype=self.dtype, data=log_prob.astype(out_dtype, copy=False))
            return {"tensor": (loss_tensor, log_tensor), "parameters": None}
        return {"tensor": loss_tensor, "parameters": None}

    # Egor Izmaylov: Function `SoftmaxCrossEntropyLoss.forward_` performs shape-only inference for SoftmaxCrossEntropyLoss, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, scores, labels, weights=None):
        loss_shape = labels.size if self.reduction == "none" else ()
        loss_tensor = Tensor_(*loss_shape, dtype=self.dtype)
        if len(self.outputs) > 1 and self.outputs[1]:
            return {"tensor": (loss_tensor, Tensor_(*scores.size, dtype=self.dtype)), "parameters": None}
        return {"tensor": loss_tensor, "parameters": None}

class MelWeightMatrix(Ops):
    # Egor Izmaylov: Function `MelWeightMatrix.__init__` initializes MelWeightMatrix, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, output_datatype=1, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = nn.onnx_dtype_mapping.get(output_datatype, "float32") if isinstance(output_datatype, int) else output_datatype
        self.version = version
        if self.lib:
            self.lib.mel_weight_matrix_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
            ]

    # Egor Izmaylov: Function `MelWeightMatrix._scalar` centralizes the scalar helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _scalar(value):
        return np.asarray(value.data).item()

    # Egor Izmaylov: Function `MelWeightMatrix._mel` centralizes the mel helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _mel(frequency):
        return 2595.0 * np.log10(1.0 + frequency / 700.0)

    # Egor Izmaylov: Function `MelWeightMatrix._hz` centralizes the hz helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _hz(mel):
        return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)

    # Egor Izmaylov: Function `MelWeightMatrix.forward` executes the concrete runtime path for MelWeightMatrix, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz):
        bins = int(self._scalar(num_mel_bins))
        dft_len = int(self._scalar(dft_length))
        rate = int(self._scalar(sample_rate))
        lower = float(self._scalar(lower_edge_hertz))
        upper = float(self._scalar(upper_edge_hertz))
        if bins < 0 or dft_len < 0 or rate <= 0 or upper < lower:
            raise ValueError("Invalid MelWeightMatrix parameters")

        num_spectrogram_bins = dft_len // 2 + 1
        if self.lib is not None and self.dtype in nn.DTYPE_MAP:
            output_shape_c = (ctypes.c_int * 2)(num_spectrogram_bins, bins)
            output_c = self.lib.create_tensor(output_shape_c, 2, nn.DTYPE_MAP[self.dtype])
            scalar_ctensors = [
                self._numpy_to_ctensor(np.ascontiguousarray(np.asarray(tensor.data).reshape(())), tensor.dtype)
                for tensor in (num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)
            ]
            self.lib.mel_weight_matrix_forward(*scalar_ctensors, output_c)
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            for c_tensor in scalar_ctensors:
                self.lib.free_tensor(c_tensor)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

        mel_points = np.linspace(self._mel(lower), self._mel(upper), bins + 2)
        hz_points = self._hz(mel_points)
        frequency_bins = np.floor((dft_len + 1) * hz_points / rate).astype(int)

        output = np.zeros((num_spectrogram_bins, bins), dtype=np.float64)
        for i in range(bins):
            left, center, right = frequency_bins[i], frequency_bins[i + 1], frequency_bins[i + 2]
            left = max(left, 0)
            center = min(max(center, 0), num_spectrogram_bins - 1)
            right = min(max(right, 0), num_spectrogram_bins)
            if center == left:
                output[center, i] = 1.0
            else:
                for j in range(left, center + 1):
                    output[j, i] = float(j - left) / float(center - left)
            if right > center:
                for j in range(center, right):
                    output[j, i] = float(right - j) / float(right - center)

        out_dtype = nn.DTYPE_TO_NUMPY.get(self.dtype, np.float32)
        out_data = output.astype(out_dtype)
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `MelWeightMatrix.forward_` performs shape-only inference for MelWeightMatrix, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz):
        if isinstance(num_mel_bins, Tensor) and isinstance(dft_length, Tensor):
            out_shape = (int(num_mel_bins.data.item()) // 1, int(dft_length.data.item()) // 2 + 1)
            out_shape = (out_shape[1], out_shape[0])
        else:
            out_shape = (1, 1)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}

class DFT(Ops):
    # Egor Izmaylov: Function `DFT.__init__` initializes DFT, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axis=1, inverse=0, onesided=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.inverse = bool(inverse)
        self.onesided = bool(onesided)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.dft_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]

    # Egor Izmaylov: Function `DFT._optional_length` centralizes the optional length helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _optional_length(dft_length, default):
        if dft_length is None:
            return int(default)
        return int(np.asarray(dft_length.data).item())

    # Egor Izmaylov: Function `DFT._as_complex` centralizes the as complex helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _as_complex(data):
        if data.shape[-1] == 1:
            return np.squeeze(data, axis=-1).astype(np.complex128)
        if data.shape[-1] == 2:
            return data[..., 0].astype(np.complex128) + 1j * data[..., 1].astype(np.complex128)
        raise ValueError(f"DFT expects the last dimension to be 1 or 2, got {data.shape[-1]}")

    # Egor Izmaylov: Function `DFT._from_complex` centralizes the from complex helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _from_complex(data, dtype, real_only=False):
        if real_only:
            out = np.real(data)[..., np.newaxis]
        else:
            out = np.stack([np.real(data), np.imag(data)], axis=-1)
        return out.astype(nn.DTYPE_TO_NUMPY.get(dtype, np.float32), copy=False)

    # Egor Izmaylov: Function `DFT._output_shape` centralizes the output shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _output_shape(self, input_shape, dft_length=None):
        if len(input_shape) < 1:
            raise ValueError("DFT expects rank >= 1")
        if input_shape[-1] not in (1, 2):
            raise ValueError(f"DFT expects the last dimension to be 1 or 2, got {input_shape[-1]}")
        axis = self.axis % len(input_shape)
        if axis == len(input_shape) - 1:
            raise ValueError("DFT axis cannot refer to the trailing complex component dimension")
        default_len = 2 * (input_shape[axis] - 1) if self.inverse and self.onesided else input_shape[axis]
        fft_len = int(default_len if dft_length is None else dft_length)
        out_shape = list(input_shape)
        if self.inverse:
            out_shape[axis] = fft_len
            out_shape[-1] = 1 if self.onesided else 2
        else:
            out_shape[axis] = fft_len // 2 + 1 if self.onesided else fft_len
            out_shape[-1] = 2
        return tuple(out_shape)

    # Egor Izmaylov: Function `DFT.forward` executes the concrete runtime path for DFT, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input, dft_length=None):
        data = np.asarray(input.data)
        axis = self.axis % data.ndim
        if axis == data.ndim - 1:
            raise ValueError("DFT axis cannot refer to the trailing complex component dimension")
        default_len = 2 * (data.shape[axis] - 1) if self.inverse and self.onesided else data.shape[axis]
        fft_len = self._optional_length(dft_length, default_len)
        out_shape = self._output_shape(data.shape, fft_len)
        if self.lib is not None and input.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.dft_forward(
                input_c,
                output_c,
                ctypes.c_int(self.axis),
                ctypes.c_int(int(self.inverse)),
                ctypes.c_int(int(self.onesided)),
                ctypes.c_int(fft_len),
            )
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        complex_data = self._as_complex(data)
        if self.inverse:
            if self.onesided:
                transformed = np.fft.irfft(complex_data, n=fft_len, axis=axis)
                out_data = self._from_complex(transformed, self.dtype, real_only=True)
            else:
                transformed = np.fft.ifft(complex_data, n=fft_len, axis=axis)
                out_data = self._from_complex(transformed, self.dtype)
        else:
            transformed = np.fft.fft(complex_data, n=fft_len, axis=axis)
            out_data = self._from_complex(transformed, self.dtype)
            if self.onesided:
                slices = [slice(None)] * out_data.ndim
                slices[axis] = slice(0, fft_len // 2 + 1)
                out_data = out_data[tuple(slices)]
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `DFT.forward_` performs shape-only inference for DFT, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input, dft_length=None):
        length = None
        if isinstance(dft_length, Tensor):
            length = int(dft_length.data.item())
        return {"tensor": Tensor_(*self._output_shape(input.size, length), dtype=self.dtype), "parameters": None}

class STFT(Ops):
    # Egor Izmaylov: Function `STFT.__init__` initializes STFT, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, onesided=1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.onesided = bool(onesided)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.stft_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]

    # Egor Izmaylov: Function `STFT._scalar` centralizes the scalar helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _scalar(value):
        return int(np.asarray(value.data).item())

    # Egor Izmaylov: Function `STFT._frame_length` centralizes the frame length helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _frame_length(signal, window=None, frame_length=None):
        if frame_length is not None:
            return STFT._scalar(frame_length)
        if window is not None:
            return int(window.data.shape[0])
        return int(signal.data.shape[-2])

    # Egor Izmaylov: Function `STFT.forward` executes the concrete runtime path for STFT, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, signal, frame_step, window=None, frame_length=None):
        data = np.asarray(signal.data)
        if data.ndim < 2 or data.shape[-1] not in (1, 2):
            raise ValueError(f"STFT expects input shape [..., signal_length, 1|2], got {signal.size}")
        step = self._scalar(frame_step)
        length = self._frame_length(signal, window, frame_length)
        if step <= 0 or length <= 0:
            raise ValueError("STFT frame_step and frame_length must be positive")
        n_frames = 1 + (data.shape[-2] - length) // step
        if n_frames < 0:
            raise ValueError("STFT frame_length cannot exceed signal length")
        bins = length // 2 + 1 if self.onesided else length
        out_shape = tuple(data.shape[:-2]) + (n_frames, bins, 2)
        if self.lib is not None and signal.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            signal_c = self._numpy_to_ctensor(np.ascontiguousarray(signal.data), signal.dtype)
            window_c = None
            if window is not None:
                if window.dtype not in nn.DTYPE_MAP:
                    self.lib.free_tensor(signal_c)
                    window_c = None
                else:
                    window_c = self._numpy_to_ctensor(np.ascontiguousarray(window.data), window.dtype)
            if window is None or window_c is not None:
                output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
                output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
                self.lib.stft_forward(
                    signal_c,
                    window_c,
                    output_c,
                    ctypes.c_int(step),
                    ctypes.c_int(length),
                    ctypes.c_int(int(self.onesided)),
                )
                out_data = self._ctensor_to_numpy(output_c, self.dtype)
                self.lib.free_tensor(signal_c)
                if window_c is not None:
                    self.lib.free_tensor(window_c)
                self.lib.free_tensor(output_c)
                return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        frame_list = []
        for frame_idx in range(n_frames):
            start = frame_idx * step
            stop = start + length
            frame = data[..., start:stop, :]
            if frame.shape[-2] < length:
                pad_width = [(0, 0)] * frame.ndim
                pad_width[-2] = (0, length - frame.shape[-2])
                frame = np.pad(frame, pad_width, mode="constant")
            frame_list.append(frame)

        frames = np.stack(frame_list, axis=-3)
        if window is None:
            win = np.ones((length,), dtype=data.dtype)
        else:
            win = np.asarray(window.data, dtype=data.dtype)
        window_shape = (1,) * (frames.ndim - 2) + (length, 1)
        weighted = frames * win.reshape(window_shape)
        out_data = DFT([], [], axis=-2, onesided=int(self.onesided), dtype=self.dtype).forward(
            Tensor(*weighted.shape, dtype=signal.dtype, data=weighted)
        )["tensor"].data
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `STFT.forward_` performs shape-only inference for STFT, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, signal, frame_step, window=None, frame_length=None):
        step = int(frame_step.data.item()) if isinstance(frame_step, Tensor) else 1
        if isinstance(frame_length, Tensor):
            length = int(frame_length.data.item())
        elif isinstance(window, Tensor):
            length = int(window.data.shape[0])
        elif hasattr(window, "size") and window is not None:
            length = int(window.size[0])
        else:
            length = int(signal.size[-2])
        n_frames = 1 + (signal.size[-2] - length) // step
        bins = length // 2 + 1 if self.onesided else length
        out_shape = tuple(signal.size[:-2]) + (n_frames, bins, 2)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}

class Unique(Ops):
    # Egor Izmaylov: Function `Unique.__init__` initializes Unique, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axis=None, sorted=1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.sorted = sorted
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.unique_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
            ]
            self.lib.unique_forward.restype = ctypes.c_int

    # Egor Izmaylov: Function `Unique._reorder_unique` centralizes the reorder unique helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _reorder_unique(self, values, indices, inverse, counts, axis=None):
        if self.sorted:
            return values, indices, inverse, counts
        order = np.argsort(indices)
        remap = np.empty_like(order)
        remap[order] = np.arange(order.size)
        if axis is None:
            values = values[order]
        else:
            values = np.take(values, order, axis=axis)
        return values, indices[order], remap[inverse], counts[order]

    # Egor Izmaylov: Function `Unique._compute` centralizes the compute helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _compute(self, x):
        data = x.data
        if self.axis is None:
            values, indices, inverse, counts = np.unique(
                data.reshape(-1), return_index=True, return_inverse=True, return_counts=True
            )
            values, indices, inverse, counts = self._reorder_unique(values, indices, inverse, counts)
        else:
            axis = self.axis if self.axis >= 0 else self.axis + data.ndim
            values, indices, inverse, counts = np.unique(
                data, axis=axis, return_index=True, return_inverse=True, return_counts=True
            )
            values, indices, inverse, counts = self._reorder_unique(values, indices, inverse, counts, axis=axis)
        return (
            values.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, values.dtype), copy=False),
            indices.astype(np.int64, copy=False),
            inverse.astype(np.int64, copy=False),
            counts.astype(np.int64, copy=False),
        )

    # Egor Izmaylov: Function `Unique.forward` executes the concrete runtime path for Unique, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        if self.axis is None and self.lib is not None and x.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            flat = np.ascontiguousarray(np.asarray(x.data).reshape(-1))
            n = int(flat.size)
            input_c = self._numpy_to_ctensor(flat, x.dtype)
            buffer_shape_c = (ctypes.c_int * 1)(n)
            values_c = self.lib.create_tensor(buffer_shape_c, 1, nn.DTYPE_MAP[self.dtype])
            indices_c = self.lib.create_tensor(buffer_shape_c, 1, nn.DTYPE_MAP["int64"])
            inverse_c = self.lib.create_tensor(buffer_shape_c, 1, nn.DTYPE_MAP["int64"])
            counts_c = self.lib.create_tensor(buffer_shape_c, 1, nn.DTYPE_MAP["int64"])
            unique_count = int(self.lib.unique_forward(
                input_c,
                values_c,
                indices_c,
                inverse_c,
                counts_c,
                ctypes.c_int(int(self.sorted)),
            ))
            values = self._ctensor_to_numpy(values_c, self.dtype)[:unique_count]
            indices = self._ctensor_to_numpy(indices_c, "int64")[:unique_count]
            inverse = self._ctensor_to_numpy(inverse_c, "int64")[:n]
            counts = self._ctensor_to_numpy(counts_c, "int64")[:unique_count]
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(values_c)
            self.lib.free_tensor(indices_c)
            self.lib.free_tensor(inverse_c)
            self.lib.free_tensor(counts_c)
        else:
            values, indices, inverse, counts = self._compute(x)
        tensors = [
            Tensor(*values.shape, dtype=self.dtype, data=values),
            Tensor(*indices.shape, dtype="int64", data=indices),
            Tensor(*inverse.shape, dtype="int64", data=inverse),
            Tensor(*counts.shape, dtype="int64", data=counts),
        ]
        selected = [tensor for name, tensor in zip(self.outputs, tensors) if name]
        return {"tensor": selected[0] if len(selected) == 1 else tuple(selected), "parameters": None}

    # Egor Izmaylov: Function `Unique.forward_` performs shape-only inference for Unique, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        if hasattr(x, "data") and x.data is not None:
            return self.forward(x)
        if self.axis is None:
            unique_dim = x.data_size
            inverse_dim = x.data_size
            value_shape = (unique_dim,)
        else:
            axis = self.axis if self.axis >= 0 else self.axis + len(x.size)
            unique_dim = x.size[axis]
            inverse_dim = x.size[axis]
            value_shape = tuple(x.size)
        tensors = [
            Tensor_(*value_shape, dtype=self.dtype),
            Tensor_(unique_dim, dtype="int64"),
            Tensor_(inverse_dim, dtype="int64"),
            Tensor_(unique_dim, dtype="int64"),
        ]
        selected = [tensor for name, tensor in zip(self.outputs, tensors) if name]
        return {"tensor": selected[0] if len(selected) == 1 else tuple(selected), "parameters": None}

# Egor Izmaylov: Function `_nms_box_to_corners` centralizes the nms box to corners helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _nms_box_to_corners(box, center_point_box):
    if center_point_box:
        x_center, y_center, width, height = box
        y1 = y_center - height / 2.0
        x1 = x_center - width / 2.0
        y2 = y_center + height / 2.0
        x2 = x_center + width / 2.0
    else:
        y1, x1, y2, x2 = box
    ymin, ymax = sorted((float(y1), float(y2)))
    xmin, xmax = sorted((float(x1), float(x2)))
    return ymin, xmin, ymax, xmax


# Egor Izmaylov: Function `_nms_iou` centralizes the nms iou helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _nms_iou(box_a, box_b, center_point_box):
    ay1, ax1, ay2, ax2 = _nms_box_to_corners(box_a, center_point_box)
    by1, bx1, by2, bx2 = _nms_box_to_corners(box_b, center_point_box)
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter = inter_h * inter_w
    area_a = max(0.0, ay2 - ay1) * max(0.0, ax2 - ax1)
    area_b = max(0.0, by2 - by1) * max(0.0, bx2 - bx1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


class NonMaxSuppression(Ops):
    # Egor Izmaylov: Function `NonMaxSuppression.__init__` initializes NonMaxSuppression, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, center_point_box=0, dtype="int64", version="17"):
        super().__init__(inputs, outputs)
        self.center_point_box = center_point_box
        self.dtype = "int64"
        self.version = version
        if self.lib:
            self.lib.non_max_suppression_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int,
            ]
            self.lib.non_max_suppression_forward.restype = ctypes.c_int

    # Egor Izmaylov: Function `NonMaxSuppression.forward` executes the concrete runtime path for NonMaxSuppression, consuming real tensor values and returning the graph-runner value contract.
    def forward(
        self,
        boxes,
        scores,
        max_output_boxes_per_class=None,
        iou_threshold=None,
        score_threshold=None,
    ):
        boxes_data = boxes.data.astype(np.float32, copy=False)
        scores_data = scores.data.astype(np.float32, copy=False)
        if boxes_data.ndim != 3 or boxes_data.shape[2] != 4:
            raise ValueError(f"NonMaxSuppression boxes must have shape [batch, boxes, 4], got {boxes.size}")
        if scores_data.ndim != 3 or scores_data.shape[0] != boxes_data.shape[0] or scores_data.shape[2] != boxes_data.shape[1]:
            raise ValueError(f"NonMaxSuppression scores must have shape [batch, classes, boxes], got {scores.size}")

        max_output = 0 if max_output_boxes_per_class is None else int(max_output_boxes_per_class.data.item())
        iou = 0.0 if iou_threshold is None else float(iou_threshold.data.item())
        score_min = -np.inf if score_threshold is None else float(score_threshold.data.item())
        if (
            self.lib is not None
            and boxes.dtype in nn.DTYPE_MAP
            and scores.dtype in nn.DTYPE_MAP
            and max_output > 0
        ):
            max_rows = boxes_data.shape[0] * scores_data.shape[1] * min(max_output, boxes_data.shape[1])
            boxes_c = self._numpy_to_ctensor(np.ascontiguousarray(boxes_data.astype(nn.DTYPE_TO_NUMPY[boxes.dtype], copy=False)), boxes.dtype)
            scores_c = self._numpy_to_ctensor(np.ascontiguousarray(scores_data.astype(nn.DTYPE_TO_NUMPY[scores.dtype], copy=False)), scores.dtype)
            output_shape_c = (ctypes.c_int * 2)(max_rows, 3)
            output_c = self.lib.create_tensor(output_shape_c, 2, nn.DTYPE_MAP[self.dtype])
            count = int(self.lib.non_max_suppression_forward(
                boxes_c,
                scores_c,
                output_c,
                ctypes.c_int(max_output),
                ctypes.c_float(iou),
                ctypes.c_float(score_min),
                ctypes.c_int(int(self.center_point_box)),
            ))
            out_data = self._ctensor_to_numpy(output_c, self.dtype)[:count]
            self.lib.free_tensor(boxes_c)
            self.lib.free_tensor(scores_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

        selected = []
        if max_output > 0:
            for batch in range(scores_data.shape[0]):
                for cls in range(scores_data.shape[1]):
                    class_scores = scores_data[batch, cls]
                    candidate_indices = np.where(class_scores >= score_min)[0]
                    order = candidate_indices[np.argsort(-class_scores[candidate_indices], kind="mergesort")]
                    kept = []
                    for box_idx in order:
                        if len(kept) >= max_output:
                            break
                        box = boxes_data[batch, box_idx]
                        if all(_nms_iou(box, boxes_data[batch, kept_idx], self.center_point_box) <= iou for kept_idx in kept):
                            kept.append(int(box_idx))
                            selected.append([batch, cls, int(box_idx)])

        selected_arr = np.asarray(selected, dtype=np.int64).reshape(-1, 3)
        return {"tensor": Tensor(*selected_arr.shape, dtype=self.dtype, data=selected_arr), "parameters": None}

    # Egor Izmaylov: Function `NonMaxSuppression.forward_` performs shape-only inference for NonMaxSuppression, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, boxes, scores, max_output_boxes_per_class=None, iou_threshold=None, score_threshold=None):
        if hasattr(boxes, "data") and boxes.data is not None and hasattr(scores, "data") and scores.data is not None:
            return self.forward(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
        if max_output_boxes_per_class is not None and hasattr(max_output_boxes_per_class, "data") and max_output_boxes_per_class.data is not None:
            max_output = int(max_output_boxes_per_class.data.item())
        else:
            max_output = 0
        first_dim = int(scores.size[0] * scores.size[1] * max_output) if len(scores.size) >= 2 else 0
        return {"tensor": Tensor_(first_dim, 3, dtype=self.dtype), "parameters": None}

class Einsum(Ops):
    # Egor Izmaylov: Function `Einsum.__init__` initializes Einsum, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, equation, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.equation = equation
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.einsum_forward.argtypes = [
                ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.POINTER(CTensor),
                ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
            ]

    # Egor Izmaylov: Function `Einsum._parse_equation` centralizes the parse equation helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _parse_equation(self, shapes):
        equation = self.equation.replace(" ", "")
        if "->" in self.equation:
            lhs, rhs = equation.split("->")
            input_labels = lhs.split(",")
            output_labels = rhs
        else:
            input_labels = equation.split(",")
            counts = {}
            for label in "".join(input_labels):
                counts[label] = counts.get(label, 0) + 1
            output_labels = "".join(sorted(label for label, count in counts.items() if count == 1))

        if len(input_labels) != len(shapes):
            raise ValueError(f"Einsum: Equation expects {len(input_labels)} inputs, got {len(shapes)}")
        if "." in "".join(input_labels) + output_labels:
            raise ValueError("Einsum: Ellipsis parsing is not supported by the C stride planner")
        if len(set(output_labels)) != len(output_labels):
            raise ValueError(f"Einsum: Output labels cannot repeat: {output_labels}")
            
        # 收集所有唯一标签及其维度大小
        unique_labels = sorted(list(set("".join(input_labels) + output_labels)))
        unique_labels = [l for l in unique_labels if l.strip()] # 去除空格
        
        label_to_dim = {}
        for i, labels in enumerate(input_labels):
            labels = labels.strip()
            shape = shapes[i]
            if len(labels) != len(shape):
                raise ValueError(f"Einsum: Labels {labels} mismatch shape {shape}")
            for l, dim in zip(labels, shape):
                if l in label_to_dim and label_to_dim[l] != dim:
                    raise ValueError(f"Einsum: Label {l!r} has inconsistent dimensions {label_to_dim[l]} and {dim}")
                label_to_dim[l] = dim

        for label in output_labels:
            if label not in label_to_dim:
                raise ValueError(f"Einsum: Output label {label!r} does not appear in any input")
        
        # 生成 C 需要的 loop_limits
        loop_limits = [label_to_dim[l] for l in unique_labels]
        
        # 计算 Strides
        # 这是一个映射：Label -> Stride (在 Input X 中)
        # 如果 Label 不在 Input X 中，Stride = 0 (广播语义)
        
        # Egor Izmaylov: Function `Einsum._parse_equation.get_tensor_strides` implements the get tensor strides step for the ONNX operator runtime and shape-inference layer, normalizing inputs and returning the exact data or metadata contract expected downstream.
        def get_tensor_strides(shape):
            # 计算 contigous strides
            strides = []
            st = 1
            for d in reversed(shape):
                strides.append(st)
                st *= d
            return list(reversed(strides))

        input_strides_flat = []
        for i, labels in enumerate(input_labels):
            labels = labels.strip()
            native_strides = get_tensor_strides(shapes[i])
            # 映射到 unique_labels 顺序
            current_tensor_strides = []
            for u_label in unique_labels:
                if u_label in labels:
                    # Repeated labels in one operand select a diagonal; the offset
                    # stride is the sum of every matching axis stride.
                    stride = sum(native_strides[idx] for idx, label in enumerate(labels) if label == u_label)
                    current_tensor_strides.append(stride)
                else:
                    current_tensor_strides.append(0) # 广播/无关维度
            input_strides_flat.extend(current_tensor_strides)
            
        output_strides_flat = []
        native_out_strides = get_tensor_strides([label_to_dim[l] for l in output_labels])
        for u_label in unique_labels:
            if u_label in output_labels:
                idx = output_labels.index(u_label)
                output_strides_flat.append(native_out_strides[idx])
            else:
                output_strides_flat.append(0) # 归约维度
                
        # 计算输出形状
        out_shape = tuple([label_to_dim[l] for l in output_labels])
        
        return unique_labels, loop_limits, input_strides_flat, output_strides_flat, out_shape

    # Egor Izmaylov: Function `Einsum._forward_ij_jk_to_ik` centralizes the forward ij jk to ik helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _forward_ij_jk_to_ik(self, left, right):
        if len(left.size) != 2 or len(right.size) != 2:
            return None
        m, k = left.size
        k2, n = right.size
        if k != k2:
            raise ValueError(f"Einsum ij,jk->ik shape mismatch: {left.size} vs {right.size}")

        a = left.data.astype(np.float32, copy=False)
        b = right.data.astype(np.float32, copy=False)
        out = np.empty((m, n), dtype=np.float32)
        for i in range(m):
            for j in range(n):
                acc = np.float32(0.0)
                for kk in range(k):
                    acc = np.float32(acc + np.float32(a[i, kk] * b[kk, j]))
                out[i, j] = acc
        return out

    # Egor Izmaylov: Function `Einsum.forward` executes the concrete runtime path for Einsum, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, *inputs):
        equation = self.equation.replace(" ", "")
        out_data = None
        if self.lib is not None and self.dtype in nn.DTYPE_MAP and all(x.dtype in nn.DTYPE_MAP for x in inputs):
            try:
                _labels, loop_limits, input_strides, output_strides, out_shape = self._parse_equation([x.size for x in inputs])
                input_ctensors = [self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype) for x in inputs]
                input_array = (ctypes.POINTER(CTensor) * len(input_ctensors))(*input_ctensors)
                output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
                output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
                loop_limits_c = (ctypes.c_int * len(loop_limits))(*loop_limits)
                input_strides_c = (ctypes.c_int * len(input_strides))(*input_strides)
                output_strides_c = (ctypes.c_int * len(output_strides))(*output_strides)
                self.lib.einsum_forward(
                    input_array,
                    len(input_ctensors),
                    output_c,
                    len(loop_limits),
                    loop_limits_c,
                    input_strides_c,
                    output_strides_c,
                )
                out_data = self._ctensor_to_numpy(output_c, self.dtype)
                for c_tensor in input_ctensors:
                    self.lib.free_tensor(c_tensor)
                self.lib.free_tensor(output_c)
            except ValueError:
                out_data = None
        if out_data is None and equation == "ij,jk->ik" and len(inputs) == 2:
            out_data = self._forward_ij_jk_to_ik(inputs[0], inputs[1])
        if out_data is None:
            out_data = np.einsum(self.equation, *(x.data for x in inputs))
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `Einsum.forward_` performs shape-only inference for Einsum, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, *inputs):
        if not inputs:
            raise ValueError("Einsum requires at least one input")
        dummy_inputs = [np.empty(x.size, dtype=np.float32) for x in inputs]
        out_shape = np.einsum(self.equation, *dummy_inputs).shape
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}
    
class Elu(Ops):
    # Egor Izmaylov: Function `Elu.__init__` initializes Elu, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, alpha=1.0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.elu_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]

    # Egor Izmaylov: Function `Elu.forward` executes the concrete runtime path for Elu, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.elu_forward(x_c, out_c, ctypes.c_float(self.alpha))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # Egor Izmaylov: Function `Elu.forward_` performs shape-only inference for Elu, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Selu(Ops):
    # Egor Izmaylov: Function `Selu.__init__` initializes Selu, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, alpha=1.67326, gamma=1.0507, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.gamma = gamma
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.selu_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float, ctypes.c_float]

    # Egor Izmaylov: Function `Selu.forward` executes the concrete runtime path for Selu, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.selu_forward(x_c, out_c, ctypes.c_float(self.alpha), ctypes.c_float(self.gamma))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # Egor Izmaylov: Function `Selu.forward_` performs shape-only inference for Selu, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class LeakyRelu(Ops):
    # Egor Izmaylov: Function `LeakyRelu.__init__` initializes LeakyRelu, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, alpha=0.01, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.leaky_relu_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]

    # Egor Izmaylov: Function `LeakyRelu.forward` executes the concrete runtime path for LeakyRelu, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.leaky_relu_forward(x_c, out_c, ctypes.c_float(self.alpha))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # Egor Izmaylov: Function `LeakyRelu.forward_` performs shape-only inference for LeakyRelu, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class ThresholdedRelu(Ops):
    # Egor Izmaylov: Function `ThresholdedRelu.__init__` initializes ThresholdedRelu, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, alpha=1.0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.thresholded_relu_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]

    # Egor Izmaylov: Function `ThresholdedRelu.forward` executes the concrete runtime path for ThresholdedRelu, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.thresholded_relu_forward(x_c, out_c, ctypes.c_float(self.alpha))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # Egor Izmaylov: Function `ThresholdedRelu.forward_` performs shape-only inference for ThresholdedRelu, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class HardSigmoid(Ops):
    # Egor Izmaylov: Function `HardSigmoid.__init__` initializes HardSigmoid, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, alpha=0.2, beta=0.5, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.beta = beta
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.hard_sigmoid_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float, ctypes.c_float]

    # Egor Izmaylov: Function `HardSigmoid.forward` executes the concrete runtime path for HardSigmoid, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.hard_sigmoid_forward(x_c, out_c, ctypes.c_float(self.alpha), ctypes.c_float(self.beta))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # Egor Izmaylov: Function `HardSigmoid.forward_` performs shape-only inference for HardSigmoid, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Celu(Ops):
    # Egor Izmaylov: Function `Celu.__init__` initializes Celu, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, alpha=1.0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.celu_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]

    # Egor Izmaylov: Function `Celu.forward` executes the concrete runtime path for Celu, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.celu_forward(x_c, out_c, ctypes.c_float(self.alpha))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # Egor Izmaylov: Function `Celu.forward_` performs shape-only inference for Celu, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Shrink(Ops):
    # Egor Izmaylov: Function `Shrink.__init__` initializes Shrink, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, bias=0.0, lambd=0.5, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.bias = bias
        self.lambd = lambd
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.shrink_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float, ctypes.c_float]

    # Egor Izmaylov: Function `Shrink.forward` executes the concrete runtime path for Shrink, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.shrink_forward(x_c, out_c, ctypes.c_float(self.bias), ctypes.c_float(self.lambd))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # Egor Izmaylov: Function `Shrink.forward_` performs shape-only inference for Shrink, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Softplus(Ops):
    # Egor Izmaylov: Function `Softplus.__init__` initializes Softplus, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Softplus.forward` executes the concrete runtime path for Softplus, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x): return {"tensor": self._execute_unary(x, "softplus_forward"), "parameters": None}
    # Egor Izmaylov: Function `Softplus.forward_` performs shape-only inference for Softplus, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Softsign(Ops):
    # Egor Izmaylov: Function `Softsign.__init__` initializes Softsign, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Softsign.forward` executes the concrete runtime path for Softsign, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x): return {"tensor": self._execute_unary(x, "softsign_forward"), "parameters": None}
    # Egor Izmaylov: Function `Softsign.forward_` performs shape-only inference for Softsign, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class HardSwish(Ops):
    # Egor Izmaylov: Function `HardSwish.__init__` initializes HardSwish, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `HardSwish.forward` executes the concrete runtime path for HardSwish, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x): return {"tensor": self._execute_unary(x, "hard_swish_forward"), "parameters": None}
    # Egor Izmaylov: Function `HardSwish.forward_` performs shape-only inference for HardSwish, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Acos(Ops):
    # Egor Izmaylov: Function `Acos.__init__` initializes Acos, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Acos.forward` executes the concrete runtime path for Acos, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x): return {"tensor": self._execute_unary(x, "acos_forward"), "parameters": None}
    # Egor Izmaylov: Function `Acos.forward_` performs shape-only inference for Acos, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Asin(Ops):
    # Egor Izmaylov: Function `Asin.__init__` initializes Asin, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Asin.forward` executes the concrete runtime path for Asin, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x): return {"tensor": self._execute_unary(x, "asin_forward"), "parameters": None}
    # Egor Izmaylov: Function `Asin.forward_` performs shape-only inference for Asin, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Cosh(Ops):
    # Egor Izmaylov: Function `Cosh.__init__` initializes Cosh, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Cosh.forward` executes the concrete runtime path for Cosh, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x): return {"tensor": self._execute_unary(x, "cosh_forward"), "parameters": None}
    # Egor Izmaylov: Function `Cosh.forward_` performs shape-only inference for Cosh, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Sinh(Ops):
    # Egor Izmaylov: Function `Sinh.__init__` initializes Sinh, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Sinh.forward` executes the concrete runtime path for Sinh, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x): return {"tensor": self._execute_unary(x, "sinh_forward"), "parameters": None}
    # Egor Izmaylov: Function `Sinh.forward_` performs shape-only inference for Sinh, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Asinh(Ops):
    # Egor Izmaylov: Function `Asinh.__init__` initializes Asinh, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Asinh.forward` executes the concrete runtime path for Asinh, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x): return {"tensor": self._execute_unary(x, "asinh_forward"), "parameters": None}
    # Egor Izmaylov: Function `Asinh.forward_` performs shape-only inference for Asinh, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Acosh(Ops):
    # Egor Izmaylov: Function `Acosh.__init__` initializes Acosh, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Acosh.forward` executes the concrete runtime path for Acosh, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x): return {"tensor": self._execute_unary(x, "acosh_forward"), "parameters": None}
    # Egor Izmaylov: Function `Acosh.forward_` performs shape-only inference for Acosh, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Atanh(Ops):
    # Egor Izmaylov: Function `Atanh.__init__` initializes Atanh, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Atanh.forward` executes the concrete runtime path for Atanh, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x): return {"tensor": self._execute_unary(x, "atanh_forward"), "parameters": None}
    # Egor Izmaylov: Function `Atanh.forward_` performs shape-only inference for Atanh, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class BitwiseAnd(Ops):
    # Egor Izmaylov: Function `BitwiseAnd.__init__` initializes BitwiseAnd, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="int32", version="18"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `BitwiseAnd.forward` executes the concrete runtime path for BitwiseAnd, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "bitwise_and_forward"), "parameters": None}
    # Egor Izmaylov: Function `BitwiseAnd.forward_` performs shape-only inference for BitwiseAnd, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class BitwiseOr(Ops):
    # Egor Izmaylov: Function `BitwiseOr.__init__` initializes BitwiseOr, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="int32", version="18"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `BitwiseOr.forward` executes the concrete runtime path for BitwiseOr, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "bitwise_or_forward"), "parameters": None}
    # Egor Izmaylov: Function `BitwiseOr.forward_` performs shape-only inference for BitwiseOr, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class BitwiseXor(Ops):
    # Egor Izmaylov: Function `BitwiseXor.__init__` initializes BitwiseXor, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="int32", version="18"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `BitwiseXor.forward` executes the concrete runtime path for BitwiseXor, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "bitwise_xor_forward"), "parameters": None}
    # Egor Izmaylov: Function `BitwiseXor.forward_` performs shape-only inference for BitwiseXor, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class BitwiseNot(Ops):
    # Egor Izmaylov: Function `BitwiseNot.__init__` initializes BitwiseNot, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="int32", version="18"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `BitwiseNot.forward` executes the concrete runtime path for BitwiseNot, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "bitwise_not_forward"), "parameters": None}
    # Egor Izmaylov: Function `BitwiseNot.forward_` performs shape-only inference for BitwiseNot, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class BitShift(Ops):
    # Egor Izmaylov: Function `BitShift.__init__` initializes BitShift, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `BitShift.forward` executes the concrete runtime path for BitShift, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, a, b):
        out_tensor = self._execute_binary_custom(a, b)
        return {"tensor": out_tensor, "parameters": None}

    # Egor Izmaylov: Function `BitShift._execute_binary_custom` centralizes the execute binary custom helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
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

    # Egor Izmaylov: Function `BitShift.forward_` performs shape-only inference for BitShift, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}
    
class ReduceL1(ReduceBase):
    # Egor Izmaylov: Function `ReduceL1._get_c_func_name` centralizes the get c func name helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _get_c_func_name(self): return "reduce_l1_forward"

class ReduceL2(ReduceBase):
    # Egor Izmaylov: Function `ReduceL2._get_c_func_name` centralizes the get c func name helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _get_c_func_name(self): return "reduce_l2_forward"

class ReduceLogSum(ReduceBase):
    # Egor Izmaylov: Function `ReduceLogSum._get_c_func_name` centralizes the get c func name helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _get_c_func_name(self): return "reduce_log_sum_forward"

class ReduceLogSumExp(ReduceBase):
    # Egor Izmaylov: Function `ReduceLogSumExp._get_c_func_name` centralizes the get c func name helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _get_c_func_name(self): return "reduce_log_sum_exp_forward"

class ReduceSumSquare(ReduceBase):
    # Egor Izmaylov: Function `ReduceSumSquare._get_c_func_name` centralizes the get c func name helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _get_c_func_name(self): return "reduce_sum_square_forward"

class AveragePool(Ops):
    # Egor Izmaylov: Function `AveragePool.__init__` initializes AveragePool, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, kernel_shape, pads, strides, dtype, dilations=[1, 1], count_include_pad=0, ceil_mode=0, auto_pad="NOTSET", version="17"):
        super().__init__(inputs, outputs)
        self.kernel_shape = kernel_shape
        self.pads = pads
        self.strides = strides
        self.dilations = dilations
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode
        self.auto_pad = auto_pad
        self.dtype = dtype
        self.version = version

        if self.lib:
            self.lib.average_pool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CPoolParams), ctypes.c_int
            ]

    # Egor Izmaylov: Function `AveragePool.forward` executes the concrete runtime path for AveragePool, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        if (
            self.lib is not None
            and x.data.ndim == 4
            and len(self.kernel_shape) == 2
            and x.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
        ):
            _rank, pads, strides, dilations = _normalize_pool_params(
                x.size, self.kernel_shape, self.pads, self.strides, self.dilations, self.auto_pad
            )
            out_shape = _pool_output_shape(
                x.size, self.kernel_shape, pads, strides, dilations, self.ceil_mode, "NOTSET"
            )

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            pads_c = (ctypes.c_int * len(pads))(*pads)
            strides_c = (ctypes.c_int * len(strides))(*strides)
            dilations_c = (ctypes.c_int * len(dilations))(*dilations)
            kernel_c = (ctypes.c_int * len(self.kernel_shape))(*self.kernel_shape)
            c_params = CPoolParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.kernel_shape = ctypes.cast(kernel_c, ctypes.POINTER(ctypes.c_int))

            self.lib.average_pool_forward(x_c, out_c, ctypes.byref(c_params), int(self.count_include_pad))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        out_data = _average_pool_nd(
            x.data, self.kernel_shape, self.pads, self.strides, self.dilations, self.count_include_pad, self.ceil_mode, self.auto_pad
        )
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `AveragePool.forward_` performs shape-only inference for AveragePool, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        out_shape = _pool_output_shape(x.size, self.kernel_shape, self.pads, self.strides, self.dilations, self.ceil_mode, self.auto_pad)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}

class LpPool(Ops):
    # Egor Izmaylov: Function `LpPool.__init__` initializes LpPool, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, kernel_shape, pads, strides, dtype, p=2, dilations=[1, 1], ceil_mode=0, auto_pad="NOTSET", version="17"):
        super().__init__(inputs, outputs)
        self.kernel_shape = kernel_shape
        self.pads = pads
        self.strides = strides
        self.dilations = dilations
        self.p = p
        self.ceil_mode = ceil_mode
        self.auto_pad = auto_pad
        self.dtype = dtype
        self.version = version

        if self.lib:
            self.lib.lp_pool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CPoolParams), ctypes.c_int
            ]

    # Egor Izmaylov: Function `LpPool.forward` executes the concrete runtime path for LpPool, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        if (
            self.lib is not None
            and x.data.ndim == 4
            and len(self.kernel_shape) == 2
            and x.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
        ):
            _rank, pads, strides, dilations = _normalize_pool_params(
                x.size, self.kernel_shape, self.pads, self.strides, self.dilations, self.auto_pad
            )
            out_shape = _pool_output_shape(
                x.size, self.kernel_shape, pads, strides, dilations, self.ceil_mode, "NOTSET"
            )

            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])

            pads_c = (ctypes.c_int * len(pads))(*pads)
            strides_c = (ctypes.c_int * len(strides))(*strides)
            dilations_c = (ctypes.c_int * len(dilations))(*dilations)
            kernel_c = (ctypes.c_int * len(self.kernel_shape))(*self.kernel_shape)
            c_params = CPoolParams()
            c_params.pads = ctypes.cast(pads_c, ctypes.POINTER(ctypes.c_int))
            c_params.strides = ctypes.cast(strides_c, ctypes.POINTER(ctypes.c_int))
            c_params.dilations = ctypes.cast(dilations_c, ctypes.POINTER(ctypes.c_int))
            c_params.kernel_shape = ctypes.cast(kernel_c, ctypes.POINTER(ctypes.c_int))

            self.lib.lp_pool_forward(x_c, out_c, ctypes.byref(c_params), int(self.p))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        out_data = _lp_pool_nd(x.data, self.kernel_shape, self.pads, self.strides, self.dilations, self.p, self.ceil_mode, self.auto_pad)
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `LpPool.forward_` performs shape-only inference for LpPool, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        out_shape = _pool_output_shape(x.size, self.kernel_shape, self.pads, self.strides, self.dilations, self.ceil_mode, self.auto_pad)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}

class GlobalAveragePool(Ops):
    # Egor Izmaylov: Function `GlobalAveragePool.__init__` initializes GlobalAveragePool, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.global_average_pool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)
            ]
    
    # Egor Izmaylov: Function `GlobalAveragePool.forward` executes the concrete runtime path for GlobalAveragePool, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        if len(x.size) < 2:
            raise ValueError("GlobalAveragePool expects input rank >= 2")
        out_shape = tuple(list(x.size[:2]) + [1] * (len(x.size) - 2))
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.global_average_pool_forward(x_c, out_c)
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
        else:
            spatial_axes = tuple(range(2, len(x.size)))
            out_data = np.mean(x.data, axis=spatial_axes, keepdims=True) if spatial_axes else x.data.copy()
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `GlobalAveragePool.forward_` performs shape-only inference for GlobalAveragePool, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        out_shape = list(x.size)
        for axis in range(2, len(out_shape)):
            out_shape[axis] = 1
        return {"tensor": Tensor_(*tuple(out_shape), dtype=self.dtype), "parameters": None}

class GlobalMaxPool(Ops):
    # Egor Izmaylov: Function `GlobalMaxPool.__init__` initializes GlobalMaxPool, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.global_max_pool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)
            ]
    
    # Egor Izmaylov: Function `GlobalMaxPool.forward` executes the concrete runtime path for GlobalMaxPool, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        if len(x.size) < 2:
            raise ValueError("GlobalMaxPool expects input rank >= 2")
        out_shape = tuple(list(x.size[:2]) + [1] * (len(x.size) - 2))
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.global_max_pool_forward(x_c, out_c)
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
        else:
            spatial_axes = tuple(range(2, len(x.size)))
            out_data = np.max(x.data, axis=spatial_axes, keepdims=True) if spatial_axes else x.data.copy()
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `GlobalMaxPool.forward_` performs shape-only inference for GlobalMaxPool, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        out_shape = list(x.size)
        for axis in range(2, len(out_shape)):
            out_shape[axis] = 1
        return {"tensor": Tensor_(*tuple(out_shape), dtype=self.dtype), "parameters": None}
    
class GlobalLpPool(Ops):
    # Egor Izmaylov: Function `GlobalLpPool.__init__` initializes GlobalLpPool, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, p=2, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.p = p
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.global_lp_pool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int
            ]

    # Egor Izmaylov: Function `GlobalLpPool.forward` executes the concrete runtime path for GlobalLpPool, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        if len(x.size) < 2:
            raise ValueError("GlobalLpPool expects input rank >= 2")
        out_shape = tuple(list(x.size[:2]) + [1] * (len(x.size) - 2))
        if self.lib is not None and x.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            out_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.global_lp_pool_forward(x_c, out_c, ctypes.c_int(self.p))
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            self.lib.free_tensor(x_c)
            self.lib.free_tensor(out_c)
        else:
            spatial_axes = tuple(range(2, len(x.size)))
            if spatial_axes:
                out_data = np.sum(np.abs(x.data) ** self.p, axis=spatial_axes, keepdims=True) ** (1.0 / self.p)
            else:
                out_data = np.abs(x.data)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `GlobalLpPool.forward_` performs shape-only inference for GlobalLpPool, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        out_shape = list(x.size)
        for axis in range(2, len(out_shape)):
            out_shape[axis] = 1
        return {"tensor": Tensor_(*tuple(out_shape), dtype=self.dtype), "parameters": None}

class Mean(Ops):
    # Egor Izmaylov: Function `Mean.__init__` initializes Mean, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.mean_forward.argtypes = [
                ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.POINTER(CTensor)
            ]

    # Egor Izmaylov: Function `Mean.forward` executes the concrete runtime path for Mean, consuming real tensor values and returning the graph-runner value contract.
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
            out_data = np.mean(np.stack(arrays, axis=0), axis=0)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `Mean.forward_` performs shape-only inference for Mean, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, *inputs):
        if not inputs:
            raise ValueError("Mean requires at least one input")
        out_shape = np.broadcast_shapes(*(x.size for x in inputs))
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}

class Size(Ops):
    # Egor Izmaylov: Function `Size.__init__` initializes Size, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="int64", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = "int64" # Size always returns int64
        self.version = version

        if self.lib:
            self.lib.size_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)
            ]

    # Egor Izmaylov: Function `Size.forward` executes the concrete runtime path for Size, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Size.forward_` performs shape-only inference for Size, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        return {"tensor": Tensor_(dtype="int64"), "parameters": None}
    
class IsInf(Ops):
    # Egor Izmaylov: Function `IsInf.__init__` initializes IsInf, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `IsInf.forward` executes the concrete runtime path for IsInf, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.isinf_forward(x_c, out_c, ctypes.c_int(self.detect_pos), ctypes.c_int(self.detect_neg))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}

    # Egor Izmaylov: Function `IsInf.forward_` performs shape-only inference for IsInf, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class OneHot(Ops):
    # Egor Izmaylov: Function `OneHot.__init__` initializes OneHot, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axis=-1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype # 由 values 决定，或者外部指定
        self.version = version
        
        if self.lib:
            self.lib.one_hot_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int
            ]

    # Egor Izmaylov: Function `OneHot.forward` executes the concrete runtime path for OneHot, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `OneHot.forward_` performs shape-only inference for OneHot, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
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

class Tril(Ops):
    # Egor Izmaylov: Function `Tril.__init__` initializes Tril, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, k=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.k = k 
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.triangular_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int]

    # Egor Izmaylov: Function `Tril.forward` executes the concrete runtime path for Tril, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Tril.forward_` performs shape-only inference for Tril, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, k_tensor=None): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Triu(Ops):
    # Egor Izmaylov: Function `Triu.__init__` initializes Triu, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, k=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.k = k
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.triangular_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int]

    # Egor Izmaylov: Function `Triu.forward` executes the concrete runtime path for Triu, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Triu.forward_` performs shape-only inference for Triu, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, k_tensor=None): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Trilu(Ops):
    # Egor Izmaylov: Function `Trilu.__init__` initializes Trilu, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, upper=1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.upper = upper
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.triangular_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int
            ]

    # Egor Izmaylov: Function `Trilu.forward` executes the concrete runtime path for Trilu, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `Trilu.forward_` performs shape-only inference for Trilu, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, k_tensor=None):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Round(Ops):
    # Egor Izmaylov: Function `Round.__init__` initializes Round, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Round.forward` executes the concrete runtime path for Round, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x): return {"tensor": self._execute_unary(x, "round_forward"), "parameters": None}
    # Egor Izmaylov: Function `Round.forward_` performs shape-only inference for Round, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Erf(Ops):
    # Egor Izmaylov: Function `Erf.__init__` initializes Erf, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Erf.forward` executes the concrete runtime path for Erf, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x): return {"tensor": self._execute_unary(x, "erf_forward"), "parameters": None}
    # Egor Izmaylov: Function `Erf.forward_` performs shape-only inference for Erf, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class BatchNormalization(Ops):
    # Egor Izmaylov: Function `BatchNormalization.__init__` initializes BatchNormalization, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `BatchNormalization._reshape_param` centralizes the reshape param helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _reshape_param(param, rank):
        return np.asarray(param.data).reshape((-1,) + (1,) * (rank - 2))

    # Egor Izmaylov: Function `BatchNormalization._normalize` centralizes the normalize helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _normalize(self, x_data, scale_data, bias_data, mean_data, var_data):
        y = scale_data * (x_data - mean_data) / np.sqrt(var_data + self.epsilon) + bias_data
        return np.asarray(y, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, x_data.dtype))

    # Egor Izmaylov: Function `BatchNormalization.forward` executes the concrete runtime path for BatchNormalization, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, scale, B, mean, var):
        x_data = np.asarray(x.data)
        rank = x_data.ndim
        scale_data = self._reshape_param(scale, rank)
        bias_data = self._reshape_param(B, rank)

        if self.training_mode:
            axes = tuple(axis for axis in range(rank) if axis != 1)
            saved_mean = np.mean(x_data, axis=axes)
            saved_var = np.var(x_data, axis=axes)
            running_mean = np.asarray(mean.data) * self.momentum + saved_mean * (1.0 - self.momentum)
            running_var = np.asarray(var.data) * self.momentum + saved_var * (1.0 - self.momentum)
            y_data = self._normalize(
                x_data,
                scale_data,
                bias_data,
                saved_mean.reshape((-1,) + (1,) * (rank - 2)),
                saved_var.reshape((-1,) + (1,) * (rank - 2)),
            )
            outputs = (
                Tensor(*x.size, dtype=self.dtype, data=y_data),
                Tensor(*saved_mean.shape, dtype=self.dtype, data=np.asarray(running_mean, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, running_mean.dtype))),
                Tensor(*saved_var.shape, dtype=self.dtype, data=np.asarray(running_var, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, running_var.dtype))),
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

    # Egor Izmaylov: Function `BatchNormalization.forward_` performs shape-only inference for BatchNormalization, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, scale, B, mean, var):
        outputs = [Tensor_(*x.size, dtype=self.dtype)]
        if self.training_mode:
            outputs.extend([Tensor_(x.size[1], dtype=self.dtype), Tensor_(x.size[1], dtype=self.dtype)])
        selected = tuple(value for name, value in zip(self.outputs, outputs) if name)
        return {"tensor": selected[0] if len(selected) == 1 else selected, "parameters": None}

class InstanceNormalization(Ops):
    # Egor Izmaylov: Function `InstanceNormalization.__init__` initializes InstanceNormalization, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `InstanceNormalization.forward` executes the concrete runtime path for InstanceNormalization, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `InstanceNormalization.forward_` performs shape-only inference for InstanceNormalization, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, scale, B): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class LayerNormalization(Ops):
    # Egor Izmaylov: Function `LayerNormalization.__init__` initializes LayerNormalization, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `LayerNormalization.forward` executes the concrete runtime path for LayerNormalization, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x, scale=None, B=None):
        x_data = np.asarray(x.data)
        rank = x_data.ndim
        axis = self.axis if self.axis >= 0 else self.axis + rank
        if axis < 0 or axis >= rank:
            raise ValueError(f"LayerNormalization axis {self.axis} is out of bounds for rank {rank}")
        wants_aux_outputs = len([name for name in self.outputs if name]) > 1
        if (
            self.lib is not None
            and not wants_aux_outputs
            and axis == rank - 1
            and self.dtype in nn.DTYPE_MAP
            and x.dtype in nn.DTYPE_MAP
            and (scale is None or scale.dtype in nn.DTYPE_MAP)
            and (B is None or B.dtype in nn.DTYPE_MAP)
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
        row_number = int(np.prod(x_data.shape[:axis], dtype=np.int64)) if axis > 0 else 1
        col_number = int(np.prod(x_data.shape[axis:], dtype=np.int64))
        stash_np_dtype = nn.DTYPE_TO_NUMPY.get(self.stash_dtype, np.float32)
        work = x_data.astype(stash_np_dtype, copy=False).reshape(row_number, col_number)
        mean = np.mean(work, axis=1, keepdims=True)
        inv_std = np.reciprocal(np.sqrt(np.mean((work - mean) ** 2, axis=1, keepdims=True) + self.epsilon))
        normalized = ((work - mean) * inv_std).reshape(x_data.shape)
        if scale is not None:
            normalized = normalized * np.asarray(scale.data)
        if B is not None:
            normalized = normalized + np.asarray(B.data)

        reduction_shape = tuple(x_data.shape[:axis]) + (1,) * (rank - axis)
        y = Tensor(*x.size, dtype=self.dtype, data=np.asarray(normalized, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, x_data.dtype)))
        mean_tensor = Tensor(*reduction_shape, dtype=self.stash_dtype, data=np.asarray(mean.reshape(reduction_shape), dtype=stash_np_dtype))
        inv_std_tensor = Tensor(*reduction_shape, dtype=self.stash_dtype, data=np.asarray(inv_std.reshape(reduction_shape), dtype=stash_np_dtype))
        outputs = (y, mean_tensor, inv_std_tensor)
        selected = tuple(value for name, value in zip(self.outputs, outputs) if name)
        return {"tensor": selected[0] if len(selected) == 1 else selected, "parameters": None}

    # Egor Izmaylov: Function `LayerNormalization.forward_` performs shape-only inference for LayerNormalization, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
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

# Egor Izmaylov: Function `_window_output_shape` centralizes the window output shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _window_output_shape(size):
    if hasattr(size, "data") and size.data is not None:
        return (int(np.asarray(size.data).item()),)
    return (1,)


# Egor Izmaylov: Function `_float32_to_bfloat16_bits` centralizes the float32 to bfloat16 bits helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _float32_to_bfloat16_bits(values):
    data = np.asarray(values, dtype=np.float32)
    bits = data.view(np.uint32)
    lsb = (bits >> 16) & 1
    guard = (bits >> 15) & 1
    sticky = (bits & 0x7FFF) != 0
    rounded = bits + ((guard & (sticky | lsb)).astype(np.uint32) << 16)
    rounded = np.where(np.isnan(data), bits, rounded)
    return (rounded >> 16).astype(np.uint16)


# Egor Izmaylov: Function `_cast_window_output` centralizes the cast window output helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _cast_window_output(values, dtype):
    if dtype == "bfloat16":
        return _float32_to_bfloat16_bits(values)
    return np.asarray(values, dtype=nn.DTYPE_TO_NUMPY.get(dtype, np.float32))


# Egor Izmaylov: Function `_window_values` centralizes the window values helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _window_values(size, periodic, dtype, kind):
    length = int(np.asarray(size.data).item())
    if length < 0:
        raise ValueError(f"Window size must be non-negative, got {length}")
    if length == 0:
        return np.empty((0,), dtype=nn.DTYPE_TO_NUMPY.get(dtype, np.float32))
    if length == 1:
        return _cast_window_output(np.ones((1,), dtype=np.float64), dtype)

    denom = length if periodic else length - 1
    n = np.arange(length, dtype=np.float64)
    if kind == "hann":
        values = np.sin(n * np.pi / denom) ** 2
    elif kind == "hamming":
        alpha = 25.0 / 46.0
        values = alpha - (1.0 - alpha) * np.cos(2.0 * np.pi * n / denom)
    elif kind == "blackman":
        values = 0.42 - 0.5 * np.cos(2.0 * np.pi * n / denom) + 0.08 * np.cos(4.0 * np.pi * n / denom)
    else:
        raise ValueError(f"Unknown window kind {kind!r}")
    return _cast_window_output(values, dtype)


# Egor Izmaylov: Function `_window_values_c_first` centralizes the window values c first helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _window_values_c_first(op, size, c_func_name, kind):
    length = int(np.asarray(size.data).item())
    if length < 0:
        raise ValueError(f"Window size must be non-negative, got {length}")
    out_shape = (length,)
    if op.lib is not None and op.dtype in nn.DTYPE_MAP and size.dtype in nn.DTYPE_MAP:
        size_c = op._numpy_to_ctensor(np.ascontiguousarray(size.data), size.dtype)
        output_shape_c = (ctypes.c_int * 1)(length)
        output_c = op.lib.create_tensor(output_shape_c, 1, nn.DTYPE_MAP[op.dtype])
        getattr(op.lib, c_func_name)(size_c, output_c, ctypes.c_int(op.periodic))
        out_data = op._ctensor_to_numpy(output_c, op.dtype)
        op.lib.free_tensor(size_c)
        op.lib.free_tensor(output_c)
        return out_data
    return _window_values(size, op.periodic, op.dtype, kind)


class HannWindow(Ops):
    # Egor Izmaylov: Function `HannWindow.__init__` initializes HannWindow, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, periodic=1, output_datatype=1, version="17"):
        super().__init__(inputs, outputs)
        self.periodic = periodic
        self.dtype = nn.onnx_dtype_mapping.get(output_datatype, "float32")
        self.version = version
        if self.lib:
            self.lib.hann_window_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]

    # Egor Izmaylov: Function `HannWindow.forward` executes the concrete runtime path for HannWindow, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, size):
        out_data = _window_values_c_first(self, size, "hann_window_forward", "hann")
        out_shape = out_data.shape
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `HannWindow.forward_` performs shape-only inference for HannWindow, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, size):
        return {"tensor": Tensor_(*_window_output_shape(size), dtype=self.dtype), "parameters": None}

class HammingWindow(Ops):
    # Egor Izmaylov: Function `HammingWindow.__init__` initializes HammingWindow, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, periodic=1, output_datatype=1, version="17"):
        super().__init__(inputs, outputs)
        self.periodic = periodic
        self.dtype = nn.onnx_dtype_mapping.get(output_datatype, "float32")
        self.version = version
        if self.lib:
            self.lib.hamming_window_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]

    # Egor Izmaylov: Function `HammingWindow.forward` executes the concrete runtime path for HammingWindow, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, size):
        out_data = _window_values_c_first(self, size, "hamming_window_forward", "hamming")
        out_shape = out_data.shape
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `HammingWindow.forward_` performs shape-only inference for HammingWindow, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, size):
        return {"tensor": Tensor_(*_window_output_shape(size), dtype=self.dtype), "parameters": None}

class BlackmanWindow(Ops):
    # Egor Izmaylov: Function `BlackmanWindow.__init__` initializes BlackmanWindow, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, periodic=1, output_datatype=1, version="17"):
        super().__init__(inputs, outputs)
        self.periodic = periodic
        self.dtype = nn.onnx_dtype_mapping.get(output_datatype, "float32")
        self.version = version
        if self.lib:
            self.lib.blackman_window_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]

    # Egor Izmaylov: Function `BlackmanWindow.forward` executes the concrete runtime path for BlackmanWindow, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, size):
        out_data = _window_values_c_first(self, size, "blackman_window_forward", "blackman")
        out_shape = out_data.shape
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `BlackmanWindow.forward_` performs shape-only inference for BlackmanWindow, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, size):
        return {"tensor": Tensor_(*_window_output_shape(size), dtype=self.dtype), "parameters": None}

class RandomNormal(Ops):
    # Egor Izmaylov: Function `RandomNormal.__init__` initializes RandomNormal, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, mean=0.0, scale=1.0, seed=0.0, dtype=1, shape=None, version="17"):
        super().__init__(inputs, outputs)
        self.mean = mean
        self.scale = scale
        self.seed = seed
        self.dtype = nn.onnx_dtype_mapping.get(dtype, "float32")
        self.shape_val = shape # list
        self.version = version
        if self.lib:
            self.lib.random_normal_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float, ctypes.c_float, ctypes.c_float]

    # Egor Izmaylov: Function `RandomNormal.forward` executes the concrete runtime path for RandomNormal, consuming real tensor values and returning the graph-runner value contract.
    def forward(self):
        # Shape 必须是初始化属性
        if self.shape_val is None:
            raise ValueError("RandomNormal requires 'shape' attribute")
        
        out_shape = tuple(self.shape_val)
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), DTYPE_MAP[self.dtype])
        
        self.lib.random_normal_forward(output_c, ctypes.c_float(self.mean), ctypes.c_float(self.scale), ctypes.c_float(self.seed))
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(output_c)
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `RandomNormal.forward_` performs shape-only inference for RandomNormal, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self):
        out_shape = tuple(self.shape_val) if self.shape_val is not None else (1,)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}

class RandomNormalLike(Ops):
    # Egor Izmaylov: Function `RandomNormalLike.__init__` initializes RandomNormalLike, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, mean=0.0, scale=1.0, seed=0.0, dtype=None, version="17"):
        super().__init__(inputs, outputs)
        self.mean = mean
        self.scale = scale
        self.seed = seed
        self.dtype = nn.onnx_dtype_mapping.get(dtype, "float32") if dtype else None
        self.version = version
        if self.lib:
            self.lib.random_normal_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float, ctypes.c_float, ctypes.c_float]

    # Egor Izmaylov: Function `RandomNormalLike.forward` executes the concrete runtime path for RandomNormalLike, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input):
        target_dtype = self.dtype if self.dtype else input.dtype
        out_shape = input.size
        
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), DTYPE_MAP[target_dtype])
        
        self.lib.random_normal_forward(output_c, ctypes.c_float(self.mean), ctypes.c_float(self.scale), ctypes.c_float(self.seed))
        
        out_data = self._ctensor_to_numpy(output_c, target_dtype)
        self.lib.free_tensor(output_c)
        return {"tensor": Tensor(*out_shape, dtype=target_dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `RandomNormalLike.forward_` performs shape-only inference for RandomNormalLike, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input):
        target_dtype = self.dtype if self.dtype else input.dtype
        return {"tensor": Tensor_(*input.size, dtype=target_dtype), "parameters": None}

class Bernoulli(Ops):
    # Egor Izmaylov: Function `Bernoulli.__init__` initializes Bernoulli, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, seed=0.0, dtype=None, version="17"):
        super().__init__(inputs, outputs)
        self.seed = seed
        self.dtype = nn.onnx_dtype_mapping.get(dtype, "float32") if dtype else None
        self.version = version
        if self.lib:
            self.lib.bernoulli_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]

    # Egor Izmaylov: Function `Bernoulli.forward` executes the concrete runtime path for Bernoulli, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input):
        target_dtype = self.dtype if self.dtype else input.dtype
        out_shape = input.size
        if self.lib is None or target_dtype not in nn.DTYPE_MAP:
            seed = None if self.seed is None or self.seed == 0.0 else int(self.seed)
            rng = np.random.default_rng(seed)
            out_data = rng.binomial(1, p=np.asarray(input.data, dtype=np.float64)).astype(
                nn.DTYPE_TO_NUMPY.get(target_dtype, np.float32)
            )
            return {"tensor": Tensor(*out_shape, dtype=target_dtype, data=out_data), "parameters": None}
        
        input_c = self._numpy_to_ctensor(input.data, input.dtype)
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), DTYPE_MAP[target_dtype])
        
        self.lib.bernoulli_forward(input_c, output_c, ctypes.c_float(self.seed))
        
        out_data = self._ctensor_to_numpy(output_c, target_dtype)
        self.lib.free_tensor(input_c); self.lib.free_tensor(output_c)
        return {"tensor": Tensor(*out_shape, dtype=target_dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `Bernoulli.forward_` performs shape-only inference for Bernoulli, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input):
        target_dtype = self.dtype if self.dtype else input.dtype
        return {"tensor": Tensor_(*input.size, dtype=target_dtype), "parameters": None}

class Dropout(Ops):
    # Egor Izmaylov: Function `Dropout.__init__` initializes Dropout, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, seed=None, ratio=0.5, training_mode=0, version="17"):
        super().__init__(inputs, outputs)
        self.seed = seed
        self.default_ratio = ratio
        self.training_mode = training_mode
        self.version = version
        if self.lib:
            self.lib.dropout_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float, ctypes.c_int]

    # Egor Izmaylov: Function `Dropout.forward` executes the concrete runtime path for Dropout, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, data, ratio=None, training_mode=None):
        r = float(self.default_ratio)
        if ratio is not None:
            r = float(ratio.data.item())
        
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
            out_data = data.data * mask_data.astype(data.data.dtype) / (1.0 - r)
        else:
            mask_data = np.ones(data.size, dtype=np.bool_)
            out_data = data.data.copy()

        output_tensor = Tensor(*data.size, dtype=data.dtype, data=out_data.astype(data.data.dtype, copy=False))
        if len(self.outputs) > 1 and self.outputs[1]:
            mask_tensor = Tensor(*data.size, dtype="bool", data=mask_data)
            return {"tensor": (output_tensor, mask_tensor), "parameters": None}
        return {"tensor": output_tensor, "parameters": None}

    # Egor Izmaylov: Function `Dropout.forward_` performs shape-only inference for Dropout, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, data, ratio=None, training_mode=None):
        output_tensor = Tensor_(*data.size, dtype=data.dtype)
        if len(self.outputs) > 1 and self.outputs[1]:
            return {"tensor": (output_tensor, Tensor_(*data.size, dtype="bool")), "parameters": None}
        return {"tensor": output_tensor, "parameters": None}

class Gelu(Ops):
    # Egor Izmaylov: Function `Gelu.__init__` initializes Gelu, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Gelu.forward` executes the concrete runtime path for Gelu, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x): return {"tensor": self._execute_unary(x, "gelu_forward"), "parameters": None}
    # Egor Izmaylov: Function `Gelu.forward_` performs shape-only inference for Gelu, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Mish(Ops):
    # Egor Izmaylov: Function `Mish.__init__` initializes Mish, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # Egor Izmaylov: Function `Mish.forward` executes the concrete runtime path for Mish, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x): return {"tensor": self._execute_unary(x, "mish_forward"), "parameters": None}
    # Egor Izmaylov: Function `Mish.forward_` performs shape-only inference for Mish, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Hardmax(Ops):
    # Egor Izmaylov: Function `Hardmax.__init__` initializes Hardmax, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axis=-1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.hardmax_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]

    # Egor Izmaylov: Function `Hardmax.forward` executes the concrete runtime path for Hardmax, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input):
        out_tensor = Tensor(*input.size, dtype=self.dtype)
        input_c = self._numpy_to_ctensor(input.data, input.dtype)
        output_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        
        self.lib.hardmax_forward(input_c, output_c, ctypes.c_int(self.axis))
        
        out_tensor.data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c); self.lib.free_tensor(output_c)
        return {"tensor": out_tensor, "parameters": None}

    # Egor Izmaylov: Function `Hardmax.forward_` performs shape-only inference for Hardmax, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input):
        return {"tensor": Tensor_(*input.size, dtype=self.dtype), "parameters": None}

class LogSoftmax(Ops):
    # Egor Izmaylov: Function `LogSoftmax.__init__` initializes LogSoftmax, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axis=-1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.log_softmax_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]

    # Egor Izmaylov: Function `LogSoftmax.forward` executes the concrete runtime path for LogSoftmax, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input):
        out_tensor = Tensor(*input.size, dtype=self.dtype)
        input_c = self._numpy_to_ctensor(input.data, input.dtype)
        output_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        
        self.lib.log_softmax_forward(input_c, output_c, ctypes.c_int(self.axis))
        
        out_tensor.data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c); self.lib.free_tensor(output_c)
        return {"tensor": out_tensor, "parameters": None}

    # Egor Izmaylov: Function `LogSoftmax.forward_` performs shape-only inference for LogSoftmax, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input):
        return {"tensor": Tensor_(*input.size, dtype=self.dtype), "parameters": None}

class LpNormalization(Ops):
    # Egor Izmaylov: Function `LpNormalization.__init__` initializes LpNormalization, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axis=-1, p=2, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.p = p
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.lp_normalization_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int]

    # Egor Izmaylov: Function `LpNormalization.forward` executes the concrete runtime path for LpNormalization, consuming real tensor values and returning the graph-runner value contract.
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

        data = np.asarray(input.data)
        norm = np.power(np.power(np.abs(data), self.p).sum(axis=axis), 1.0 / self.p)
        norm = np.expand_dims(norm, axis)
        out_data = np.where(norm == 0, 0, data / norm)
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # Egor Izmaylov: Function `LpNormalization.forward_` performs shape-only inference for LpNormalization, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input):
        return {"tensor": Tensor_(*input.size, dtype=self.dtype), "parameters": None}

class DepthToSpace(Ops):
    # Egor Izmaylov: Function `DepthToSpace.__init__` initializes DepthToSpace, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `DepthToSpace.forward` executes the concrete runtime path for DepthToSpace, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `DepthToSpace.forward_` performs shape-only inference for DepthToSpace, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input):
        N, C, H, W = input.size
        bs = self.blocksize
        new_C = C // (bs * bs)
        out_shape = (N, new_C, H * bs, W * bs)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}

class SpaceToDepth(Ops):
    # Egor Izmaylov: Function `SpaceToDepth.__init__` initializes SpaceToDepth, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, blocksize, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.blocksize = blocksize
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.space_to_depth_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int
            ]

    # Egor Izmaylov: Function `SpaceToDepth.forward` executes the concrete runtime path for SpaceToDepth, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `SpaceToDepth.forward_` performs shape-only inference for SpaceToDepth, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input):
        N, C, H, W = input.size
        bs = self.blocksize
        out_shape = (N, C * bs * bs, H // bs, W // bs)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}

class ReverseSequence(Ops):
    # Egor Izmaylov: Function `ReverseSequence.__init__` initializes ReverseSequence, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `ReverseSequence.forward` executes the concrete runtime path for ReverseSequence, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input, sequence_lens):
        out_tensor = Tensor(*input.size, dtype=self.dtype)
        
        in_c = self._numpy_to_ctensor(input.data, input.dtype)
        seq_c = self._numpy_to_ctensor(sequence_lens.data, sequence_lens.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        
        self.lib.reverse_sequence_forward(in_c, seq_c, out_c, ctypes.c_int(self.time_axis), ctypes.c_int(self.batch_axis))
        
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(in_c); self.lib.free_tensor(seq_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}

    # Egor Izmaylov: Function `ReverseSequence.forward_` performs shape-only inference for ReverseSequence, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input, sequence_lens):
        return {"tensor": Tensor_(*input.size, dtype=self.dtype), "parameters": None}

class Compress(Ops):
    # Egor Izmaylov: Function `Compress.__init__` initializes Compress, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, axis=None, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.compress_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int
            ]

    # Egor Izmaylov: Function `Compress.forward` executes the concrete runtime path for Compress, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input, condition):
        cond = np.asarray(condition.data).astype(bool).reshape(-1)
        if self.axis is None:
            out_data = np.compress(cond, np.asarray(input.data).reshape(-1), axis=0)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
            return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

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

    # Egor Izmaylov: Function `Compress.forward_` performs shape-only inference for Compress, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
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
    # Egor Izmaylov: Function `ScatterElements.__init__` initializes ScatterElements, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `ScatterElements.forward` executes the concrete runtime path for ScatterElements, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, data, indices, updates):
        out_tensor = Tensor(*data.size, dtype=self.dtype, data=data.data.copy())
        
        d_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        i_c = self._numpy_to_ctensor(indices.data, indices.dtype)
        u_c = self._numpy_to_ctensor(updates.data, updates.dtype)
        
        self.lib.scatter_elements_forward(d_c, i_c, u_c, ctypes.c_int(self.axis), ctypes.c_int(self.reduction))
        
        out_tensor.data = self._ctensor_to_numpy(d_c, self.dtype)
        self.lib.free_tensor(d_c); self.lib.free_tensor(i_c); self.lib.free_tensor(u_c)
        return {"tensor": out_tensor, "parameters": None}

    # Egor Izmaylov: Function `ScatterElements.forward_` performs shape-only inference for ScatterElements, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, data, indices, updates):
        return {"tensor": Tensor_(*data.size, dtype=self.dtype), "parameters": None}

class GroupNormalization(Ops):
    # Egor Izmaylov: Function `GroupNormalization.__init__` initializes GroupNormalization, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
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

    # Egor Izmaylov: Function `GroupNormalization.forward` executes the concrete runtime path for GroupNormalization, consuming real tensor values and returning the graph-runner value contract.
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

    # Egor Izmaylov: Function `GroupNormalization.forward_` performs shape-only inference for GroupNormalization, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x, scale, bias):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class StringNormalizer(Ops):
    # Egor Izmaylov: Function `StringNormalizer.__init__` initializes StringNormalizer, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(
        self,
        inputs,
        outputs,
        case_change_action="NONE",
        is_case_sensitive=0,
        locale="",
        stopwords=None,
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.case_change_action = case_change_action
        self.is_case_sensitive = bool(is_case_sensitive)
        self.locale = locale
        self.stopwords = list(stopwords or [])
        self.dtype = "string"
        self.version = version

    # Egor Izmaylov: Function `StringNormalizer._strip_accents` centralizes the strip accents helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _strip_accents(text):
        try:
            text.encode("ASCII", errors="strict")
            return text
        except UnicodeEncodeError:
            normalized = unicodedata.normalize("NFKD", text)
            return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    # Egor Izmaylov: Function `StringNormalizer._remove_stopwords` centralizes the remove stopwords helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _remove_stopwords(text, stops):
        return " ".join(token for token in text.split(" ") if token not in stops)

    # Egor Izmaylov: Function `StringNormalizer._normalize_text` centralizes the normalize text helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _normalize_text(self, value):
        if isinstance(value, float) and np.isnan(value):
            return ""
        text = self._strip_accents(str(value))
        raw_stops = set(self.stopwords)
        if self.case_change_action == "LOWER":
            stops = {word.lower() for word in self.stopwords}
        elif self.case_change_action == "UPPER":
            stops = {word.upper() for word in self.stopwords}
        elif self.case_change_action == "NONE":
            stops = raw_stops
        else:
            raise ValueError(f"Unknown case_change_action {self.case_change_action!r}")

        if self.is_case_sensitive and raw_stops:
            text = self._remove_stopwords(text, raw_stops)
        if self.case_change_action == "LOWER":
            text = text.lower()
        elif self.case_change_action == "UPPER":
            text = text.upper()
        if not self.is_case_sensitive and stops:
            text = self._remove_stopwords(text, stops)
        return text

    # Egor Izmaylov: Function `StringNormalizer.forward` executes the concrete runtime path for StringNormalizer, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        data = np.asarray(x.data, dtype=np.str_)
        if data.ndim == 1:
            normalized = [self._normalize_text(value) for value in data.tolist()]
            normalized = [value for value in normalized if len(value) > 0]
            if not normalized:
                normalized = [""]
            out_data = np.asarray(normalized, dtype=np.str_)
        elif data.ndim == 2 and data.shape[0] == 1:
            normalized = [self._normalize_text(value) for value in data[0].tolist()]
            normalized = [value for value in normalized if len(value) > 0]
            if not normalized:
                normalized = [""]
            out_data = np.asarray([normalized], dtype=np.str_)
        else:
            raise ValueError(f"StringNormalizer expects shape [C] or [1, C], got {x.size}")
        return {"tensor": Tensor(*out_data.shape, dtype="string", data=out_data), "parameters": None}

    # Egor Izmaylov: Function `StringNormalizer.forward_` performs shape-only inference for StringNormalizer, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        if isinstance(x, Tensor):
            return {"tensor": Tensor_(*self.forward(x)["tensor"].size, dtype="string"), "parameters": None}
        return {"tensor": Tensor_(*x.size, dtype="string"), "parameters": None}

class TfIdfVectorizer(Ops):
    # Egor Izmaylov: Function `TfIdfVectorizer.__init__` initializes TfIdfVectorizer, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(
        self,
        inputs,
        outputs,
        mode,
        ngram_counts,
        ngram_indexes,
        max_skip_count,
        min_gram_length,
        max_gram_length,
        pool_int64s=None,
        pool_strings=None,
        weights=None,
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.mode = mode
        self.ngram_counts = list(ngram_counts)
        self.ngram_indexes = list(ngram_indexes)
        self.max_skip_count = int(max_skip_count)
        self.min_gram_length = int(min_gram_length)
        self.max_gram_length = int(max_gram_length)
        self.pool_int64s = list(pool_int64s or [])
        self.pool_strings = list(pool_strings or [])
        self.weights = list(weights or [])
        self.dtype = "float32"
        self.version = version
        self._ngram_map = self._build_ngram_map()
        self.output_size = max(self.ngram_indexes) + 1 if self.ngram_indexes else 0

    # Egor Izmaylov: Function `TfIdfVectorizer._build_ngram_map` centralizes the build ngram map helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _build_ngram_map(self):
        pool = self.pool_strings if self.pool_strings else self.pool_int64s
        ngram_map = {}
        ngram_id = 0
        for size_index, start in enumerate(self.ngram_counts):
            ngram_size = size_index + 1
            end = self.ngram_counts[size_index + 1] if size_index + 1 < len(self.ngram_counts) else len(pool)
            item_count = max(0, end - start)
            ngram_count = item_count // ngram_size if ngram_size > 0 else 0
            for idx in range(ngram_count):
                gram_start = start + idx * ngram_size
                gram = tuple(pool[gram_start:gram_start + ngram_size])
                if self.min_gram_length <= ngram_size <= self.max_gram_length and ngram_id < len(self.ngram_indexes):
                    ngram_map[gram] = self.ngram_indexes[ngram_id]
                ngram_id += 1
        return ngram_map

    # Egor Izmaylov: Function `TfIdfVectorizer._rows` centralizes the rows helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _rows(self, data):
        if data.ndim == 0:
            return [data.reshape(1)], False
        if data.ndim == 1:
            return [data], False
        if data.ndim == 2:
            if data.shape[0] < 1:
                raise ValueError("TfIdfVectorizer 2-D input must have B > 0")
            return [data[i] for i in range(data.shape[0])], True
        raise ValueError(f"TfIdfVectorizer expects scalar, 1-D, or 2-D input, got shape {data.shape}")

    # Egor Izmaylov: Function `TfIdfVectorizer._count_row` centralizes the count row helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _count_row(self, row):
        counts = np.zeros((self.output_size,), dtype=np.float32)
        if self.output_size == 0:
            return counts
        row_values = row.tolist()
        min_size = self.min_gram_length
        for skip_distance in range(1, self.max_skip_count + 2):
            start_size = min_size
            if skip_distance > 1 and start_size == 1:
                start_size = 2
            for start in range(len(row_values)):
                for ngram_size in range(start_size, self.max_gram_length + 1):
                    last = start + skip_distance * (ngram_size - 1)
                    if last >= len(row_values):
                        break
                    gram = tuple(row_values[start + skip_distance * offset] for offset in range(ngram_size))
                    out_idx = self._ngram_map.get(gram)
                    if out_idx is not None:
                        counts[out_idx] += 1.0
        return counts

    # Egor Izmaylov: Function `TfIdfVectorizer._apply_mode` centralizes the apply mode helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    def _apply_mode(self, counts):
        mode = self.mode.upper()
        if mode == "TF":
            return counts
        if mode == "IDF":
            if self.weights:
                weights = np.asarray(self.weights, dtype=np.float32)
                return np.where(counts > 0, weights[:self.output_size], 0.0).astype(np.float32)
            return (counts > 0).astype(np.float32)
        if mode == "TFIDF":
            if self.weights:
                weights = np.asarray(self.weights, dtype=np.float32)
                return counts * weights[:self.output_size]
            return counts
        raise ValueError(f"Unsupported TfIdfVectorizer mode {self.mode!r}")

    # Egor Izmaylov: Function `TfIdfVectorizer.forward` executes the concrete runtime path for TfIdfVectorizer, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        data = np.asarray(x.data)
        rows, batched = self._rows(data)
        vectors = [self._apply_mode(self._count_row(row)) for row in rows]
        out_data = np.stack(vectors, axis=0).astype(np.float32)
        if not batched:
            out_data = out_data.reshape((self.output_size,))
        return {"tensor": Tensor(*out_data.shape, dtype="float32", data=out_data), "parameters": None}

    # Egor Izmaylov: Function `TfIdfVectorizer.forward_` performs shape-only inference for TfIdfVectorizer, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        if len(x.size) == 2:
            out_shape = (x.size[0], self.output_size)
        else:
            out_shape = (self.output_size,)
        return {"tensor": Tensor_(*out_shape, dtype="float32"), "parameters": None}

# Egor Izmaylov: Function `_numpy_dtype_to_tensor_dtype` centralizes the numpy dtype to tensor dtype helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _numpy_dtype_to_tensor_dtype(array):
    return nn.NUMPY_TO_DTYPE.get(array.dtype.type, "float32")


# Egor Izmaylov: Function `_tensor_from_numpy` centralizes the tensor from numpy helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _tensor_from_numpy(array):
    array = np.asarray(array)
    dtype = _numpy_dtype_to_tensor_dtype(array)
    return Tensor(*array.shape, dtype=dtype, data=array)


# Egor Izmaylov: Function `_tensor_to_numpy` centralizes the tensor to numpy helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _tensor_to_numpy(value):
    if isinstance(value, Tensor):
        return value.data
    if isinstance(value, Tensor_):
        return np.zeros(value.size, dtype=nn.DTYPE_TO_NUMPY.get(value.dtype, np.float32))
    return np.asarray(value)


# Egor Izmaylov: Function `_reference_feed_value` centralizes the reference feed value helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _reference_feed_value(value):
    if isinstance(value, (list, tuple)):
        return [_reference_feed_value(item) for item in value]
    return _tensor_to_numpy(value)


# Egor Izmaylov: Function `_graph_local_value_names` centralizes the graph local value names helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _graph_local_value_names(graph_proto):
    names = {value.name for value in graph_proto.input if value.name}
    names.update(value.name for value in graph_proto.initializer if value.name)
    names.update(value for node in graph_proto.node for value in node.output if value)
    return names


# Egor Izmaylov: Function `_graph_external_names` centralizes the graph external names helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _graph_external_names(graph_proto):
    local_names = _graph_local_value_names(graph_proto)
    used_names = {value for node in graph_proto.node for value in node.input if value}
    used_names.update(value.name for value in graph_proto.output if value.name)

    nested_external_names = set()
    for node in graph_proto.node:
        for attr in node.attribute:
            if attr.type == attr.GRAPH:
                nested_external_names.update(_graph_external_names(attr.g))
            elif attr.type == attr.GRAPHS:
                for nested_graph in attr.graphs:
                    nested_external_names.update(_graph_external_names(nested_graph))
    used_names.update(nested_external_names)
    return {name for name in used_names if name not in local_names}


# Egor Izmaylov: Function `_graph_value_shape` centralizes the graph value shape helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _graph_value_shape(value_info):
    tensor_type = value_info.type.tensor_type
    dtype = nn.onnx_dtype_mapping.get(tensor_type.elem_type, "float32")
    dims = []
    for dim in tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else 1)
    return Tensor_(*dims, dtype=dtype)


# Egor Izmaylov: Function `_run_graph_proto` centralizes the run graph proto helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
def _run_graph_proto(graph_proto, feeds, outer_scope=None):
    from onnx import helper
    from onnx.reference import ReferenceEvaluator

    model = helper.make_model(graph_proto, opset_imports=[helper.make_opsetid("", 17)])
    evaluator = ReferenceEvaluator(model)
    input_names = {value.name for value in graph_proto.input}
    needed_names = input_names | _graph_external_names(graph_proto)
    graph_feeds = {}
    if outer_scope:
        graph_feeds.update(
            {
                name: _reference_feed_value(value)
                for name, value in outer_scope.items()
                if name in needed_names
            }
        )
    graph_feeds.update(
        {
            name: _reference_feed_value(value)
            for name, value in feeds.items()
            if name in needed_names
        }
    )
    return evaluator.run(None, graph_feeds)


class If(Ops):
    # Egor Izmaylov: Function `If.__init__` initializes If, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, then_branch, else_branch, version="17"):
        super().__init__(inputs, outputs)
        self.then_branch = then_branch
        self.else_branch = else_branch
        self.version = version
        self.outer_scope_names = sorted(
            _graph_external_names(then_branch) | _graph_external_names(else_branch)
        )

    # Egor Izmaylov: Function `If.forward` executes the concrete runtime path for If, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, cond):
        return self.forward_with_context(None, cond)

    # Egor Izmaylov: Function `If.forward_with_context` runs control-flow graph bodies for If with an outer-scope feed dictionary so subgraphs can resolve captured values.
    def forward_with_context(self, outer_scope, cond):
        condition = bool(np.asarray(cond.data).item())
        graph = self.then_branch if condition else self.else_branch
        outputs = tuple(_tensor_from_numpy(value) for value in _run_graph_proto(graph, {}, outer_scope))
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}

    # Egor Izmaylov: Function `If.forward_` performs shape-only inference for If, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, cond):
        graph = self.then_branch
        outputs = tuple(_graph_value_shape(value_info) for value_info in graph.output)
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}


class Loop(Ops):
    # Egor Izmaylov: Function `Loop.__init__` initializes Loop, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, body, version="17"):
        super().__init__(inputs, outputs)
        self.body = body
        self.version = version
        self.outer_scope_names = sorted(_graph_external_names(body))

    # Egor Izmaylov: Function `Loop._trip_count` centralizes the trip count helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _trip_count(m):
        if m is None:
            return None
        return int(np.asarray(m.data).item())

    # Egor Izmaylov: Function `Loop._condition` centralizes the condition helper logic for the ONNX operator runtime and shape-inference layer, so edge-case normalization stays in one implementation boundary.
    @staticmethod
    def _condition(cond):
        if cond is None:
            return True
        return bool(np.asarray(cond.data).item())

    # Egor Izmaylov: Function `Loop.forward` executes the concrete runtime path for Loop, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, m=None, cond=None, *loop_vars):
        return self.forward_with_context(None, m, cond, *loop_vars)

    # Egor Izmaylov: Function `Loop.forward_with_context` runs control-flow graph bodies for Loop with an outer-scope feed dictionary so subgraphs can resolve captured values.
    def forward_with_context(self, outer_scope, m=None, cond=None, *loop_vars):
        trip_count = self._trip_count(m)
        condition = self._condition(cond)
        if trip_count is None and cond is None:
            raise ValueError("Loop without trip count or condition would be unbounded")
        body_inputs = [value.name for value in self.body.input]
        state_values = [_tensor_to_numpy(value) for value in loop_vars]
        scan_outputs = None
        iteration = 0
        last_outputs = None
        while condition and (trip_count is None or iteration < trip_count):
            feeds = {}
            if body_inputs:
                feeds[body_inputs[0]] = np.asarray(iteration, dtype=np.int64)
            if len(body_inputs) > 1:
                feeds[body_inputs[1]] = np.asarray(condition, dtype=np.bool_)
            for name, value in zip(body_inputs[2:], state_values):
                feeds[name] = value
            last_outputs = list(_run_graph_proto(self.body, feeds, outer_scope))
            condition = bool(np.asarray(last_outputs[0]).item())
            state_values = [np.asarray(value) for value in last_outputs[1:1 + len(state_values)]]
            produced_scan = last_outputs[1 + len(state_values):]
            if scan_outputs is None:
                scan_outputs = [[] for _ in produced_scan]
            for bucket, value in zip(scan_outputs, produced_scan):
                bucket.append(np.asarray(value))
            iteration += 1
        if last_outputs is None:
            final_values = state_values
        else:
            final_values = state_values
        if scan_outputs is None:
            scan_value_infos = self.body.output[1 + len(state_values):]
            stacked_scan = []
            for value_info in scan_value_infos:
                inferred = _graph_value_shape(value_info)
                np_dtype = nn.DTYPE_TO_NUMPY.get(inferred.dtype, np.float32)
                stacked_scan.append(np.empty((0, *inferred.size), dtype=np_dtype))
        else:
            stacked_scan = [np.stack(bucket, axis=0) for bucket in scan_outputs]
        outputs = tuple(_tensor_from_numpy(value) for value in final_values + stacked_scan)
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}

    # Egor Izmaylov: Function `Loop.forward_` performs shape-only inference for Loop, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, m=None, cond=None, *loop_vars):
        outputs = []
        state_count = max(0, len(self.body.input) - 2)
        for value_info in self.body.output[1:1 + state_count]:
            outputs.append(_graph_value_shape(value_info))
        for value_info in self.body.output[1 + state_count:]:
            scan = _graph_value_shape(value_info)
            outputs.append(Tensor_(1, *scan.size, dtype=scan.dtype))
        return {"tensor": outputs[0] if len(outputs) == 1 else tuple(outputs), "parameters": None}


class Scan(Ops):
    # Egor Izmaylov: Function `Scan.__init__` initializes Scan, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(
        self,
        inputs,
        outputs,
        body,
        num_scan_inputs,
        scan_input_axes=None,
        scan_input_directions=None,
        scan_output_axes=None,
        scan_output_directions=None,
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.body = body
        self.num_scan_inputs = int(num_scan_inputs)
        self.scan_input_axes = list(scan_input_axes or [0] * self.num_scan_inputs)
        self.scan_input_directions = list(scan_input_directions or [0] * self.num_scan_inputs)
        self.scan_output_axes = list(scan_output_axes or [])
        self.scan_output_directions = list(scan_output_directions or [])
        self.version = version
        self.outer_scope_names = sorted(_graph_external_names(body))

    # Egor Izmaylov: Function `Scan.forward` executes the concrete runtime path for Scan, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, *inputs):
        return self.forward_with_context(None, *inputs)

    # Egor Izmaylov: Function `Scan.forward_with_context` runs control-flow graph bodies for Scan with an outer-scope feed dictionary so subgraphs can resolve captured values.
    def forward_with_context(self, outer_scope, *inputs):
        num_states = len(inputs) - self.num_scan_inputs
        states = [_tensor_to_numpy(value) for value in inputs[:num_states]]
        scan_inputs = [_tensor_to_numpy(value) for value in inputs[num_states:]]
        body_inputs = [value.name for value in self.body.input]
        body_outputs = [value.name for value in self.body.output]
        trip_count = scan_inputs[0].shape[self.scan_input_axes[0]]
        collected = None
        for iteration in range(trip_count):
            feeds = {name: value for name, value in zip(body_inputs[:num_states], states)}
            for index, value in enumerate(scan_inputs):
                axis = self.scan_input_axes[index] if index < len(self.scan_input_axes) else 0
                take_index = trip_count - 1 - iteration if self.scan_input_directions[index] else iteration
                feeds[body_inputs[num_states + index]] = np.take(value, take_index, axis=axis)
            result = list(_run_graph_proto(self.body, feeds, outer_scope))
            output_map = dict(zip(body_outputs, result))
            states = [np.asarray(output_map[name]) for name in body_outputs[:num_states]]
            scan_values = [np.asarray(output_map[name]) for name in body_outputs[num_states:]]
            if collected is None:
                collected = [[] for _ in scan_values]
            for bucket, value in zip(collected, scan_values):
                bucket.append(value)
        scan_outputs = []
        for index, bucket in enumerate(collected or []):
            values = list(reversed(bucket)) if index < len(self.scan_output_directions) and self.scan_output_directions[index] else bucket
            axis = self.scan_output_axes[index] if index < len(self.scan_output_axes) else 0
            scan_outputs.append(np.stack(values, axis=axis))
        outputs = tuple(_tensor_from_numpy(value) for value in states + scan_outputs)
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}

    # Egor Izmaylov: Function `Scan.forward_` performs shape-only inference for Scan, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, *inputs):
        num_states = len(inputs) - self.num_scan_inputs
        outputs = []
        for value_info in self.body.output[:num_states]:
            outputs.append(_graph_value_shape(value_info))
        for idx, value_info in enumerate(self.body.output[num_states:]):
            elem = _graph_value_shape(value_info)
            scan_input = inputs[num_states + min(idx, self.num_scan_inputs - 1)]
            axis = self.scan_input_axes[min(idx, len(self.scan_input_axes) - 1)] if self.scan_input_axes else 0
            length = scan_input.size[axis]
            out_axis = self.scan_output_axes[idx] if idx < len(self.scan_output_axes) else 0
            shape = list(elem.size)
            shape.insert(out_axis, length)
            outputs.append(Tensor_(*shape, dtype=elem.dtype))
        return {"tensor": outputs[0] if len(outputs) == 1 else tuple(outputs), "parameters": None}


class SequenceMap(Ops):
    # Egor Izmaylov: Function `SequenceMap.__init__` initializes SequenceMap, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, body, version="17"):
        super().__init__(inputs, outputs)
        self.body = body
        self.version = version
        self.outer_scope_names = sorted(_graph_external_names(body))

    # Egor Izmaylov: Function `SequenceMap.forward` executes the concrete runtime path for SequenceMap, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, input_sequence, *additional_inputs):
        return self.forward_with_context(None, input_sequence, *additional_inputs)

    # Egor Izmaylov: Function `SequenceMap.forward_with_context` runs control-flow graph bodies for SequenceMap with an outer-scope feed dictionary so subgraphs can resolve captured values.
    def forward_with_context(self, outer_scope, input_sequence, *additional_inputs):
        body_inputs = [value.name for value in self.body.input]
        collected = None
        for idx, item in enumerate(input_sequence):
            feeds = {body_inputs[0]: _tensor_to_numpy(item)}
            for name, value in zip(body_inputs[1:], additional_inputs):
                feeds[name] = _tensor_to_numpy(value[idx]) if isinstance(value, list) else _tensor_to_numpy(value)
            result = [_tensor_from_numpy(value) for value in _run_graph_proto(self.body, feeds, outer_scope)]
            if collected is None:
                collected = [[] for _ in result]
            for bucket, value in zip(collected, result):
                bucket.append(value)
        outputs = tuple(collected or [])
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}

    # Egor Izmaylov: Function `SequenceMap.forward_` performs shape-only inference for SequenceMap, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, input_sequence, *additional_inputs):
        outputs = tuple([] for _ in self.body.output)
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}

class Binarizer(Ops):
    # Egor Izmaylov: Function `Binarizer.__init__` initializes Binarizer, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, threshold=0.0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.threshold = threshold
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.binarizer_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]

    # Egor Izmaylov: Function `Binarizer.forward` executes the concrete runtime path for Binarizer, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        
        self.lib.binarizer_forward(x_c, out_c, ctypes.c_float(self.threshold))
        
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}

    # Egor Izmaylov: Function `Binarizer.forward_` performs shape-only inference for Binarizer, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class DynamicQuantizeLinear(Ops):
    # Egor Izmaylov: Function `DynamicQuantizeLinear.__init__` initializes DynamicQuantizeLinear, stores constructor-provided state, and prepares later runtime, shape-inference, or verification dispatch.
    def __init__(self, inputs, outputs, dtype="uint8", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = "uint8"
        self.version = version
        if self.lib:
            self.lib.dynamic_quantize_linear_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), 
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)
            ]

    # Egor Izmaylov: Function `DynamicQuantizeLinear.forward` executes the concrete runtime path for DynamicQuantizeLinear, consuming real tensor values and returning the graph-runner value contract.
    def forward(self, x):
        # Outputs: y (uint8), y_scale (float), y_zp (uint8)
        y = Tensor(*x.size, dtype="uint8")
        y_scale = Tensor(1, dtype="float32")
        y_zp = Tensor(1, dtype="uint8")
        
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        y_c = self._numpy_to_ctensor(y.data, "uint8")
        scale_c = self._numpy_to_ctensor(y_scale.data, "float32")
        zp_c = self._numpy_to_ctensor(y_zp.data, "uint8")
        
        self.lib.dynamic_quantize_linear_forward(x_c, y_c, scale_c, zp_c)
        
        y.data = self._ctensor_to_numpy(y_c, "uint8")
        y_scale.data = self._ctensor_to_numpy(scale_c, "float32")
        y_zp.data = self._ctensor_to_numpy(zp_c, "uint8")
        
        self.lib.free_tensor(x_c); self.lib.free_tensor(y_c); self.lib.free_tensor(scale_c); self.lib.free_tensor(zp_c)
        
        return {"tensor": [y, y_scale, y_zp], "parameters": None}

    # Egor Izmaylov: Function `DynamicQuantizeLinear.forward_` performs shape-only inference for DynamicQuantizeLinear, returning `Tensor_` metadata without touching numeric storage or C backend buffers.
    def forward_(self, x):
        return {
            "tensor": [Tensor_(*x.size, dtype="uint8"), Tensor_(1, dtype="float32"), Tensor_(1, dtype="uint8")],
            "parameters": None
        }
