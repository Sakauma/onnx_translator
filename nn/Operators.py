from nn import Ops
from nn import Tensor, Tensor_, DTYPE_MAP, CTensor
import nn
import ctypes
import numpy as np
from typing import List, Union
import os

class CConvParams(ctypes.Structure):
    _fields_ = [
        ("pads", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("dilations", ctypes.POINTER(ctypes.c_int)),
        ("group", ctypes.c_int)
    ]

class CPoolParams(ctypes.Structure):
    _fields_ = [
        ("pads", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("dilations", ctypes.POINTER(ctypes.c_int)),
        ("kernel_shape", ctypes.POINTER(ctypes.c_int))
    ]
    
class CReduceParams(ctypes.Structure):
    _fields_ = [
        ("axes", ctypes.POINTER(ctypes.c_int)),
        ("num_axes", ctypes.c_int),
        ("keepdims", ctypes.c_int)
    ]

class RELU(Ops):
    """ReLU激活函数操作类"""
    
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

    def forward(self, input: Tensor) -> Tensor:
        """
        ReLU函数的C后端实现，使用真实数据进行计算
        
        Args:
            input: 输入张量
            
        Returns:
            Tensor: 经过ReLU激活后的输出张量
        """
        out_tensor = self._execute_unary(input, "relu_forward")
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

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

    def forward(self, input: Tensor) -> Tensor:
        """
        Abs函数的C后端实现，使用真实数据进行计算
        """
        out_tensor = self._execute_unary(input, "abs_forward")
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

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

    def __init__(self, inputs, outputs, dtype, version="17"):
        super(SUB, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

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

    def __init__(self, inputs, outputs, dtype, version="17"):
        super(MUL, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

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

    def __init__(self, inputs, outputs, dtype, version="17"):
        super(DIV, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

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
    def __init__(self, inputs, outputs, axis=1, dtype=None, version="17"):
        super(QuantizeLinear, self).__init__(inputs, outputs)
        self.dtype = dtype 
        self.axis = axis # 保存 axis
        self.version = version

    def forward(self, x, y_scale, y_zero_point) -> Tensor:
        if y_scale.data.ndim == 1 and x.data.ndim > 1:
            new_shape = [1] * x.data.ndim
            safe_axis = self.axis if self.axis >= 0 else self.axis + x.data.ndim
            if safe_axis < x.data.ndim:
                new_shape[safe_axis] = y_scale.data.size
            y_scale.data = y_scale.data.reshape(new_shape)
            y_zero_point.data = y_zero_point.data.reshape(new_shape)
            
        out_tensor = self._execute_ternary(x, y_scale, y_zero_point, "quantize_linear_forward")
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

class DequantizeLinear(Ops):
    def __init__(self, inputs, outputs, dtype=None, version="17"):
        super(DequantizeLinear, self).__init__(inputs, outputs)
        self.dtype = dtype # 通常为 float32
        self.version = version

    def forward(self, x, x_scale, x_zero_point) -> Tensor:
        out_tensor = self._execute_ternary(x, x_scale, x_zero_point, "dequantize_linear_forward")
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, x, x_scale, x_zero_point) -> Tensor_:
        try:
            bcast_shape = np.broadcast_shapes(x.size, x_scale.size, x_zero_point.size)
        except:
            bcast_shape = x.size

        output_tensor = Tensor_(*bcast_shape, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values
    
class Conv(Ops):
    def __init__(self, inputs, outputs, pads, strides, dilations, group, dtype, version="17"):
        super(Conv, self).__init__(inputs, outputs)
        # 必须完整保存所有参数
        self.pads = pads         # [top, left, bottom, right]
        self.strides = strides   # [h, w]
        self.dilations = dilations # [h, w]
        self.group = group
        self.dtype = dtype
        self.version = version

        # 注册 C 函数参数类型
        if self.lib:
            self.lib.conv2d_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), 
                ctypes.POINTER(CTensor), ctypes.POINTER(CConvParams)
            ]

    def forward(self, x: Tensor, w: Tensor, b: Tensor = None) -> dict:
        # 1. 计算输出形状
        N, C_in, H_in, W_in = x.size
        M, C_in_g, K_h, K_w = w.size # W: [OutCh, InCh/Group, KH, KW]
        
        # Output Size 公式
        H_out = (H_in + self.pads[0] + self.pads[2] - self.dilations[0] * (K_h - 1) - 1) // self.strides[0] + 1
        W_out = (W_in + self.pads[1] + self.pads[3] - self.dilations[1] * (K_w - 1) - 1) // self.strides[1] + 1
        out_shape = (N, M, H_out, W_out)
        
        # 2. 准备 C 参数
        pads_arr = (ctypes.c_int * 4)(*self.pads)
        strides_arr = (ctypes.c_int * 2)(*self.strides)
        dilations_arr = (ctypes.c_int * 2)(*self.dilations)
        
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

    def forward_(self, x: Tensor_, w: Tensor_, b: Tensor_ = None) -> dict:
        # 仅做形状推断
        N, C_in, H_in, W_in = x.size
        M, C_in_g, K_h, K_w = w.size
        H_out = (H_in + self.pads[0] + self.pads[2] - self.dilations[0] * (K_h - 1) - 1) // self.strides[0] + 1
        W_out = (W_in + self.pads[1] + self.pads[3] - self.dilations[1] * (K_w - 1) - 1) // self.strides[1] + 1
        
        output_tensor = Tensor_(N, M, H_out, W_out, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

class MaxPool(Ops):
    def __init__(self, inputs, outputs, kernel_shape, pads, strides, dtype, dilations=[1, 1], version="17"):
        super(MaxPool, self).__init__(inputs, outputs)
        self.kernel_shape = kernel_shape
        self.pads = pads
        self.strides = strides
        self.dilations = dilations
        self.dtype = dtype
        self.version = version

        if self.lib:
            self.lib.max_pool_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CPoolParams)
            ]

    def forward(self, x: Tensor) -> dict:
        # 1. 计算输出形状
        N, C, H_in, W_in = x.size
        K_h, K_w = self.kernel_shape
        H_out = (H_in + self.pads[0] + self.pads[2] - self.dilations[0] * (K_h - 1) - 1) // self.strides[0] + 1
        W_out = (W_in + self.pads[1] + self.pads[3] - self.dilations[1] * (K_w - 1) - 1) // self.strides[1] + 1
        out_shape = (N, C, H_out, W_out)

        # 2. 准备参数
        pads_arr = (ctypes.c_int * 4)(*self.pads)
        strides_arr = (ctypes.c_int * 2)(*self.strides)
        kernel_arr = (ctypes.c_int * 2)(*self.kernel_shape)
        dilations_arr = (ctypes.c_int * 2)(*self.dilations)
        
        c_params = CPoolParams()
        c_params.pads = ctypes.cast(pads_arr, ctypes.POINTER(ctypes.c_int))
        c_params.strides = ctypes.cast(strides_arr, ctypes.POINTER(ctypes.c_int))
        c_params.kernel_shape = ctypes.cast(kernel_arr, ctypes.POINTER(ctypes.c_int))
        c_params.dilations = ctypes.cast(dilations_arr, ctypes.POINTER(ctypes.c_int))

        # 3. 准备 Tensor
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        output_shape_c = (ctypes.c_int * 4)(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, 4, DTYPE_MAP[self.dtype])

        # 4. 执行
        self.lib.max_pool_forward(x_c, output_c, ctypes.byref(c_params))

        # 5. 返回
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(x_c)
        self.lib.free_tensor(output_c)

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, x: Tensor_) -> dict:
        N, C, H_in, W_in = x.size
        K_h, K_w = self.kernel_shape
        H_out = (H_in + self.pads[0] + self.pads[2] - K_h) // self.strides[0] + 1
        W_out = (W_in + self.pads[1] + self.pads[3] - K_w) // self.strides[1] + 1
        output_tensor = Tensor_(N, C, H_out, W_out, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

class Gemm(Ops):
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

    def forward(self, A: Tensor, B: Tensor, C: Tensor = None) -> dict:
        # 维度推断 (假设 A, B 至少 2D)
        M = A.size[0] if self.transA == 0 else A.size[1]
        N = B.size[1] if self.transB == 0 else B.size[0]
        out_shape = (M, N)

        a_c = self._numpy_to_ctensor(A.data, A.dtype)
        b_c = self._numpy_to_ctensor(B.data, B.dtype)
        c_c = self._numpy_to_ctensor(C.data, C.dtype) if C is not None else ctypes.POINTER(CTensor)()

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

    def forward_(self, A: Tensor_, B: Tensor_, C: Tensor_ = None) -> dict:
        M = A.size[0] if self.transA == 0 else A.size[1]
        N = B.size[1] if self.transB == 0 else B.size[0]
        output_tensor = Tensor_(M, N, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

class Softmax(Ops):
    def __init__(self, inputs, outputs, axis, dtype, version="17"):
        super(Softmax, self).__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        
        if self.lib:
            self.lib.softmax_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int
            ]

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

    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values
    
class EXP(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(EXP, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "exp_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class LOG(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(LOG, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "log_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class SQRT(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(SQRT, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "sqrt_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class SIGMOID(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(SIGMOID, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "sigmoid_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class TANH(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(TANH, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "tanh_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}
    
class Flatten(Ops):
    def __init__(self, inputs, outputs, axis=1, dtype="float32", version="17"):
        super(Flatten, self).__init__(inputs, outputs)
        self.axis = axis
        self.dtype = dtype
        self.version = version

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

    def forward(self, input: Tensor) -> dict:
        out_shape = self._calc_shape(input.size)
        
        # 调用 C 的 Flatten (本质是 Copy)
        input_c = self._numpy_to_ctensor(input.data, self.dtype)
        output_shape_c = (ctypes.c_int * 2)(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, 2, nn.DTYPE_MAP[self.dtype])
        
        self.lib.flatten_forward(input_c, output_c)
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, input: Tensor_) -> dict:
        out_shape = self._calc_shape(input.size)
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

class Reshape(Ops):
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super(Reshape, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, data: Tensor, shape: Tensor) -> dict:
        # 从 shape tensor 中读取目标形状
        # shape tensor 通常是 int64，我们在 Python 层将其转为 list
        target_shape = shape.data.astype(np.int64).flatten().tolist()
        
        # 处理 -1 (自动推断维度)
        total_size = data.data_size
        infer_idx = -1
        current_size = 1
        for i, s in enumerate(target_shape):
            if s == -1:
                infer_idx = i
            else:
                current_size *= s
        
        if infer_idx != -1:
            target_shape[infer_idx] = total_size // current_size
            
        final_shape = tuple(target_shape)

        # C 调用
        input_c = self._numpy_to_ctensor(data.data, self.dtype)
        output_shape_c = (ctypes.c_int * len(final_shape))(*final_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(final_shape), nn.DTYPE_MAP[self.dtype])
        
        self.lib.reshape_forward(input_c, output_c)
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)

        out_tensor = Tensor(*final_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, data: Tensor_, shape: Tensor_) -> dict:
        # 在图推断模式下，如果不传真实数据，很难知道 shape tensor 的具体值
        # 这里做个简化假设：如果是在 ModelInitParas 里生成的 Graph，Shape 应该是已知的
        # 如果无法推断，返回一个占位符
        # 为了演示，假设 output 是 Tensor_ 并拥有一个假定 shape
        # 实际严谨实现需要 Constant Folding 支持
        output_tensor = Tensor_(1, dtype=self.dtype) # 无法推断，给个 Dummy
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

class Transpose(Ops):
    def __init__(self, inputs, outputs, perm, dtype="float32", version="17"):
        super(Transpose, self).__init__(inputs, outputs)
        self.perm = perm
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.transpose_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor), ctypes.POINTER(ctypes.c_int)
            ]

    def forward(self, input: Tensor) -> dict:
        # 计算输出形状
        out_shape = [input.size[i] for i in self.perm]
        
        input_c = self._numpy_to_ctensor(input.data, self.dtype)
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
        
        # 传入 perm 数组
        perm_arr = (ctypes.c_int * len(self.perm))(*self.perm)
        
        self.lib.transpose_forward(input_c, output_c, perm_arr)
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, input: Tensor_) -> dict:
        out_shape = [input.size[i] for i in self.perm]
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values
    
class Pow(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Pow, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input_a: Tensor, input_b: Tensor) -> dict:
        out_tensor = self._execute_binary(input_a, input_b, "pow_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    def forward_(self, input_a: Tensor_, input_b: Tensor_) -> dict:
        # 简单广播推断
        try:
            bcast = np.broadcast_shapes(input_a.size, input_b.size)
        except:
            bcast = input_a.size
        output_tensor = Tensor_(*bcast, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Max(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Max, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input_a: Tensor, input_b: Tensor) -> dict:
        out_tensor = self._execute_binary(input_a, input_b, "max_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    def forward_(self, input_a: Tensor_, input_b: Tensor_) -> dict:
        try:
            bcast = np.broadcast_shapes(input_a.size, input_b.size)
        except:
            bcast = input_a.size
        output_tensor = Tensor_(*bcast, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Min(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Min, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input_a: Tensor, input_b: Tensor) -> dict:
        out_tensor = self._execute_binary(input_a, input_b, "min_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    def forward_(self, input_a: Tensor_, input_b: Tensor_) -> dict:
        try:
            bcast = np.broadcast_shapes(input_a.size, input_b.size)
        except:
            bcast = input_a.size
        output_tensor = Tensor_(*bcast, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Squeeze(Ops):
    def __init__(self, inputs, outputs, axes=None, dtype="float32", version="17"):
        super(Squeeze, self).__init__(inputs, outputs)
        self.axes = axes
        self.dtype = dtype
        self.version = version

    def _calc_shape(self, in_shape, axes):
        # 如果 axes 为 None，挤压所有为 1 的维度
        new_shape = []
        # 处理 axes 归一化 (支持负数索引)
        ndim = len(in_shape)
        if axes is not None:
            norm_axes = [ax + ndim if ax < 0 else ax for ax in axes]
        else:
            norm_axes = None

        for i, dim in enumerate(in_shape):
            if norm_axes is not None:
                if i in norm_axes and dim == 1:
                    continue # Squeeze
                new_shape.append(dim)
            else:
                if dim != 1:
                    new_shape.append(dim)
        return tuple(new_shape)

    def forward(self, data: Tensor, axes: Tensor = None) -> dict:
        # axes 是输入 tensor，不是属性
        target_axes = self.axes
        if axes is not None:
            target_axes = axes.data.flatten().tolist()
        
        out_shape = self._calc_shape(data.size, target_axes)
        
        # 复用 Reshape/Flatten 的逻辑：直接内存拷贝
        input_c = self._numpy_to_ctensor(data.data, self.dtype)
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
        
        # 借用 reshape_forward 
        self.lib.reshape_forward(input_c, output_c)
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    def forward_(self, data: Tensor_, axes: Tensor_ = None) -> dict:
        # 图推断模式下，如果 axes 是动态输入，很难推断 shape
        # 这里做简化假设
        out_shape = data.size # 无法推断时保持原样，或返回 Dummy
        if self.axes is not None:
             out_shape = self._calc_shape(data.size, self.axes)
        
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Unsqueeze(Ops):
    def __init__(self, inputs, outputs, axes=None, dtype="float32", version="17"):
        super(Unsqueeze, self).__init__(inputs, outputs)
        self.axes = axes
        self.dtype = dtype
        self.version = version

    def _calc_shape(self, in_shape, axes):
        # Unsqueeze: 在指定位置插入维度 1
        # 排序 axes 以便按顺序插入
        axes = sorted([ax + len(in_shape) + 1 if ax < 0 else ax for ax in axes])
        new_shape = list(in_shape)
        for ax in axes:
            new_shape.insert(ax, 1)
        return tuple(new_shape)

    def forward(self, data: Tensor, axes: Tensor = None) -> dict:
        target_axes = self.axes
        if axes is not None:
            target_axes = axes.data.flatten().tolist()
            
        out_shape = self._calc_shape(data.size, target_axes)
        
        input_c = self._numpy_to_ctensor(data.data, self.dtype)
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
        
        self.lib.reshape_forward(input_c, output_c)
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    def forward_(self, data: Tensor_, axes: Tensor_ = None) -> dict:
        out_shape = data.size
        if self.axes is not None:
            out_shape = self._calc_shape(data.size, self.axes)
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}
    
class Concat(Ops):
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

    def _calc_shape(self, input_tensors):
        # 假设除 axis 外其他维度一致
        base_shape = list(input_tensors[0].size)
        ndim = len(base_shape)
        axis = self.axis if self.axis >= 0 else self.axis + ndim
        
        total_dim = 0
        for t in input_tensors:
            total_dim += t.size[axis]
        
        base_shape[axis] = total_dim
        return tuple(base_shape), axis

    def forward(self, *inputs) -> dict:
        input_list = list(inputs)
        out_shape, axis = self._calc_shape(input_list)
        
        # 构建 C 指针数组
        CTensorPtr = ctypes.POINTER(nn.CTensor)
        InputArrayType = CTensorPtr * len(input_list)
        input_array = InputArrayType()
        
        # 防止 GC 回收
        c_refs = []
        for idx, tensor in enumerate(input_list):
            c_t = self._numpy_to_ctensor(tensor.data, tensor.dtype)
            input_array[idx] = c_t
            c_refs.append(c_t)
            
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
        
        self.lib.concat_forward(input_array, len(input_list), output_c, axis)
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        
        self.lib.free_tensor(output_c)
        for c_t in c_refs:
            self.lib.free_tensor(c_t)

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, *inputs) -> dict:
        input_list = list(inputs)
        out_shape, _ = self._calc_shape(input_list)
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

class Slice(Ops):
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super(Slice, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.slice_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor), 
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
            ]

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
            
            s = max(0, min(s, dim_len))
            e = max(0, min(e, dim_len))
            
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
            
        input_c = self._numpy_to_ctensor(data.data, self.dtype)
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
        
        c_starts = (ctypes.c_int * ndim)(*full_starts)
        c_steps = (ctypes.c_int * ndim)(*full_steps)
        
        self.lib.slice_forward(input_c, output_c, c_starts, c_steps)
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, data: Tensor_, starts: Tensor_, ends: Tensor_, axes: Tensor_ = None, steps: Tensor_ = None) -> dict:
        # 图推断模式下，暂时无法精确推断 Slice 输出形状，假设与输入一致
        output_tensor = Tensor_(*data.size, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values
    
class Neg(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Neg, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "neg_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Reciprocal(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Reciprocal, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "reciprocal_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Ceil(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Ceil, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "ceil_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Floor(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Floor, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "floor_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}

class Cast(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Cast, self).__init__(inputs, outputs)
        self.dtype = dtype # 这里的 dtype 就是目标类型
        self.version = version
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "cast_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}
    
class Clip(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Clip, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        
        if self.lib:
             self.lib.clip_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor), 
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor)
            ]

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
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(MatMul, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input_a: Tensor, input_b: Tensor) -> dict:
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
            final_shape.pop(-1) # 删除 N (此时 N=1)
            
        if is_a_1d:
            final_shape.pop(-1) # 删除 M (此时 M=1，因为 N 已经被 pop 了或者还在最后)
            
        # 如果变成了标量或形状改变，reshape 数据
        if tuple(final_shape) != tuple(out_shape_for_c):
            out_data = out_data.reshape(final_shape)

        out_tensor = Tensor(*final_shape, dtype=self.dtype, data=out_data)
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, input_a: Tensor_, input_b: Tensor_) -> dict:
        shape_a = list(input_a.size) if isinstance(input_a.size, (list, tuple)) else [input_a.size]
        shape_b = list(input_b.size) if isinstance(input_b.size, (list, tuple)) else [input_b.size]
        
        is_a_1d = (len(shape_a) == 1)
        is_b_1d = (len(shape_b) == 1)
        
        if is_a_1d: shape_a = [1] + shape_a
        if is_b_1d: shape_b = shape_b + [1]
            
        M = shape_a[-2]
        N = shape_b[-1]
        
        batch_a = shape_a[:-2]
        batch_b = shape_b[:-2]
        
        try:
            batch_out = np.broadcast_shapes(batch_a, batch_b)
        except:
            batch_out = batch_a # Fallback
            
        temp_out_shape = list(batch_out) + [M, N]
        
        # 还原形状
        final_shape = list(temp_out_shape)
        if is_b_1d: final_shape.pop(-1)
        if is_a_1d: final_shape.pop(-1)
        
        output_tensor = Tensor_(*final_shape, dtype=self.dtype)
        values = {"tensor": output_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values
    
class Gather(Ops):
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

    def forward(self, data: Tensor, indices: Tensor) -> dict:
        # 计算输出形状: data.shape[:axis] + indices.shape + data.shape[axis+1:]
        axis = self.axis if self.axis >= 0 else self.axis + len(data.size)
        out_shape = data.size[:axis] + indices.size + data.size[axis+1:]
        
        data_c = self._numpy_to_ctensor(data.data, data.dtype)
        indices_c = self._numpy_to_ctensor(indices.data, indices.dtype)
        
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
        
        self.lib.gather_forward(data_c, indices_c, output_c, ctypes.c_int(axis))
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(data_c); self.lib.free_tensor(indices_c); self.lib.free_tensor(output_c)
        
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    def forward_(self, data: Tensor_, indices: Tensor_) -> dict:
        axis = self.axis if self.axis >= 0 else self.axis + len(data.size)
        d_size = list(data.size) if isinstance(data.size, tuple) else data.size
        i_size = list(indices.size) if isinstance(indices.size, tuple) else indices.size
        
        out_shape = tuple(d_size[:axis] + i_size + d_size[axis+1:])
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None, "graph": None}
    
class Expand(Ops):
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super(Expand, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

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
        
        input_c = self._numpy_to_ctensor(input.data, input.dtype)
        output_shape_c = (ctypes.c_int * len(final_shape))(*final_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(final_shape), nn.DTYPE_MAP[self.dtype])
        
        self.lib.expand_forward(input_c, output_c)
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)
        
        return {"tensor": Tensor(*final_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    def forward_(self, input: Tensor_, shape: Tensor_) -> dict:
        # 静态图推断较难知道 shape 的具体值，暂返回未知大小的 Tensor_
        return {"tensor": Tensor_(1, dtype=self.dtype), "parameters": None, "graph": None}
    
class Shape(Ops):
    def __init__(self, inputs, outputs, start=0, end=None, dtype="int64", version="17"):
        super(Shape, self).__init__(inputs, outputs)
        self.start = start
        self.end = end
        self.dtype = "int64" # Shape 输出永远是 int64
        self.version = version

    def forward(self, input: Tensor) -> dict:
        dims = list(input.size)
        # 处理 start/end
        if self.end is None: self.end = len(dims)
        sliced_dims = dims[self.start : self.end]
        
        out_data = np.array(sliced_dims, dtype=np.int64)
        out_tensor = Tensor(len(sliced_dims), dtype="int64", data=out_data)
        
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    def forward_(self, input: Tensor_) -> dict:
        # Shape 的输出形状取决于 input 的 rank
        dims = list(input.size)
        if self.end is None: self.end = len(dims)
        out_len = len(dims[self.start : self.end])
        return {"tensor": Tensor_(out_len, dtype="int64"), "parameters": None, "graph": None}
    
class Constant(Ops):
    def __init__(self, inputs, outputs, value=None, dtype="float32", version="17"):
        super(Constant, self).__init__(inputs, outputs)
        self.value = value
        self.dtype = dtype
        self.version = version

    def forward(self) -> dict:
        if isinstance(self.value, np.ndarray):
            val_data = self.value
            val_shape = self.value.shape
        elif isinstance(self.value, Tensor):
            val_data = self.value.data
            val_shape = self.value.size
        else: # Scalar
            val_data = np.array([self.value])
            val_shape = (1,)

        input_c = self._numpy_to_ctensor(val_data, self.dtype)
        output_shape_c = (ctypes.c_int * len(val_shape))(*val_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(val_shape), nn.DTYPE_MAP[self.dtype])
        self.lib.flatten_forward(input_c, output_c)
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)

        return {"tensor": Tensor(*val_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    def forward_(self) -> dict:
        # 图推断模式
        shape = self.value.shape if hasattr(self.value, 'shape') else (1,)
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None, "graph": None}
    
class Equal(Ops):
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "equal_forward"), "parameters": None}
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class Greater(Ops):
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "greater_forward"), "parameters": None}
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class Less(Ops):
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "less_forward"), "parameters": None}
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class GreaterOrEqual(Ops):
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "greater_or_equal_forward"), "parameters": None}
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class LessOrEqual(Ops):
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "less_or_equal_forward"), "parameters": None}
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class Not(Ops):
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "not_forward"), "parameters": None}
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class And(Ops):
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "and_forward"), "parameters": None}
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class Or(Ops):
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "or_forward"), "parameters": None}
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class Xor(Ops):
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "xor_forward"), "parameters": None}
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class IsNaN(Ops):
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "isnan_forward"), "parameters": None}
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Sin(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "sin_forward"), "parameters": None}
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Tan(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "tan_forward"), "parameters": None}
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Atan(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "atan_forward"), "parameters": None}
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Sign(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "sign_forward"), "parameters": None}
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}
        
class Identity(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, x):
        return {"tensor": self._execute_unary(x, "identity_forward"), "parameters": None}
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}

class Mod(Ops):
    def __init__(self, inputs, outputs, dtype, fmod=0, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.fmod = fmod 
        self.version = version
    def forward(self, a, b):
        return {"tensor": self._execute_binary(a, b, "mod_forward"), "parameters": None}
    def forward_(self, a, b):
        try: shape = np.broadcast_shapes(a.size, b.size)
        except: shape = a.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}

class Where(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    def forward(self, cond, x, y):
        return {"tensor": self._execute_ternary(cond, x, y, "where_forward"), "parameters": None}
    def forward_(self, cond, x, y):
        try: shape = np.broadcast_shapes(cond.size, x.size, y.size)
        except: shape = x.size
        return {"tensor": Tensor_(*shape, dtype=self.dtype), "parameters": None}
    
class ConstantOfShape(Ops):
    def __init__(self, inputs, outputs, value=None, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.value_tensor = None
        # 如果 value 属性存在，它是一个 TensorProto (Numpy array)
        if value is not None:
             # 创建一个单元素 Tensor 包装这个值
             self.value_tensor = Tensor(1, dtype=dtype, data=value.flatten())
        else:
             # 默认值为 0.0
             self.value_tensor = Tensor(1, dtype="float32", data=np.array([0.0], dtype=np.float32))
        
        self.dtype = dtype
        self.version = version

    def forward(self, shape_tensor):
        # 1. 从 shape_tensor 读取目标形状
        target_shape = shape_tensor.data.astype(np.int64).flatten().tolist()
        if not target_shape: target_shape = [1] # Handle scalar case if needed
        
        # 2. 创建输出
        output_tensor = Tensor(*target_shape, dtype=self.dtype)
        
        # 3. 调用 C
        out_c = self._numpy_to_ctensor(output_tensor.data, self.dtype) # 此时 data 是全0的
        val_c = self._numpy_to_ctensor(self.value_tensor.data, self.value_tensor.dtype)
        
        self.lib.constant_of_shape_forward(out_c, val_c)
        
        out_data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(out_c)
        self.lib.free_tensor(val_c)
        
        output_tensor.data = out_data
        return {"tensor": output_tensor, "parameters": None}

    def forward_(self, shape_tensor):
        # 静态图无法得知 shape_tensor 具体数值，返回一个 Dummy
        return {"tensor": Tensor_(1, dtype=self.dtype), "parameters": None}

class Range(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

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

    def forward_(self, start, limit, delta):
        # 无法推断具体长度
        return {"tensor": Tensor_(1, dtype=self.dtype), "parameters": None}

class Tile(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input, repeats):
        rep = repeats.data.astype(np.int64).flatten()
        in_shape = np.array(input.size)
        if len(rep) != len(in_shape):
            raise ValueError(f"Tile: repeats dim {len(rep)} != input dim {len(in_shape)}")
            
        out_shape = tuple((in_shape * rep).tolist())
        input_c = self._numpy_to_ctensor(input.data, self.dtype)
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
        self.lib.tile_forward(input_c, output_c)
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c); self.lib.free_tensor(output_c)
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    def forward_(self, input, repeats):
        return {"tensor": Tensor_(1, dtype=self.dtype), "parameters": None}

class Pad(Ops):
    def __init__(self, inputs, outputs, mode="constant", dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.mode = mode # constant, reflect, edge
        self.dtype = dtype
        self.version = version
        self.mode_int = {"constant": 0, "reflect": 1, "edge": 2}.get(mode, 0)

    def forward(self, data, pads, constant_value=None):
        # pads: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        p = pads.data.astype(np.int64).flatten()
        ndim = len(data.size)
        pad_begins = p[:ndim]
        pad_ends = p[ndim:]
        
        out_shape = []
        for i in range(ndim):
            out_shape.append(data.size[i] + pad_begins[i] + pad_ends[i])
        out_shape = tuple(out_shape)
        
        data_c = self._numpy_to_ctensor(data.data, self.dtype)
        pads_c = self._numpy_to_ctensor(pads.data, "int64")
        const_c = self._numpy_to_ctensor(constant_value.data, constant_value.dtype) if constant_value else ctypes.POINTER(CTensor)()
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
        self.lib.pad_forward(data_c, output_c, pads_c, const_c, ctypes.c_int(self.mode_int))
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(data_c); self.lib.free_tensor(pads_c); self.lib.free_tensor(output_c)
        if constant_value: self.lib.free_tensor(const_c)
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}
    
    def forward_(self, data, pads, constant_value=None):
        return {"tensor": Tensor_(*data.size, dtype=self.dtype), "parameters": None}

class Split(Ops):
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

    def forward(self, input, split=None):
        axis = self.axis if self.axis >= 0 else self.axis + len(input.size)
        dim_len = input.size[axis]
        
        if split is not None:
            split_sizes = split.data.astype(np.int64).flatten().tolist()
        else:
            num_outputs = len(self.outputs)
            div = dim_len // num_outputs
            split_sizes = [div] * num_outputs
            if dim_len % num_outputs != 0:
                 split_sizes[-1] += dim_len % num_outputs
        result_tensors = []
        current_start = 0
        input_c = self._numpy_to_ctensor(input.data, self.dtype)
        ndim = len(input.size)
        
        for size in split_sizes:
            starts = [0] * ndim
            steps = [1] * ndim
            
            out_shape = list(input.size)
            out_shape[axis] = size
            out_shape = tuple(out_shape)
            starts[axis] = current_start
            output_shape_c = (ctypes.c_int * ndim)(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, ndim, nn.DTYPE_MAP[self.dtype])
            c_starts = (ctypes.c_int * ndim)(*starts)
            c_steps = (ctypes.c_int * ndim)(*steps)
            self.lib.slice_forward(input_c, output_c, c_starts, c_steps)
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(output_c)
            result_tensors.append(Tensor(*out_shape, dtype=self.dtype, data=out_data))
            current_start += size
            
        self.lib.free_tensor(input_c)
        return {"tensor": result_tensors, "parameters": None}

    def forward_(self, input, split=None):
        # 返回与输出数量相同的占位符列表
        return {"tensor": [Tensor_(1, dtype=self.dtype) for _ in self.outputs], "parameters": None}
    
# Reduce 基类，复用 Shape 计算逻辑
class ReduceBase(Ops):
    def __init__(self, inputs, outputs, axes=None, keepdims=1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axes = axes # 初始 axes，可能为 None
        self.keepdims = keepdims
        self.dtype = dtype
        self.version = version

        # 注册参数类型
        if self.lib:
            func_name = self._get_c_func_name()
            if hasattr(self.lib, func_name):
                getattr(self.lib, func_name).argtypes = [
                    ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CReduceParams)
                ]

    def _get_c_func_name(self):
        raise NotImplementedError

    def _prepare_axes(self, input_shape, runtime_axes=None):
        ndim = len(input_shape)
        # 优先级: 运行时输入 > 属性 > 默认(全归约)
        target_axes = None
        
        if runtime_axes is not None:
            # 如果 axes 是作为 Tensor 输入传进来的
            target_axes = runtime_axes.data.astype(np.int64).flatten().tolist()
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

    def forward(self, data, axes_tensor=None):
        real_axes = self._prepare_axes(data.size, axes_tensor)
        out_shape = self._calc_out_shape(data.size, real_axes)
        
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

    def forward_(self, data, axes_tensor=None):
        # 静态推断：如果没有 axes_tensor 且 self.axes 为 None，假设全归约
        # 如果有 axes_tensor，无法推断具体轴，只能假设输出 Rank (如果 keepdims=1)
        real_axes = self._prepare_axes(data.size, None) # 忽略动态 axes
        out_shape = self._calc_out_shape(data.size, real_axes)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}

class ReduceMean(ReduceBase):
    def _get_c_func_name(self): return "reduce_mean_forward"

class ReduceSum(ReduceBase):
    def _get_c_func_name(self): return "reduce_sum_forward"

class ReduceMax(ReduceBase):
    def _get_c_func_name(self): return "reduce_max_forward"

class ReduceMin(ReduceBase):
    def _get_c_func_name(self): return "reduce_min_forward"

class ReduceProd(ReduceBase):
    def _get_c_func_name(self): return "reduce_prod_forward"
    
class ArgBase(Ops):
    def __init__(self, inputs, outputs, axis=0, keepdims=1, select_last_index=0, dtype="int64", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.keepdims = keepdims
        self.select_last_index = select_last_index
        self.dtype = "int64" # ArgMax 输出必定是索引
        self.version = version

    def _get_c_func_name(self): raise NotImplementedError

    def forward(self, data):
        ndim = len(data.size)
        axis = self.axis if self.axis >= 0 else self.axis + ndim
        
        out_shape = list(data.size)
        if self.keepdims:
            out_shape[axis] = 1
        else:
            out_shape.pop(axis)
        out_shape = tuple(out_shape)
        
        input_c = self._numpy_to_ctensor(data.data, data.dtype)
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
        
        getattr(self.lib, self._get_c_func_name())(
            input_c, output_c, ctypes.c_int(axis), ctypes.c_int(self.select_last_index)
        )
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        self.lib.free_tensor(input_c); self.lib.free_tensor(output_c)
        
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    def forward_(self, data):
        ndim = len(data.size)
        axis = self.axis if self.axis >= 0 else self.axis + ndim
        out_shape = list(data.size)
        if self.keepdims: out_shape[axis] = 1
        else: out_shape.pop(axis)
        return {"tensor": Tensor_(*tuple(out_shape), dtype=self.dtype), "parameters": None}

class ArgMax(ArgBase):
    def _get_c_func_name(self): return "argmax_forward"

class ArgMin(ArgBase):
    def _get_c_func_name(self): return "argmin_forward"