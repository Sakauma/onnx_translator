from nn import Ops
from nn import Tensor, Tensor_
import nn
import ctypes
import numpy as np
from typing import List, Union
import os


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
        # 将输入转换为C张量
        input_c = self._numpy_to_ctensor(input.data, self.dtype)

        # 创建输出C张量
        output_shape = (ctypes.c_int * len(input.size))(*input.size)
        output_c = self.lib.create_tensor(output_shape, len(input.size), nn.DTYPE_MAP[self.dtype])

        # 调用C函数
        self.lib.relu_forward(input_c, output_c)

        # 转换回numpy并创建输出张量
        output_data = self._ctensor_to_numpy(output_c, self.dtype)
        output_tensor = Tensor(*input.size, dtype=self.dtype, data=output_data)

        # 清理资源
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
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
        output_tensor = input
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
        # 将输入转换为C张量
        input_c = self._numpy_to_ctensor(input.data, self.dtype)

        # 创建输出C张量
        output_shape = (ctypes.c_int * len(input.size))(*input.size)
        output_c = self.lib.create_tensor(output_shape, len(input.size), nn.DTYPE_MAP[self.dtype])

        # 调用C函数
        self.lib.cos_forward(input_c, output_c)

        # 转换回numpy并创建输出张量
        output_data = self._ctensor_to_numpy(output_c, self.dtype)
        output_tensor = Tensor(*input.size, dtype=self.dtype, data=output_data)

        # 清理资源
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
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
        output_tensor = input
        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values


class ADD(Ops):
    """加法操作类"""

    def __init__(self, inputs, outputs, dtype, version="17"):
        super(ADD, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input1: Tensor, input2: Tensor) -> dict:
        """
        加法操作的C后端实现 - 简化版本，假设形状相同

        Args:
            input1: 第一个输入张量
            input2: 第二个输入张量

        Returns:
            dict: 包含输出张量的字典
        """
        # 检查形状是否相同
        if input1.size != input2.size:
            raise ValueError(f"ADD操作输入形状不匹配: {input1.size} vs {input2.size}")

        print(f"DEBUG: 开始ADD forward, 输入形状: {input1.size}")

        # 将输入转换为C张量
        input1_c = self._numpy_to_ctensor(input1.data, self.dtype)
        input2_c = self._numpy_to_ctensor(input2.data, self.dtype)

        # 创建输出C张量（使用与输入相同的形状）
        output_shape = input1.size
        output_shape_c = (ctypes.c_int * len(output_shape))(*output_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(output_shape), nn.DTYPE_MAP[self.dtype])

        print(f"DEBUG: C张量创建成功，准备调用add_forward")

        try:
            # 调用C函数
            self.lib.add_forward(input1_c, input2_c, output_c)
            print("DEBUG: add_forward调用成功")
        except Exception as e:
            print(f"DEBUG: add_forward调用失败: {e}")
            raise

        # 转换回numpy并创建输出张量
        output_data = self._ctensor_to_numpy(output_c, self.dtype)
        output_tensor = Tensor(*output_shape, dtype=self.dtype, data=output_data)

        # 清理资源
        self.lib.free_tensor(input1_c)
        self.lib.free_tensor(input2_c)
        self.lib.free_tensor(output_c)

        # 返回正确的格式
        values = {
            "tensor": output_tensor,
            "parameters": None,
            "graph": None
        }
        return {"values": values}

    def forward_(self, input1: Tensor_, input2: Tensor_) -> dict:
        """
        加法操作的Python实现 - 简化版本

        Args:
            input1: 第一个输入张量占位符
            input2: 第二个输入张量占位符

        Returns:
            dict: 包含输出张量占位符的字典
        """
        # 检查形状是否相同
        if input1.size != input2.size:
            raise ValueError(f"ADD操作输入形状不匹配: {input1.size} vs {input2.size}")

        output_tensor = Tensor_(*input1.size, dtype=self.dtype)

        values = {
            "tensor": output_tensor,
            "parameters": None,
            "graph": None
        }
        return {"values": values}

class MUL(Ops):
    """乘法操作类"""

    def __init__(self, inputs, outputs, dtype, version="17"):
        super(MUL, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        """
        乘法操作的C后端实现 - 简化版本，假设形状相同

        Args:
            input1: 第一个输入张量
            input2: 第二个输入张量

        Returns:
            Tensor: 乘法结果张量
        """
        if input1.size != input2.size:
            raise ValueError(f"MUL操作输入形状不匹配: {input1.size} vs {input2.size}")

        input1_c = self._numpy_to_ctensor(input1.data, self.dtype)
        input2_c = self._numpy_to_ctensor(input2.data, self.dtype)

        output_shape = input1.size
        output_shape_c = (ctypes.c_int * len(output_shape))(*output_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(output_shape), nn.DTYPE_MAP[self.dtype])

        self.lib.mul_forward(input1_c, input2_c, output_c)

        output_data = self._ctensor_to_numpy(output_c, self.dtype)
        output_tensor = Tensor(*output_shape, dtype=self.dtype, data=output_data)

        self.lib.free_tensor(input1_c)
        self.lib.free_tensor(input2_c)
        self.lib.free_tensor(output_c)

        values = {
            "tensor": output_tensor,
            "parameters": None,
            "graph": None
        }
        return {"values": values}

    def forward_(self, input1: Tensor_, input2: Tensor_) -> Tensor_:
        """
        乘法操作的Python实现

        Args:
            input1: 第一个输入张量占位符
            input2: 第二个输入张量占位符

        Returns:
            Tensor_: 输出张量占位符
        """
        if input1.size != input2.size:
            raise ValueError(f"MUL操作输入形状不匹配: {input1.size} vs {input2.size}")

        output_tensor = Tensor_(*input1.size, dtype=self.dtype)

        values = {
            "tensor": output_tensor,
            "parameters": None,
            "graph": None
        }
        return {"values": values}


class SUB(Ops):
    """减法操作类"""

    def __init__(self, inputs, outputs, dtype, version="17"):
        super(SUB, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        """
        减法操作的C后端实现 - 简化版本，假设形状相同

        Args:
            input1: 第一个输入张量
            input2: 第二个输入张量

        Returns:
            Tensor: 减法结果张量
        """
        if input1.size != input2.size:
            raise ValueError(f"SUB操作输入形状不匹配: {input1.size} vs {input2.size}")

        input1_c = self._numpy_to_ctensor(input1.data, self.dtype)
        input2_c = self._numpy_to_ctensor(input2.data, self.dtype)

        output_shape = input1.size
        output_shape_c = (ctypes.c_int * len(output_shape))(*output_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(output_shape), nn.DTYPE_MAP[self.dtype])

        self.lib.sub_forward(input1_c, input2_c, output_c)

        output_data = self._ctensor_to_numpy(output_c, self.dtype)
        output_tensor = Tensor(*output_shape, dtype=self.dtype, data=output_data)

        self.lib.free_tensor(input1_c)
        self.lib.free_tensor(input2_c)
        self.lib.free_tensor(output_c)

        values = {
            "tensor": output_tensor,
            "parameters": None,
            "graph": None
        }
        return {"values": values}

    def forward_(self, input1: Tensor_, input2: Tensor_) -> Tensor_:
        """
        减法操作的Python实现

        Args:
            input1: 第一个输入张量占位符
            input2: 第二个输入张量占位符

        Returns:
            Tensor_: 输出张量占位符
        """
        if input1.size != input2.size:
            raise ValueError(f"SUB操作输入形状不匹配: {input1.size} vs {input2.size}")

        output_tensor = Tensor_(*input1.size, dtype=self.dtype)

        values = {
            "tensor": output_tensor,
            "parameters": None,
            "graph": None
        }
        return {"values": values}


class DIV(Ops):
    """除法操作类"""

    def __init__(self, inputs, outputs, dtype, version="17"):
        super(DIV, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        """
        除法操作的C后端实现 - 简化版本，假设形状相同

        Args:
            input1: 第一个输入张量
            input2: 第二个输入张量

        Returns:
            Tensor: 除法结果张量
        """
        if input1.size != input2.size:
            raise ValueError(f"DIV操作输入形状不匹配: {input1.size} vs {input2.size}")

        input1_c = self._numpy_to_ctensor(input1.data, self.dtype)
        input2_c = self._numpy_to_ctensor(input2.data, self.dtype)

        output_shape = input1.size
        output_shape_c = (ctypes.c_int * len(output_shape))(*output_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(output_shape), nn.DTYPE_MAP[self.dtype])

        self.lib.div_forward(input1_c, input2_c, output_c)

        output_data = self._ctensor_to_numpy(output_c, self.dtype)
        output_tensor = Tensor(*output_shape, dtype=self.dtype, data=output_data)

        self.lib.free_tensor(input1_c)
        self.lib.free_tensor(input2_c)
        self.lib.free_tensor(output_c)

        values = {
            "tensor": output_tensor,
            "parameters": None,
            "graph": None
        }
        return {"values": values}

    def forward_(self, input1: Tensor_, input2: Tensor_) -> Tensor_:
        """
        除法操作的Python实现

        Args:
            input1: 第一个输入张量占位符
            input2: 第二个输入张量占位符

        Returns:
            Tensor_: 输出张量占位符
        """
        if input1.size != input2.size:
            raise ValueError(f"DIV操作输入形状不匹配: {input1.size} vs {input2.size}")

        output_tensor = Tensor_(*input1.size, dtype=self.dtype)

        values = {
            "tensor": output_tensor,
            "parameters": None,
            "graph": None
        }
        return {"values": values}


class RESHAPE(Ops):
    """形状重塑操作类"""

    def __init__(self, inputs, outputs, dtype, new_shape=None, version="17"):
        super(RESHAPE, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.new_shape = new_shape
        self.version = version

    def forward(self, input: Tensor) -> Tensor:
        """
        Reshape操作的C后端实现

        Args:
            input: 输入张量

        Returns:
            Tensor: 重塑形状后的输出张量
        """
        input_c = self._numpy_to_ctensor(input.data, self.dtype)

        # 使用指定的新形状，如果没有指定则保持原状
        output_shape = self.new_shape if self.new_shape is not None else input.size
        output_shape_c = (ctypes.c_int * len(output_shape))(*output_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(output_shape), nn.DTYPE_MAP[self.dtype])

        self.lib.reshape_forward(input_c, output_c)

        output_data = self._ctensor_to_numpy(output_c, self.dtype)
        output_tensor = Tensor(*output_shape, dtype=self.dtype, data=output_data)

        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)

        values = {
            "tensor": output_tensor,
            "parameters": None,
            "graph": None
        }
        return {"values": values}

    def forward_(self, input: Tensor_) -> Tensor_:
        """
        Reshape操作的Python实现

        Args:
            input: 输入张量占位符

        Returns:
            Tensor_: 输出张量占位符
        """
        output_shape = self.new_shape if self.new_shape is not None else input.size
        output_tensor = Tensor_(*output_shape, dtype=self.dtype)

        values = {
            "tensor": output_tensor,
            "parameters": None,
            "graph": None
        }
        return {"values": values}