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

