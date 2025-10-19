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
        """
        初始化ADD操作

        Args:
            inputs: 输入节点列表（应该包含两个输入张量）
            outputs: 输出节点列表
            dtype: 数据类型
            version: 操作版本号
        """
        super(ADD, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, *inputs) -> Tensor:
        """
        ADD函数的C后端实现，使用真实数据进行计算

        Args:
            *inputs: 输入张量列表（应该包含两个张量）

        Returns:
            dict: 包含输出张量的字典
        """
        # 检查输入数量
        if len(inputs) != 2:
            raise ValueError(f"ADD操作需要2个输入，但得到了{len(inputs)}个")

        input_a, input_b = inputs

        # 将输入转换为C张量
        input_c_a = self._numpy_to_ctensor(input_a.data, self.dtype)
        input_c_b = self._numpy_to_ctensor(input_b.data, self.dtype)

        # 创建输出C张量
        output_shape = (ctypes.c_int * len(input_a.size))(*input_a.size)
        output_c = self.lib.create_tensor(output_shape, len(input_a.size), nn.DTYPE_MAP[self.dtype])

        # 调用C函数进行加法运算
        self.lib.add_forward(input_c_a, input_c_b, output_c)

        # 转换回numpy并创建输出张量
        output_data = self._ctensor_to_numpy(output_c, self.dtype)
        output_tensor = Tensor(*input_a.size, dtype=self.dtype, data=output_data)

        # 清理资源
        self.lib.free_tensor(input_c_a)
        self.lib.free_tensor(input_c_b)
        self.lib.free_tensor(output_c)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, *inputs) -> Tensor_:
        """
        ADD函数的Python实现，不使用真实数据进行计算

        Args:
            *inputs: 输入张量占位符列表（应该包含两个张量）

        Returns:
            dict: 包含输出张量占位符的字典
        """
        # 检查输入数量
        if len(inputs) != 2:
            raise ValueError(f"ADD操作需要2个输入，但得到了{len(inputs)}个")

        input_a, input_b = inputs

        # 对于占位符计算，直接返回一个具有相同形状的占位符
        output_tensor = Tensor_(*input_a.size, dtype=self.dtype)
        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

class MUL(Ops):
    """乘法操作类"""

    def __init__(self, inputs, outputs, dtype, version="17"):
        """
        初始化MUL操作

        Args:
            inputs: 输入节点列表（应该包含两个输入张量）
            outputs: 输出节点列表
            dtype: 数据类型
            version: 操作版本号
        """
        super(MUL, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, *inputs) -> Tensor:
        """
        MUL函数的C后端实现，使用真实数据进行计算

        Args:
            *inputs: 输入张量列表（应该包含两个张量）

        Returns:
            dict: 包含输出张量的字典
        """
        # 检查输入数量
        if len(inputs) != 2:
            raise ValueError(f"MUL操作需要2个输入，但得到了{len(inputs)}个")

        input_a, input_b = inputs

        # 将输入转换为C张量
        input_c_a = self._numpy_to_ctensor(input_a.data, self.dtype)
        input_c_b = self._numpy_to_ctensor(input_b.data, self.dtype)

        # 创建输出C张量
        output_shape = (ctypes.c_int * len(input_a.size))(*input_a.size)
        output_c = self.lib.create_tensor(output_shape, len(input_a.size), nn.DTYPE_MAP[self.dtype])

        # 调用C函数进行乘法运算
        self.lib.mul_forward(input_c_a, input_c_b, output_c)

        # 转换回numpy并创建输出张量
        output_data = self._ctensor_to_numpy(output_c, self.dtype)
        output_tensor = Tensor(*input_a.size, dtype=self.dtype, data=output_data)

        # 清理资源
        self.lib.free_tensor(input_c_a)
        self.lib.free_tensor(input_c_b)
        self.lib.free_tensor(output_c)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, *inputs) -> Tensor_:
        """
        MUL函数的Python实现，不使用真实数据进行计算

        Args:
            *inputs: 输入张量占位符列表（应该包含两个张量）

        Returns:
            dict: 包含输出张量占位符的字典
        """
        # 检查输入数量
        if len(inputs) != 2:
            raise ValueError(f"MUL操作需要2个输入，但得到了{len(inputs)}个")

        input_a, input_b = inputs

        # 对于占位符计算，直接返回一个具有相同形状的占位符
        output_tensor = Tensor_(*input_a.size, dtype=self.dtype)
        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

class SUB(Ops):
    """减法操作类"""

    def __init__(self, inputs, outputs, dtype, version="17"):
        """
        初始化SUB操作

        Args:
            inputs: 输入节点列表（应该包含两个输入张量）
            outputs: 输出节点列表
            dtype: 数据类型
            version: 操作版本号
        """
        super(SUB, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, *inputs) -> Tensor:
        """
        SUB函数的C后端实现，使用真实数据进行计算

        Args:
            *inputs: 输入张量列表（应该包含两个张量）

        Returns:
            dict: 包含输出张量的字典
        """
        # 检查输入数量
        if len(inputs) != 2:
            raise ValueError(f"SUB操作需要2个输入，但得到了{len(inputs)}个")

        input_a, input_b = inputs

        # 将输入转换为C张量
        input_c_a = self._numpy_to_ctensor(input_a.data, self.dtype)
        input_c_b = self._numpy_to_ctensor(input_b.data, self.dtype)

        # 创建输出C张量
        output_shape = (ctypes.c_int * len(input_a.size))(*input_a.size)
        output_c = self.lib.create_tensor(output_shape, len(input_a.size), nn.DTYPE_MAP[self.dtype])

        # 调用C函数进行减法运算（注意：这里应该是sub_forward而不是mul_forward）
        self.lib.sub_forward(input_c_a, input_c_b, output_c)

        # 转换回numpy并创建输出张量
        output_data = self._ctensor_to_numpy(output_c, self.dtype)
        output_tensor = Tensor(*input_a.size, dtype=self.dtype, data=output_data)

        # 清理资源
        self.lib.free_tensor(input_c_a)
        self.lib.free_tensor(input_c_b)
        self.lib.free_tensor(output_c)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, *inputs) -> Tensor_:
        """
        SUB函数的Python实现，不使用真实数据进行计算

        Args:
            *inputs: 输入张量占位符列表（应该包含两个张量）

        Returns:
            dict: 包含输出张量占位符的字典
        """
        # 检查输入数量
        if len(inputs) != 2:
            raise ValueError(f"SUB操作需要2个输入，但得到了{len(inputs)}个")

        input_a, input_b = inputs

        # 对于占位符计算，直接返回一个具有相同形状的占位符
        output_tensor = Tensor_(*input_a.size, dtype=self.dtype)
        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

class DIV(Ops):
    """除法操作类"""

    def __init__(self, inputs, outputs, dtype, version="17"):
        """
        初始化DIV操作

        Args:
            inputs: 输入节点列表（应该包含两个输入张量）
            outputs: 输出节点列表
            dtype: 数据类型
            version: 操作版本号
        """
        super(DIV, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, *inputs) -> Tensor:
        """
        DIV函数的C后端实现，使用真实数据进行计算

        Args:
            *inputs: 输入张量列表（应该包含两个张量）

        Returns:
            dict: 包含输出张量的字典
        """
        # 检查输入数量
        if len(inputs) != 2:
            raise ValueError(f"DIV操作需要2个输入，但得到了{len(inputs)}个")

        input_a, input_b = inputs

        # 检查除数是否包含零（避免除零错误）
        if np.any(input_b.data == 0):
            raise ValueError("DIV操作中除数不能为零")

        # 将输入转换为C张量
        input_c_a = self._numpy_to_ctensor(input_a.data, self.dtype)
        input_c_b = self._numpy_to_ctensor(input_b.data, self.dtype)

        # 创建输出C张量
        output_shape = (ctypes.c_int * len(input_a.size))(*input_a.size)
        output_c = self.lib.create_tensor(output_shape, len(input_a.size), nn.DTYPE_MAP[self.dtype])

        # 调用C函数进行除法运算
        self.lib.div_forward(input_c_a, input_c_b, output_c)

        # 转换回numpy并创建输出张量
        output_data = self._ctensor_to_numpy(output_c, self.dtype)
        output_tensor = Tensor(*input_a.size, dtype=self.dtype, data=output_data)

        # 清理资源
        self.lib.free_tensor(input_c_a)
        self.lib.free_tensor(input_c_b)
        self.lib.free_tensor(output_c)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, *inputs) -> Tensor_:
        """
        DIV函数的Python实现，不使用真实数据进行计算

        Args:
            *inputs: 输入张量占位符列表（应该包含两个张量）

        Returns:
            dict: 包含输出张量占位符的字典
        """
        # 检查输入数量
        if len(inputs) != 2:
            raise ValueError(f"DIV操作需要2个输入，但得到了{len(inputs)}个")

        input_a, input_b = inputs

        # 对于占位符计算，直接返回一个具有相同形状的占位符
        output_tensor = Tensor_(*input_a.size, dtype=self.dtype)
        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values


class RESHAPE(Ops):
    """重塑操作类"""

    def __init__(self, inputs, outputs, dtype, version="17"):
        """
        初始化RESHAPE操作

        Args:
            inputs: 输入节点列表（应该包含输入张量和目标形状）
            outputs: 输出节点列表
            dtype: 数据类型
            version: 操作版本号
        """
        super(RESHAPE, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, *inputs) -> Tensor:
        """
        RESHAPE函数的C后端实现，使用真实数据进行计算

        Args:
            *inputs: 输入张量列表（应该包含输入张量和形状张量）

        Returns:
            dict: 包含输出张量的字典
        """
        # 检查输入数量
        if len(inputs) != 2:
            raise ValueError(f"RESHAPE操作需要2个输入，但得到了{len(inputs)}个")

        input_tensor, shape_tensor = inputs

        # 将输入转换为C张量
        input_c = self._numpy_to_ctensor(input_tensor.data, self.dtype)

        # 形状张量应该是int64类型的一维张量
        if shape_tensor.dtype != 'int64':
            raise ValueError(f"RESHAPE的形状张量必须是int64类型，但得到了{shape_tensor.dtype}")

        shape_data = shape_tensor.data.astype(np.int64)
        shape_c = self._numpy_to_ctensor(shape_data, 'int64')

        # 创建输出C张量
        # 注意：输出形状需要在C函数中计算，这里先创建一个临时形状
        temp_shape = (ctypes.c_int * 1)(1)
        output_c = self.lib.create_tensor(temp_shape, 1, nn.DTYPE_MAP[self.dtype])

        # 调用C函数进行重塑运算
        self.lib.reshape_forward(input_c, shape_c, output_c)

        # 从C张量获取输出形状和数据
        output_ndim = output_c.ndim
        output_shape = tuple(output_c.shape[i] for i in range(output_ndim))
        output_data = self._ctensor_to_numpy(output_c, self.dtype)

        # 创建输出张量
        output_tensor = Tensor(*output_shape, dtype=self.dtype, data=output_data)

        # 清理资源
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(shape_c)
        self.lib.free_tensor(output_c)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, *inputs) -> Tensor_:
        """
        RESHAPE函数的Python实现，不使用真实数据进行计算

        Args:
            *inputs: 输入张量占位符列表（应该包含输入张量和形状张量）

        Returns:
            dict: 包含输出张量占位符的字典
        """
        # 检查输入数量
        if len(inputs) != 2:
            raise ValueError(f"RESHAPE操作需要2个输入，但得到了{len(inputs)}个")

        input_tensor, shape_tensor = inputs

        # 对于占位符计算，需要根据输入形状和目标形状计算输出形状
        if hasattr(shape_tensor, 'data') and shape_tensor.data is not None:
            # 如果有真实的形状数据
            new_shape = tuple(shape_tensor.data)
        else:
            # 如果没有真实数据，使用占位符的形状（假设形状张量是常量）
            # 这里需要根据实际情况处理，可能需要从其他途径获取形状信息
            raise ValueError("RESHAPE操作需要已知的目标形状")

        # 处理-1通配符
        input_total_elements = np.prod(input_tensor.size)
        new_shape_list = list(new_shape)

        # 计算非-1维度的乘积
        known_product = 1
        minus_one_count = 0
        minus_one_index = -1

        for i, dim in enumerate(new_shape_list):
            if dim == -1:
                minus_one_count += 1
                minus_one_index = i
            elif dim > 0:
                known_product *= dim
            else:
                raise ValueError(f"Invalid dimension {dim} in reshape shape")

        # 检查-1的个数
        if minus_one_count > 1:
            raise ValueError("At most one -1 allowed in reshape shape")

        # 计算-1对应的维度
        if minus_one_count == 1:
            if known_product == 0 or input_total_elements % known_product != 0:
                raise ValueError(f"Cannot infer -1 dimension: input has {input_total_elements} elements, "
                                 f"but other dimensions product is {known_product}")
            inferred_dim = input_total_elements // known_product
            new_shape_list[minus_one_index] = inferred_dim

        # 验证总元素数匹配
        output_total_elements = np.prod(new_shape_list)
        if output_total_elements != input_total_elements:
            raise ValueError(f"Total elements mismatch: input {input_total_elements} vs output {output_total_elements}")

        # 创建输出张量占位符
        output_tensor = Tensor_(*new_shape_list, dtype=self.dtype)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

    def _validate_shape(self, shape_data, input_total_elements):
        """
        验证形状数据的有效性并处理-1通配符

        Args:
            shape_data: 目标形状数据
            input_total_elements: 输入张量的总元素数

        Returns:
            tuple: 处理后的有效形状
        """
        shape_list = list(shape_data)
        minus_one_count = 0
        known_product = 1
        minus_one_index = -1

        for i, dim in enumerate(shape_list):
            if dim == -1:
                minus_one_count += 1
                minus_one_index = i
            elif dim > 0:
                known_product *= dim
            else:
                raise ValueError(f"Invalid dimension {dim} in reshape shape")

        if minus_one_count > 1:
            raise ValueError("At most one -1 allowed in reshape shape")

        if minus_one_count == 1:
            if known_product == 0 or input_total_elements % known_product != 0:
                raise ValueError(f"Cannot infer -1 dimension: input has {input_total_elements} elements, "
                                 f"but other dimensions product is {known_product}")
            inferred_dim = input_total_elements // known_product
            shape_list[minus_one_index] = inferred_dim

        output_total_elements = np.prod(shape_list)
        if output_total_elements != input_total_elements:
            raise ValueError(f"Total elements mismatch: input {input_total_elements} vs output {output_total_elements}")

        return tuple(shape_list)