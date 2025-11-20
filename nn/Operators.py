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
        output_c = self.lib.create_tensor(
            output_shape, len(input.size), nn.DTYPE_MAP[self.dtype])

        # 调用C函数
        self.lib.relu_forward(input_c, output_c)

        # 转换回numpy并创建输出张量
        output_data = self._ctensor_to_numpy(output_c, input.dtype)
        output_tensor = Tensor(
            *input.size,
            dtype=input.dtype,
            data=output_data)

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
        input_c = self._numpy_to_ctensor(input.data, input.dtype)

        # 创建输出C张量
        output_shape = (ctypes.c_int * len(input.size))(*input.size)
        output_c = self.lib.create_tensor(
            output_shape, len(input.size), nn.DTYPE_MAP[self.dtype])

        # 调用C函数
        self.lib.cos_forward(input_c, output_c)

        # 转换回numpy并创建输出张量
        output_data = self._ctensor_to_numpy(output_c, input.dtype)
        output_tensor = Tensor(
            *input.size,
            dtype=input.dtype,
            data=output_data)

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
        # 将输入转换为C张量
        input_c = self._numpy_to_ctensor(input.data, input.dtype)

        # 创建输出C张量
        output_shape = (ctypes.c_int * len(input.size))(*input.size)
        output_c = self.lib.create_tensor(
            output_shape, len(input.size), nn.DTYPE_MAP[self.dtype])

        # 调用C函数
        self.lib.abs_forward(input_c, output_c)

        # 转换回numpy并创建输出张量
        output_data = self._ctensor_to_numpy(output_c, input.dtype)
        output_tensor = Tensor(
            *input.size,
            dtype=input.dtype,
            data=output_data)

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
        Abs函数的Python实现，不使用真实数据进行计算
        """
        output_tensor = input
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
        self.dtype = dtype  # 这是ONNX图推断的输出类型
        self.version = version

    def forward(self, *inputs) -> Tensor:
        """
        加法函数的C后端实现，在Python层处理广播
        """
        if len(inputs) != 2:
            raise ValueError(
                f"ADD operator expects 2 inputs, but got {len(inputs)}")

        a = inputs[0]
        b = inputs[1]

        # 1. Python (NumPy) 层处理广播
        # np.broadcast_arrays 会返回两个具有相同（广播后）形状的新数组
        try:
            # astype(np.float64) 确保广播时使用高精度，防止精度损失
            a_bcast_data, b_bcast_data = np.broadcast_arrays(a.data, b.data)
        except ValueError as e:
            print(
                f"Error during broadcasting inputs with shapes {a.size} and {b.size}")
            raise e

        # 2. Python (NumPy) 层处理类型提升
        # 确定最佳的输出数据类型
        output_dtype_np = np.result_type(a.data, b.data)

        # 查找NPS dtype字符串，如果找不到（例如uint32），则默认为float32或float64
        if output_dtype_np.type in nn.NUMPY_TO_DTYPE:
            output_dtype_str = nn.NUMPY_TO_DTYPE[output_dtype_np.type]
        elif 'float' in str(output_dtype_np):
            output_dtype_str = "float64"
        elif 'int' in str(output_dtype_np):
            output_dtype_str = "int64"
        else:
            output_dtype_str = "float32"  # 最终备用

        output_shape = a_bcast_data.shape

        # 3. 准备C张量 (A, B, Output)
        # 确保广播后的输入数据类型与原始张量类型一致，再传入C
        # （注意：广播后的数组a_bcast_data可能继承了类型提升后的dtype，我们用astype(a.data.dtype)把它转回去）
        a_data_contiguous = np.ascontiguousarray(
            a_bcast_data.astype(a.data.dtype, copy=False))
        b_data_contiguous = np.ascontiguousarray(
            b_bcast_data.astype(b.data.dtype, copy=False))
        a_c = self._numpy_to_ctensor(a_data_contiguous, a.dtype)
        b_c = self._numpy_to_ctensor(b_data_contiguous, b.dtype)

        # 4. 创建输出C张量
        output_shape_c = (ctypes.c_int * len(output_shape))(*output_shape)
        output_c = self.lib.create_tensor(
            output_shape_c,
            len(output_shape),
            nn.DTYPE_MAP[output_dtype_str])

        # 5. 调用C函数 (此时A, B, O的形状一致)
        self.lib.add_forward(a_c, b_c, output_c)

        # 6. 转换回NPS/NumPy张量
        output_data = self._ctensor_to_numpy(output_c, output_dtype_str)
        output_tensor = Tensor(
            *output_shape,
            dtype=output_dtype_str,
            data=output_data)

        # 7. 清理资源
        self.lib.free_tensor(a_c)
        self.lib.free_tensor(b_c)
        self.lib.free_tensor(output_c)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, *inputs) -> Tensor_:
        """
        Add函数的Python实现，不使用真实数据进行计算 (用于图推断)
        """
        if len(inputs) != 2:
            raise ValueError(
                f"ADD operator expects 2 inputs (Tensor_), but got {len(inputs)}")

        a = inputs[0]
        b = inputs[1]

        # 1. 计算广播后的形状 (不计算数据)
        temp_a = np.empty(a.size, dtype=np.uint8)  # 使用uint8节省内存
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
            output_dtype_str = "float32"  # 最终备用

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
            raise ValueError(
                f"SUB operator expects 2 inputs, but got {len(inputs)}")

        a = inputs[0]
        b = inputs[1]

        # 1. Python (NumPy) 层处理广播
        try:
            a_bcast_data, b_bcast_data = np.broadcast_arrays(a.data, b.data)
        except ValueError as e:
            print(
                f"Error during broadcasting inputs with shapes {a.size} and {b.size}")
            raise e

        # 2. Python (NumPy) 层处理类型提升
        output_dtype_np = np.result_type(a.data, b.data)

        if output_dtype_np.type in nn.NUMPY_TO_DTYPE:
            output_dtype_str = nn.NUMPY_TO_DTYPE[output_dtype_np.type]
        elif 'float' in str(output_dtype_np):
            output_dtype_str = "float64"
        elif 'int' in str(output_dtype_np):
            output_dtype_str = "int64"
        else:
            output_dtype_str = "float32"

        output_shape = a_bcast_data.shape

        # 3. 准备C张量
        a_data_contiguous = np.ascontiguousarray(
            a_bcast_data.astype(a.data.dtype, copy=False))
        b_data_contiguous = np.ascontiguousarray(
            b_bcast_data.astype(b.data.dtype, copy=False))
        a_c = self._numpy_to_ctensor(a_data_contiguous, a.dtype)
        b_c = self._numpy_to_ctensor(b_data_contiguous, b.dtype)

        # 4. 创建输出C张量
        output_shape_c = (ctypes.c_int * len(output_shape))(*output_shape)
        output_c = self.lib.create_tensor(
            output_shape_c,
            len(output_shape),
            nn.DTYPE_MAP[output_dtype_str])

        # 5. 调用C函数
        self.lib.sub_forward(a_c, b_c, output_c)

        # 6. 转换回NPS/NumPy张量
        output_data = self._ctensor_to_numpy(output_c, output_dtype_str)
        output_tensor = Tensor(
            *output_shape,
            dtype=output_dtype_str,
            data=output_data)

        # 7. 清理资源
        self.lib.free_tensor(a_c)
        self.lib.free_tensor(b_c)
        self.lib.free_tensor(output_c)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, *inputs) -> Tensor_:
        """
        Sub函数的Python实现，不使用真实数据进行计算 (用于图推断)
        """
        if len(inputs) != 2:
            raise ValueError(
                f"SUB operator expects 2 inputs (Tensor_), but got {len(inputs)}")

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
            raise ValueError(
                f"MUL operator expects 2 inputs, but got {len(inputs)}")

        a = inputs[0]
        b = inputs[1]

        # 1. Python (NumPy) 层处理广播
        try:
            a_bcast_data, b_bcast_data = np.broadcast_arrays(a.data, b.data)
        except ValueError as e:
            print(
                f"Error during broadcasting inputs with shapes {a.size} and {b.size}")
            raise e

        # 2. Python (NumPy) 层处理类型提升
        output_dtype_np = np.result_type(a.data, b.data)

        if output_dtype_np.type in nn.NUMPY_TO_DTYPE:
            output_dtype_str = nn.NUMPY_TO_DTYPE[output_dtype_np.type]
        elif 'float' in str(output_dtype_np):
            output_dtype_str = "float64"
        elif 'int' in str(output_dtype_np):
            output_dtype_str = "int64"
        else:
            output_dtype_str = "float32"

        output_shape = a_bcast_data.shape

        # 3. 准备C张量
        a_data_contiguous = np.ascontiguousarray(
            a_bcast_data.astype(a.data.dtype, copy=False))
        b_data_contiguous = np.ascontiguousarray(
            b_bcast_data.astype(b.data.dtype, copy=False))
        a_c = self._numpy_to_ctensor(a_data_contiguous, a.dtype)
        b_c = self._numpy_to_ctensor(b_data_contiguous, b.dtype)

        # 4. 创建输出C张量
        output_shape_c = (ctypes.c_int * len(output_shape))(*output_shape)
        output_c = self.lib.create_tensor(
            output_shape_c,
            len(output_shape),
            nn.DTYPE_MAP[output_dtype_str])

        # 5. 调用C函数
        self.lib.mul_forward(a_c, b_c, output_c)

        # 6. 转换回NPS/NumPy张量
        output_data = self._ctensor_to_numpy(output_c, output_dtype_str)
        output_tensor = Tensor(
            *output_shape,
            dtype=output_dtype_str,
            data=output_data)

        # 7. 清理资源
        self.lib.free_tensor(a_c)
        self.lib.free_tensor(b_c)
        self.lib.free_tensor(output_c)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, *inputs) -> Tensor_:
        """
        Mul函数的Python实现，不使用真实数据进行计算 (用于图推断)
        """
        if len(inputs) != 2:
            raise ValueError(
                f"MUL operator expects 2 inputs (Tensor_), but got {len(inputs)}")

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
            raise ValueError(
                f"DIV operator expects 2 inputs, but got {len(inputs)}")

        a = inputs[0]
        b = inputs[1]

        # 1. Python (NumPy) 层处理广播
        try:
            a_bcast_data, b_bcast_data = np.broadcast_arrays(a.data, b.data)
        except ValueError as e:
            print(
                f"Error during broadcasting inputs with shapes {a.size} and {b.size}")
            raise e

        # 2. Python (NumPy) 层处理类型提升
        output_dtype_np = np.result_type(a.data, b.data)

        if output_dtype_np.type in nn.NUMPY_TO_DTYPE:
            output_dtype_str = nn.NUMPY_TO_DTYPE[output_dtype_np.type]
        elif 'float' in str(output_dtype_np):
            output_dtype_str = "float64"
        elif 'int' in str(output_dtype_np):
            output_dtype_str = "int64"
        else:
            output_dtype_str = "float32"

        output_shape = a_bcast_data.shape

        # 3. 准备C张量
        a_data_contiguous = np.ascontiguousarray(
            a_bcast_data.astype(a.data.dtype, copy=False))
        b_data_contiguous = np.ascontiguousarray(
            b_bcast_data.astype(b.data.dtype, copy=False))
        a_c = self._numpy_to_ctensor(a_data_contiguous, a.dtype)
        b_c = self._numpy_to_ctensor(b_data_contiguous, b.dtype)

        # 4. 创建输出C张量
        output_shape_c = (ctypes.c_int * len(output_shape))(*output_shape)
        output_c = self.lib.create_tensor(
            output_shape_c,
            len(output_shape),
            nn.DTYPE_MAP[output_dtype_str])

        # 5. 调用C函数
        self.lib.div_forward(a_c, b_c, output_c)

        # 6. 转换回NPS/NumPy张量
        output_data = self._ctensor_to_numpy(output_c, output_dtype_str)
        output_tensor = Tensor(
            *output_shape,
            dtype=output_dtype_str,
            data=output_data)

        # 7. 清理资源
        self.lib.free_tensor(a_c)
        self.lib.free_tensor(b_c)
        self.lib.free_tensor(output_c)

        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, *inputs) -> Tensor_:
        """
        Div函数的Python实现，不使用真实数据进行计算 (用于图推断)
        """
        if len(inputs) != 2:
            raise ValueError(
                f"DIV operator expects 2 inputs (Tensor_), but got {len(inputs)}")

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

class TANH(Ops):
    """Tanh激活函数操作类"""

    def __init__(self, inputs, outputs, dtype, version="17"):
        super(TANH, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input: Tensor) -> Tensor:
        # 将输入转换为C张量
        input_c = self._numpy_to_ctensor(input.data, self.dtype)
        
        # 创建输出C张量
        output_shape = (ctypes.c_int * len(input.size))(*input.size)
        output_c = self.lib.create_tensor(output_shape, len(input.size), nn.DTYPE_MAP[self.dtype])
        
        # 调用C函数
        self.lib.tanh_forward(input_c, output_c)
        
        # 转换回numpy并创建输出张量
        output_data = self._ctensor_to_numpy(output_c, input.dtype)
        output_tensor = Tensor(*input.size, dtype=input.dtype, data=output_data)
        
        values = {
            'tensor': output_tensor,
            "parameters": None,
            "graph": None
        }
        self.parameters = {'values': values}
        return values

class RESHAPE(Ops):
    """Reshape操作类"""

    def __init__(self, inputs, outputs, dtype, version="17"):
        super(RESHAPE, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
class RESHAPE(Ops):
    """Reshape操作类"""

    def __init__(self, inputs, outputs, dtype, version="17"):
        super(RESHAPE, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, *args):
        """
        Reshape函数的C后端实现

        Args:
            args[0]: 输入张量
            args[1]: 新形状张量

        Returns:
            Tensor: 重塑后的输出张量
        """
        if len(args) != 2:
            raise ValueError(f"Reshape expects 2 arguments, got {len(args)}")
        
        input = args[0]
        shape = args[1]
        
        # 调试信息: 查看输入数据
        print("=== RESHAPE Debug ===")
        print("input.data shape:", input.data.shape)
        print("shape.data:", shape.data)
        print("shape.data type:", type(shape.data))
        print("shape.data shape:", shape.data.shape)
        
        # 修复：正确处理形状数据
        if hasattr(shape.data, 'shape') and shape.data.size > 1:
            # shape.data 是多元素数组，展平后转换
            new_shape = [int(x) for x in shape.data.flatten()]
        elif hasattr(shape.data, 'shape') and shape.data.size == 1:
            # shape.data 是单元素数组
            new_shape = [int(shape.data)]
        else:
            # shape.data 是标量
            new_shape = [int(shape.data)]
        
        print("new_shape:", new_shape)

        # 验证新形状的总元素数是否与输入匹配
        input_size = np.prod(input.data.shape)
        new_size = np.prod(new_shape)
        print(f"input_size: {input_size}, new_size: {new_size}")

        if input_size != new_size:
            print(f"Warning: Reshape size mismatch: {input_size} != {new_size}")
            # 自动调整：如果有-1，计算合适的维度
            if -1 in new_shape:
                count_neg_one = new_shape.count(-1)
                if count_neg_one > 1:
                    raise ValueError("can only specify one unknown dimension")
                
                # 计算自动调整的维度
                known_dims_prod = abs(np.prod([x for x in new_shape if x != -1]))
                auto_dim = input_size // known_dims_prod
                new_shape = [auto_dim if x == -1 else x for x in new_shape]
                print(f"Auto-adjusted new_shape: {new_shape}")
            else:
                raise ValueError(f"Reshape size mismatch: {input_size} != {new_size}")
        
        new_ndim = len(new_shape)
        
        # 将输入转换为C张量
        input_c = self._numpy_to_ctensor(input.data, self.dtype)
        
        # 创建输出C张量
        output_shape_array = (ctypes.c_int * new_ndim)(*new_shape)
        output_c = self.lib.create_tensor(output_shape_array, new_ndim, nn.DTYPE_MAP[self.dtype])
        
        # 调用C函数
        self.lib.reshape_forward(input_c, output_c, output_shape_array, new_ndim)
        
        # 转换回numpy并创建输出张量
        output_data = self._ctensor_to_numpy(output_c, input.dtype)
        output_tensor = Tensor(*new_shape, dtype=input.dtype, data=output_data)
        
        values = {
            'tensor': output_tensor,
            "parameters": None,
            "graph": None
        }
        self.parameters = {'values': values}
        return values

class UNSQUEEZE(Ops):
    """Unsqueeze操作类"""

    def __init__(self, inputs, outputs, dtype, version="17"):
        super(UNSQUEEZE, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input: Tensor, axes: Tensor) -> Tensor:
        # 从轴张量中提取轴
        axis = int(axes.data[0])
        
        # 计算输出形状
        output_shape = list(input.size)
        output_shape.insert(axis, 1)
        output_ndim = len(output_shape)
        
        # 将输入转换为C张量
        input_c = self._numpy_to_ctensor(input.data, self.dtype)
        
        # 创建输出C张量
        output_c_shape = (ctypes.c_int * output_ndim)(*output_shape)
        output_c = self.lib.create_tensor(output_c_shape, output_ndim, nn.DTYPE_MAP[self.dtype])
        
        # 调用C函数
        self.lib.unsqueeze_forward(input_c, output_c, axis)
        
        # 转换回numpy并创建输出张量
        output_data = self._ctensor_to_numpy(output_c, input.dtype)
        output_tensor = Tensor(*output_shape, dtype=input.dtype, data=output_data)
        
        values = {
            'tensor': output_tensor,
            "parameters": None,
            "graph": None
        }
        self.parameters = {'values': values}
        return values
