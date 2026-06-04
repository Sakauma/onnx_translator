# /**
#   ******************************************************************************
#   * @file        elementwise_basic.py
#   * @author      Egor Izmaylov
#   * @brief       保存 `elementwise_basic` 分组中的 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *

class RELU(Ops):
    """ReLU激活函数操作类"""
    
    # 初始化 `RELU` 的构造参数，保存后续运行、形状推断或验证所需的状态。
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

    # 执行 `RELU` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
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

    # 执行 `RELU` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
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
    
    # 初始化 `COS` 的构造参数，保存后续运行、形状推断或验证所需的状态。
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

    # 执行 `COS` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
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

    # 执行 `COS` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
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
    
    # 初始化 `ABS` 的构造参数，保存后续运行、形状推断或验证所需的状态。
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

    # 执行 `ABS` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> Tensor:
        """
        Abs函数的C后端实现，使用真实数据进行计算
        """
        out_tensor = self._execute_unary(input, "abs_forward")
        values = {"tensor": out_tensor, "parameters": None, "graph": None}
        self.parameters = {"values": values}
        return values

    # 执行 `ABS` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
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

    # 初始化 `ADD` 的构造参数，保存后续运行、形状推断或验证所需的状态。
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

    # 执行 `ADD` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
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

    # 执行 `ADD` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
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

    # 初始化 `SUB` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(SUB, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `SUB` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
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
        
    # 执行 `SUB` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
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

    # 初始化 `MUL` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(MUL, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `MUL` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
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

    # 执行 `MUL` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
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

    # 初始化 `DIV` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(DIV, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `DIV` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
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

    # 执行 `DIV` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
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


class EXP(Ops):
    # 初始化 `EXP` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(EXP, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `EXP` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "exp_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # 执行 `EXP` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}


class LOG(Ops):
    # 初始化 `LOG` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(LOG, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `LOG` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "log_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # 执行 `LOG` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}


class SQRT(Ops):
    # 初始化 `SQRT` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(SQRT, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `SQRT` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "sqrt_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # 执行 `SQRT` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}


class SIGMOID(Ops):
    # 初始化 `SIGMOID` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(SIGMOID, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `SIGMOID` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "sigmoid_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # 执行 `SIGMOID` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}


class TANH(Ops):
    # 初始化 `TANH` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(TANH, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `TANH` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "tanh_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # 执行 `TANH` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}


class Pow(Ops):
    # 初始化 `Pow` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Pow, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `Pow` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input_a: Tensor, input_b: Tensor) -> dict:
        integer_dtypes = {"int4", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}
        signed_integer_dtypes = {"int4", "int8", "int16", "int32", "int64"}
        if input_a.dtype in integer_dtypes and input_b.dtype in signed_integer_dtypes:
            if np.any(np.asarray(input_b.data) < 0):
                raise ValueError("Integers to negative integer powers are not allowed.")
        out_tensor = self._execute_binary(input_a, input_b, "pow_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # 执行 `Pow` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input_a: Tensor_, input_b: Tensor_) -> dict:
        # 简单广播推断
        try:
            bcast = np.broadcast_shapes(input_a.size, input_b.size)
        except:
            bcast = input_a.size
        output_tensor = Tensor_(*bcast, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}


class Max(Ops):
    # 初始化 `Max` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Max, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `Max` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, *inputs: Tensor) -> dict:
        if not inputs:
            raise ValueError("Max requires at least one input")
        if self.lib is None:
            arrays = np.broadcast_arrays(*(_tensor_data_as_numeric(x) for x in inputs))
            out_data = arrays[0]
            for arr in arrays[1:]:
                out_data = np.maximum(out_data, arr)
            out_data = _cast_numeric_to_dtype(out_data, self.dtype)
            return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}
        out_tensor = inputs[0]
        for next_tensor in inputs[1:]:
            out_tensor = self._execute_binary(out_tensor, next_tensor, "max_forward")
        if self.dtype and out_tensor.dtype != self.dtype:
            out_data = out_tensor.data.astype(nn.DTYPE_TO_NUMPY[self.dtype], copy=False)
            out_tensor = Tensor(*out_data.shape, dtype=self.dtype, data=out_data)
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # 执行 `Max` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, *inputs: Tensor_) -> dict:
        if not inputs:
            raise ValueError("Max requires at least one input")
        bcast = np.broadcast_shapes(*(x.size for x in inputs))
        output_tensor = Tensor_(*bcast, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}


class Min(Ops):
    # 初始化 `Min` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Min, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `Min` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, *inputs: Tensor) -> dict:
        if not inputs:
            raise ValueError("Min requires at least one input")
        if self.lib is None:
            arrays = np.broadcast_arrays(*(_tensor_data_as_numeric(x) for x in inputs))
            out_data = arrays[0]
            for arr in arrays[1:]:
                out_data = np.minimum(out_data, arr)
            out_data = _cast_numeric_to_dtype(out_data, self.dtype)
            return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}
        out_tensor = inputs[0]
        for next_tensor in inputs[1:]:
            out_tensor = self._execute_binary(out_tensor, next_tensor, "min_forward")
        if self.dtype and out_tensor.dtype != self.dtype:
            out_data = out_tensor.data.astype(nn.DTYPE_TO_NUMPY[self.dtype], copy=False)
            out_tensor = Tensor(*out_data.shape, dtype=self.dtype, data=out_data)
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # 执行 `Min` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, *inputs: Tensor_) -> dict:
        if not inputs:
            raise ValueError("Min requires at least one input")
        bcast = np.broadcast_shapes(*(x.size for x in inputs))
        output_tensor = Tensor_(*bcast, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}


class Neg(Ops):
    # 初始化 `Neg` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Neg, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Neg` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "neg_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}
    # 执行 `Neg` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}


class Reciprocal(Ops):
    # 初始化 `Reciprocal` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Reciprocal, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Reciprocal` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "reciprocal_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}
    # 执行 `Reciprocal` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}


class Ceil(Ops):
    # 初始化 `Ceil` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Ceil, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Ceil` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "ceil_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}
    # 执行 `Ceil` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}


class Floor(Ops):
    # 初始化 `Floor` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Floor, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Floor` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor) -> dict:
        out_tensor = self._execute_unary(input, "floor_forward")
        return {"tensor": out_tensor, "parameters": None, "graph": None}
    # 执行 `Floor` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_) -> dict:
        output_tensor = Tensor_(*input.size, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}
