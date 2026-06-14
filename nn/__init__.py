# /**
#   ******************************************************************************
#   * @file        __init__.py
#   * @author      Egor Izmaylov
#   * @brief       定义 Tensor、Tensor_、Ops 和 Graph 等核心运行时抽象，以及 Python 与 C 后端共享的 dtype 映射。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from collections import OrderedDict
import ctypes
import numpy as np
from typing import List, Union
import os
import nn

TENSOR_OPS_LIB_PATH = os.environ.get(
    "TENSOR_OPS_LIB",
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "tensor_ops.so")),
)
# TENSOR_OPS_LIB 允许测试指向刚构建好的共享库，而不需要修改导入代码。
# 默认路径会从仓库根目录解析 tensor_ops.so，方便本地开发和验证。

class CTensor(ctypes.Structure):
    """C张量结构体，用于与C库交互"""
    _fields_ = [
        ("data", ctypes.c_void_p),                # 数据指针
        ("shape", ctypes.POINTER(ctypes.c_int)),  # 形状数组指针
        ("ndim", ctypes.c_int),                   # 维度数
        ("size", ctypes.c_size_t),                # 总元素数
        ("dtype", ctypes.c_int)                   # 数据类型
    ]

# 这些整数编码必须与 tensor_ops/tensor_ops.h 中的 DataType 枚举保持一致。
# 如果映射不一致，ctypes 调用会让 C 后端按错误元素类型解释同一段内存。
# 数据类型映射到整数编码
DTYPE_MAP = {
    "float8_e4m3": 0,
    "float8_e5m2": 1,
    "float16": 2,
    "bfloat16": 3,
    "float32": 4,
    "float64": 5,
    "int4": 6,
    "int8": 7,
    "uint8": 8,# unit为无符号数
    "int16": 9,
    "int32": 10,
    "int64": 11,
    "uint16": 12,
    "uint32": 13,
    "uint64": 14,
    "bool": 15,
    "complex64": 16,
    "complex128": 17,
}

# 数据类型映射到NumPy类型
DTYPE_TO_NUMPY = {
    "float8_e4m3": np.uint8, 
    "float8_e5m2": np.uint8,
    "float16": np.float16,
    "bfloat16": np.uint16,
    "float32": np.float32,
    "float64": np.float64,
    "int4": np.int8,
    "int8": np.int8,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
    "string": np.str_,
    "complex64": np.complex64,
    "complex128": np.complex128,
}

# NumPy 类型到 NPS 字符串类型的反向映射
NUMPY_TO_DTYPE = {
    np.float16: "float16",
    np.uint16: "bfloat16",
    np.float32: "float32",
    np.float64: "float64",
    np.int8: "int8",
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.uint32: "uint32",
    np.uint64: "uint64",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
    np.bool_: "bool",
    np.str_: "string",
    np.complex64: "complex64",
    np.complex128: "complex128",
    # int4 需要显式指定 dtype="int4"
}

# 动态添加对平台特定类型的支持
NUMPY_TO_DTYPE[np.dtype('intc').type] = "int32" if np.dtype('intc').itemsize == 4 else "int64"
if hasattr(np, 'uint32'):
    NUMPY_TO_DTYPE[np.uint32] = "uint32" 
if hasattr(np, 'uint64'):
    NUMPY_TO_DTYPE[np.uint64] = "uint64"

# ONNXImport 使用该表将 TensorProto dtype id 转换为本地 dtype 字符串。
# 如果启用新的 ONNX dtype，需要同时补齐 NumPy 表示和 C 枚举映射后再接入 tensor_ops。
# ONNX数据类型映射
onnx_dtype_mapping = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "float64", # 对应 ONNX 'double'
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
    17: "float8_e4m3",
    18: "float8_e4m3",
    19: "float8_e5m2",
    20: "float8_e5m2",
    22: "int4",
}

class Tensor:
    """张量类，用于存储和操作多维数组数据"""
    
    # 初始化 `Tensor` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, *size, dtype="float32", data=None):
        """
        初始化张量
        
        Args:
            *size: 张量的维度大小
            dtype: 数据类型
            data: 初始化数据，如果为None则初始化为零矩阵
        """
        # self.size = size[0] if (isinstance(size[0], list) and len(size) == 1) else size
        # self.data_size = 1
        # for s in self.size:
        #     self.data_size *= s
        # self.dtype = dtype
        
        # if data is not None:
        #     self.data = data
        # else:
        #     np_dtype = DTYPE_TO_NUMPY[dtype]
        #     self.data = np.zeros(self.size, dtype=np_dtype)
        if len(size) == 1 and isinstance(size[0], list):
            self.size = size[0]
        else:
            self.size = size
            
        self.data_size = 1
        for s in self.size:
            self.data_size *= s# 计算总元素个数
        self.dtype = dtype
        
        if data is not None:
            self.data = data
        else:
            # 安全获取 numpy 类型
            np_dtype = DTYPE_TO_NUMPY.get(dtype, np.float32)
            self.data = np.zeros(self.size, dtype=np_dtype)# 创建一个全0的矩阵

class Tensor_:
    """张量占位符类，用于图构建阶段"""
    
    # 初始化 `Tensor_` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, *size, dtype="float32"):
        """
        初始化张量占位符
        
        Args:
            *size: 张量的维度大小
            dtype: 数据类型
        """
        # self.size = size[0] if (isinstance(size[0], list) and len(size) == 1) else size
        # self.data_size = 1
        # for s in self.size:
        #     self.data_size *= s
        # self.dtype = dtype
        if len(size) == 1 and isinstance(size[0], list):
            self.size = size[0]
        else:
            self.size = size
            
        self.data_size = 1
        for s in self.size:
            self.data_size *= s
        self.dtype = dtype

class Ops:
    """操作基类，所有计算操作的父类"""
    _lib = None
    _lib_initialized = False

    # 封装 `_get_lib` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @classmethod
    def _get_lib(cls):
        """
        获取C库实例，确保只初始化一次
        
        Returns:
            ctypes.CDLL: C库实例
        """
        if cls._lib is None:
            # 加载C库
            if not os.path.exists(TENSOR_OPS_LIB_PATH):
                raise FileNotFoundError(
                    f"C backend library not found: {TENSOR_OPS_LIB_PATH}. Run `make` first."
                )
            cls._lib = ctypes.CDLL(TENSOR_OPS_LIB_PATH)
            
            # 设置函数返回类型
            cls._lib.create_tensor.restype = ctypes.POINTER(CTensor)
            
            # 设置函数参数类型
            cls._lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
            cls._lib.free_tensor.argtypes = [ctypes.POINTER(CTensor)]
            cls._lib.relu_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            cls._lib.cos_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            cls._lib.abs_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            cls._lib.add_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            cls._lib.sub_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            cls._lib.mul_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            cls._lib.div_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            cls._lib.quantize_linear_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            try:
                cls._lib.quantize_linear_forward_precision.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]
            except AttributeError:
                pass
            try:
                cls._lib.quantize_linear_forward_precision_saturate.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_int]
            except AttributeError:
                pass
            cls._lib.dequantize_linear_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            
            # 初始化余弦查找表
            cls._lib.init_cos_lut.argtypes = []
            cls._lib.init_cos_lut()
            cls._lib_initialized = True
            
        return cls._lib

    # 初始化 `Ops` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs):
        """
        初始化操作
        
        Args:
            inputs: 输入节点列表
            outputs: 输出节点列表
        """
        self.inputs = inputs
        self.outputs = outputs
        self.parameters = {}
        self.name = None
        self.lib = self._get_lib()
    
    # 封装 `_execute_unary` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _execute_unary(self, input_tensor, c_func_name):
        """通用一元算子执行模板"""
        # 1. 准备连续内存输入
        in_data = np.ascontiguousarray(input_tensor.data)# 确保数据连续
        input_c = self._numpy_to_ctensor(in_data, input_tensor.dtype)
        
        # 2. 准备输出 (优先使用 self.dtype，否则沿用输入类型)
        out_dtype = self.dtype if self.dtype else input_tensor.dtype
        out_shape = (ctypes.c_int * len(input_tensor.size))(*input_tensor.size)
        output_c = self.lib.create_tensor(out_shape, len(input_tensor.size), DTYPE_MAP[out_dtype])
        
        # 3. 动态调用 C 函数
        getattr(self.lib, c_func_name)(input_c, output_c)
        
        # 4. 转换结果并释放
        out_data = self._ctensor_to_numpy(output_c, out_dtype)
        self.lib.free_tensor(input_c)# 调用C函数释放内存
        self.lib.free_tensor(output_c)
        
        return Tensor(*input_tensor.size, dtype=out_dtype, data=out_data)
    
    # 封装 `_execute_binary` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _execute_binary(self, input_a, input_b, c_func_name):
        """通用二元算子执行模板 (含广播逻辑)"""
        # 1. 广播处理
        try:
            a_bcast, b_bcast = np.broadcast_arrays(input_a.data, input_b.data)
        except ValueError as e:
            print(f"Broadcasting error: {input_a.size} vs {input_b.size}")
            raise e
            
        out_shape = a_bcast.shape
        
        # 2. 类型推断 (优先 self.dtype -> 其次 numpy 推断 -> 默认 float32)
        if self.dtype:
            out_dtype = self.dtype
        else:
            res_type = np.result_type(a_bcast, b_bcast)
            out_dtype = NUMPY_TO_DTYPE.get(res_type.type, "float32")

        # 3. 转换为 C 张量
        #a_c = self._numpy_to_ctensor(np.ascontiguousarray(a_bcast.astype(input_a.dtype, copy=False)), input_a.dtype)
        #b_c = self._numpy_to_ctensor(np.ascontiguousarray(b_bcast.astype(input_b.dtype, copy=False)), input_b.dtype)
        np_dtype_a = DTYPE_TO_NUMPY[input_a.dtype]
        a_data_safe = a_bcast.astype(np_dtype_a, copy=False)
        a_c = self._numpy_to_ctensor(np.ascontiguousarray(a_data_safe), input_a.dtype)
        np_dtype_b = DTYPE_TO_NUMPY[input_b.dtype]
        b_data_safe = b_bcast.astype(np_dtype_b, copy=False)
        b_c = self._numpy_to_ctensor(np.ascontiguousarray(b_data_safe), input_b.dtype)
        
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), DTYPE_MAP[out_dtype])

        getattr(self.lib, c_func_name)(a_c, b_c, output_c)

        out_data = self._ctensor_to_numpy(output_c, out_dtype)
        self.lib.free_tensor(a_c)
        self.lib.free_tensor(b_c)
        self.lib.free_tensor(output_c)# 释放内存

        return Tensor(*out_shape, dtype=out_dtype, data=out_data)
    
    # 封装 `_execute_ternary` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _execute_ternary(self, in_a, in_b, in_c, c_func_name, extra_int_arg=None):
        """通用三元算子执行模板 (含广播逻辑，用于 QDQ)"""
        try:
            a_bc, b_bc, c_bc = np.broadcast_arrays(in_a.data, in_b.data, in_c.data)
        except ValueError as e:
            print(f"Broadcasting error in ternary op: {in_a.size}, {in_b.size}, {in_c.size}")
            raise e
            
        out_shape = a_bc.shape
        out_dtype = self.dtype if self.dtype else "float32"
        # 实现 `prep_ctensor` 步骤，规范化输入并返回下游期望的数据或元信息。
        def prep_ctensor(arr_bcast, original_tensor):
            np_dtype = nn.DTYPE_TO_NUMPY[original_tensor.dtype]
            arr_safe = arr_bcast.astype(np_dtype, copy=False)
            return self._numpy_to_ctensor(np.ascontiguousarray(arr_safe), original_tensor.dtype)

        a_c = prep_ctensor(a_bc, in_a)
        b_c = prep_ctensor(b_bc, in_b)
        c_c = prep_ctensor(c_bc, in_c)
        
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[out_dtype])
        if extra_int_arg is None:
            getattr(self.lib, c_func_name)(a_c, b_c, c_c, output_c)
        else:
            extra_args = extra_int_arg if isinstance(extra_int_arg, (tuple, list)) else (extra_int_arg,)
            getattr(self.lib, c_func_name)(a_c, b_c, c_c, output_c, *[ctypes.c_int(int(arg)) for arg in extra_args])

        out_data = self._ctensor_to_numpy(output_c, out_dtype)
        self.lib.free_tensor(a_c)
        self.lib.free_tensor(b_c)
        self.lib.free_tensor(c_c)
        self.lib.free_tensor(output_c)

        return Tensor(*out_shape, dtype=out_dtype, data=out_data)

    # 执行 `Ops` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input):
        """
        前向传播方法（使用真实数据计算）
        
        Args:
            input: 输入数据
            
        Returns:
            计算结果
        """
        pass

    # 执行 `Ops` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input):
        """
        前向传播方法（不使用真实数据计算，用于图构建）
        
        Args:
            input: 输入数据占位符
            
        Returns:
            计算结果占位符
        """
        pass

    # 封装 `_numpy_to_ctensor` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _numpy_to_ctensor(self, arr: np.ndarray, dtype: str) -> ctypes.POINTER(CTensor):
        """
        将NumPy数组转换为C张量
        
        Args:
            arr: NumPy数组
            dtype: 数据类型
            
        Returns:
            ctypes.POINTER(CTensor): C张量指针
        """
        # 创建形状数组
        shape = (ctypes.c_int * len(arr.shape))(*arr.shape)
        # 创建C张量
        c_tensor = self.lib.create_tensor(shape, len(arr.shape), DTYPE_MAP[dtype])
        # 复制数据
        data_size = arr.size * arr.itemsize
        ctypes.memmove(c_tensor.contents.data, arr.ctypes.data, data_size)
        return c_tensor

    # 封装 `_ctensor_to_numpy` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _ctensor_to_numpy(self, c_tensor: ctypes.POINTER(CTensor), dtype: str) -> np.ndarray:
        """
        将C张量转换为NumPy数组
        
        Args:
            c_tensor: C张量指针
            dtype: 数据类型
            
        Returns:
            np.ndarray: NumPy数组
        """
        # 获取形状
        shape = [c_tensor.contents.shape[i] for i in range(c_tensor.contents.ndim)]

        # 从C数据创建NumPy数组
        np_dtype = DTYPE_TO_NUMPY[dtype]
        arr = np.frombuffer(
            (ctypes.c_byte * (c_tensor.contents.size * np.dtype(np_dtype).itemsize)).from_address(c_tensor.contents.data),
            dtype=np_dtype
        ).reshape(shape)

        return arr.copy()

class Graph:
    """计算图类，用于管理操作节点和数据流"""
    
    # 初始化 `Graph` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, ops, input_name, output_name=None, model_name=None):
        """
        初始化计算图
        
        Args:
            ops: 操作节点列表
            input_name: 输入节点名称
            output_name: 输出节点名称
            model_name: 模型名称
        """
        self.input_name = input_name if isinstance(input_name, list) else [input_name]
        self.input_name = [name for name in self.input_name if name]
        if output_name is None:
            self.output_name = []
        elif isinstance(output_name, list):
            self.output_name = [name for name in output_name if name]
        else:
            self.output_name = [output_name] if output_name else []
        self.ops = OrderedDict()
        self.update(ops)
        self.model_name = model_name

    # 实现 `update` 步骤，规范化输入并返回下游期望的数据或元信息。
    def update(self, ops):
        """
        更新计算图中的操作节点

        Args:
            ops: 操作节点列表
        """
        name_dict = {}
        self.ops = OrderedDict()
        self.output_in_degree = {na: 0 for na in self.input_name}
        produced_edges = []

        for op in ops:
            # 生成操作名称
            name = str(op.__class__).split("'")[1].split(".")[-1]
            if name not in name_dict:
                name_dict[name] = 0
            else:
                name_dict[name] += 1

            # 设置操作名称
            op_name = op.name or name + ".%d" % name_dict[name]
            if op_name in self.ops:
                suffix = 1
                base_name = op_name
                while f"{base_name}.{suffix}" in self.ops:
                    suffix += 1
                op_name = f"{base_name}.{suffix}"
            op.name = op_name
            self.ops[op.name] = op

            # 更新输入输出节点的入度
            for i in self._op_consumed_edges(op):
                if i and i in self.output_in_degree:
                    self.output_in_degree[i] += 1

            for o in op.outputs:
                if not o:
                    continue
                if o not in self.output_in_degree:
                    self.output_in_degree[o] = 0
                    produced_edges.append(o)
                else:
                    raise ValueError(f"output edge name {o} repeat!!!")

        # 如果没有指定输出节点，则自动推断为无消费者的算子输出。
        if not self.output_name:
            self.output_name = [
                edge for edge in produced_edges
                if self.output_in_degree.get(edge, 0) == 0
            ]

    # 封装 `_init_edge_data` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _init_edge_data(self, inputs):
        if len(inputs) != len(self.input_name):
            raise ValueError(
                f"Graph expects {len(self.input_name)} inputs, got {len(inputs)}"
            )
        return {na: inputs[idx] for idx, na in enumerate(self.input_name)}

    # 封装 `_extract_tensor_result` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _extract_tensor_result(outputs):
        if isinstance(outputs, dict):
            if "tensor" in outputs:
                return outputs["tensor"]
            raise KeyError("operator result dict does not contain 'tensor'")
        return outputs

    # 封装 `_normalize_multi_output` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _normalize_multi_output(outputs, idx):
        if isinstance(outputs, (list, tuple)):
            if idx < len(outputs):
                return outputs[idx]
            raise IndexError(f"operator returned {len(outputs)} outputs, index {idx} requested")
        if idx == 0:
            return outputs
        raise TypeError(f"operator should return a list/tuple for multiple outputs, got {type(outputs)}")

    # 封装 `_op_consumed_edges` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _op_consumed_edges(op):
        names = []
        for name in list(op.inputs) + list(getattr(op, "outer_scope_names", [])):
            if name and name not in names:
                names.append(name)
        return names

    # 封装 `_collect_graph_outputs` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _collect_graph_outputs(self, edge_data_buffer):
        missing = [name for name in self.output_name if name not in edge_data_buffer]
        if missing:
            raise KeyError(f"Graph output(s) not produced: {missing}")
        outputs = [edge_data_buffer[name] for name in self.output_name]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    # 执行 `Graph` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, *inputs):
        """
        执行前向传播计算（使用真实数据）

        Args:
            *inputs: 输入数据

        Returns:
            计算结果
        """
        # 初始化边数据缓冲区。每次执行都复制使用计数，避免污染图对象。
        edge_data_buffer = self._init_edge_data(inputs)
        edge_usage = dict(self.output_in_degree)
        protected_outputs = set(self.output_name)
        outputs = ()

        length = len(self.ops)

        # 依次执行每个操作
        for (cc, op_na) in zip(range(length), self.ops):
            op = self.ops[op_na]
            op_inputs = tuple(None if not na else edge_data_buffer[na] for na in op.inputs)
            if hasattr(op, "forward_with_context"):
                outputs = self._extract_tensor_result(op.forward_with_context(edge_data_buffer, *op_inputs))
            else:
                outputs = self._extract_tensor_result(op.forward(*op_inputs))

            # 更新入度
            for inp_na in self._op_consumed_edges(op):
                if inp_na and inp_na in edge_usage:
                    edge_usage[inp_na] -= 1

            # 保存输出结果
            for idx, out_na in enumerate(op.outputs):
                if not out_na:
                    continue
                edge_data_buffer[out_na] = (
                    outputs if len(op.outputs) == 1
                    else self._normalize_multi_output(outputs, idx)
                )

            # 清理无用的边数据
            for na in list(edge_data_buffer.keys()):
                if edge_usage.get(na, 0) == 0 and na not in protected_outputs:
                    edge_data_buffer.pop(na)

        return self._collect_graph_outputs(edge_data_buffer)

    # 执行 `Graph` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, *inputs):
        """
        执行前向传播计算（不使用真实数据，用于图构建）
        """
        # 初始化边数据缓冲区。每次执行都复制使用计数，避免污染图对象。
        edge_data_buffer = self._init_edge_data(inputs)
        edge_usage = dict(self.output_in_degree)
        protected_outputs = set(self.output_name)
        outputs = ()

        length = len(self.ops)

        # 依次执行每个操作
        for (cc, op_na) in zip(range(length), self.ops):
            op = self.ops[op_na]

            op_inputs_list = []
            try:
                for na in op.inputs:
                    if na == "":
                        op_inputs_list.append(None)
                    else:
                        if na not in edge_data_buffer:
                            raise KeyError(f"找不到输入边: '{na}'")
                        op_inputs_list.append(edge_data_buffer[na])
                
                op_inputs = tuple(op_inputs_list)
                
            except KeyError as e:
                #print(f"\n [图构建错误] 算子 {op_na} ({op.__class__.__name__}) 输入缺失: {e}")
                #print(f"   该算子需要的输入: {op.inputs}")
                # print(f"   当前缓冲区可用边: {list(edge_data_buffer.keys())}") # 调试时可开启
                raise e

            # --- 执行算子推断 ---
            try:
                outputs = op.forward_(*op_inputs)
            except Exception as e:
                #print(f"\n [算子推断崩溃] 在执行 {op_na} ({op.__class__.__name__}) 时发生错误！")
                #print(f"   错误信息: {e}")
                
                input_info = []
                for x in op_inputs:
                    if x is None: input_info.append("None")
                    elif hasattr(x, 'size'): input_info.append(f"Tensor(shape={x.size})")
                    else: input_info.append(str(x))
                print(f"   输入情况: {input_info}")
                raise e
            
            # 处理输出结果
            if isinstance(outputs, dict):
                if "graph" in outputs:
                    outputs, graph = outputs["tensor"], outputs["graph"]
                    do_graph = True
                elif "parameters" in outputs:
                    outputs, parameters = outputs["tensor"], outputs["parameters"]
                    do_graph = False
            
            # 更新入度
            for inp_na in self._op_consumed_edges(op):
                if inp_na and inp_na in edge_usage: # 忽略空字符串和常量
                    edge_usage[inp_na] -= 1
                
            # 分配输出
            try:
                for idx, out_na in enumerate(op.outputs):
                    # 如果输出名为空（有些算子有可选输出），跳过
                    if not out_na: continue
                        
                    if len(op.outputs) == 1:
                        edge_data_buffer[out_na] = outputs
                        continue
                    
                    if not isinstance(outputs, (list, tuple)):
                         # 容错：如果算子应该多输出但只回了一个对象，且索引为0，则尝试直接赋值
                         if idx == 0:
                             edge_data_buffer[out_na] = outputs
                             continue
                         else:
                             raise TypeError(f"算子应返回列表，实际返回: {type(outputs)}")
                    
                    if idx < len(outputs):
                        edge_data_buffer[out_na] = outputs[idx]
                    else:
                        # 输出数量不足，可能是算子实现问题，也可能是该输出确实没生成
                        pass 
            except Exception as e:
                print(f"❌ [输出分配错误] {op_na}: {e}")
                raise e
                
            # 清理无用的边数据
            for na in list(edge_data_buffer.keys()):
                if edge_usage.get(na, 0) == 0 and na not in protected_outputs:
                    edge_data_buffer.pop(na)

        return self._collect_graph_outputs(edge_data_buffer)
