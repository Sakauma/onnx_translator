import sys
from collections import OrderedDict
import ctypes
import numpy as np
from typing import List, Union
import os


class CTensor(ctypes.Structure):
    """C张量结构体，用于与C库交互"""
    _fields_ = [
        ("data", ctypes.c_void_p),           # 数据指针
        ("shape", ctypes.POINTER(ctypes.c_int)),  # 形状数组指针
        ("ndim", ctypes.c_int),              # 维度数
        ("size", ctypes.c_size_t),           # 总元素数
        ("dtype", ctypes.c_int)              # 数据类型
    ]


# 数据类型映射到整数编码
DTYPE_MAP = {
    "float16": 0,
    "bfloat16": 1,
    "float32": 2,
    "float64": 3,
    "int8": 4,
    "int16": 5,
    "int32": 6,
    "int64": 7
}


# 数据类型映射到NumPy类型
DTYPE_TO_NUMPY = {
    "float16": np.float16,
    "bfloat16": np.uint16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64
}

# NumPy 类型到 NPS 字符串类型的反向映射
NUMPY_TO_DTYPE = {
    np.float16: "float16",
    np.uint16: "bfloat16",
    np.float32: "float32",
    np.float64: "float64",
    np.int8: "int8",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
}

# 动态添加对平台特定类型的支持
NUMPY_TO_DTYPE[np.dtype('intc').type] = "int32" if np.dtype('intc').itemsize == 4 else "int64"
if hasattr(np, 'uint32'):
    NUMPY_TO_DTYPE[np.uint32] = "uint32" 
if hasattr(np, 'uint64'):
    NUMPY_TO_DTYPE[np.uint64] = "uint64"


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
    16: "bfloat16"
}


class Tensor:
    """张量类，用于存储和操作多维数组数据"""
    
    def __init__(self, *size, dtype="float32", data=None):
        """
        初始化张量
        
        Args:
            *size: 张量的维度大小
            dtype: 数据类型
            data: 初始化数据，如果为None则初始化为零矩阵
        """
        self.size = size[0] if (isinstance(size[0], list) and len(size) == 1) else size
        self.data_size = 1
        for s in self.size:
            self.data_size *= s
        self.dtype = dtype
        
        if data is not None:
            self.data = data
        else:
            np_dtype = DTYPE_TO_NUMPY[dtype]
            self.data = np.zeros(self.size, dtype=np_dtype)


class Tensor_:
    """张量占位符类，用于图构建阶段"""
    
    def __init__(self, *size, dtype="float32"):
        """
        初始化张量占位符
        
        Args:
            *size: 张量的维度大小
            dtype: 数据类型
        """
        self.size = size[0] if (isinstance(size[0], list) and len(size) == 1) else size
        self.data_size = 1
        for s in self.size:
            self.data_size *= s
        self.dtype = dtype


class Ops:
    """操作基类，所有计算操作的父类"""
    _lib = None
    _lib_initialized = False

    @classmethod
    def _get_lib(cls):
        """
        获取C库实例，确保只初始化一次
        
        Returns:
            ctypes.CDLL: C库实例
        """
        if cls._lib is None:
            # 加载C库
            cls._lib = ctypes.CDLL('./tensor_ops.so')
            
            # 设置函数返回类型
            cls._lib.create_tensor.restype = ctypes.POINTER(CTensor)
            
            # 设置函数参数类型
            cls._lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
            cls._lib.free_tensor.argtypes = [ctypes.POINTER(CTensor)]
            cls._lib.relu_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            cls._lib.cos_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            cls._lib.abs_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            cls._lib.add_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            
            # 初始化余弦查找表
            cls._lib.init_cos_lut.argtypes = []
            cls._lib.init_cos_lut()
            cls._lib_initialized = True
            
        return cls._lib

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

    def forward(self, input):
        """
        前向传播方法（使用真实数据计算）
        
        Args:
            input: 输入数据
            
        Returns:
            计算结果
        """
        pass

    def forward_(self, input):
        """
        前向传播方法（不使用真实数据计算，用于图构建）
        
        Args:
            input: 输入数据占位符
            
        Returns:
            计算结果占位符
        """
        pass

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
        self.output_name = output_name if isinstance(output_name, list) else [output_name]
        self.ops = OrderedDict()
        self.update(ops)
        self.model_name = model_name

    def update(self, ops):
        """
        更新计算图中的操作节点
        
        Args:
            ops: 操作节点列表
        """
        name_dict = {}
        self.output_in_degree = {na: 0 for na in self.input_name}
        
        for op in ops:
            # 生成操作名称
            name = str(op.__class__).split("'")[1].split(".")[-1]
            if name not in name_dict:
                name_dict[name] = 0
            else:
                name_dict[name] += 1
                
            # 设置操作名称
            if not op.name:
                op.name = name + ".%d" % name_dict[name]
                self.ops[op.name] = op
                
            # 更新输入输出节点的入度
            for i in op.inputs:
                if i in self.output_in_degree:
                    self.output_in_degree[i] += 1
                    
            for o in op.outputs:
                if o not in self.output_in_degree:
                    self.output_in_degree[o] = 0
                else:
                    print("output edge name %s repeat!!!" % o)
                    sys.exit()
                    
            # 如果没有指定输出节点，则自动推断
            if not self.output_name[0]:
                for na in self.output_in_degree:
                    if self.output_in_degree[na] == 0:
                        self.output_name.append(na)
                        self.output_in_degree[na] = 1
                        self.output_name = self.output_name[1:]

    def forward(self, *inputs):
        """
        执行前向传播计算（使用真实数据）
        
        Args:
            *inputs: 输入数据
            
        Returns:
            计算结果
        """
        # 初始化边数据缓冲区
        edge_data_buffer = {}
        outputs = ()
        
        # 设置输入数据
        for idx, na in enumerate(self.input_name):
            edge_data_buffer[na] = inputs[idx]
            
        length = len(self.ops)
        
        # 依次执行每个操作
        for (cc, op_na) in zip(range(length), self.ops):
            op = self.ops[op_na]
            inputs = (edge_data_buffer[na] for na in op.inputs)
            outputs = op.forward(*inputs)
            
            # 处理输出结果
            if "graph" in outputs:
                outputs, graph = outputs["tensor"], outputs["graph"]
                do_graph = True
            elif "parameters" in outputs:
                outputs, parameters = outputs["tensor"], outputs["parameters"]
                do_graph = False
                
            # 更新入度
            for idx, inp_na in enumerate(op.inputs):
                self.output_in_degree[inp_na] -= 1
                
            # 保存输出结果
            for idx, out_na in enumerate(op.outputs):
                if len(op.outputs) == 1:
                    edge_data_buffer[out_na] = outputs
                    continue
                edge_data_buffer[out_na] = outputs[idx]
                
            # 清理无用的边数据
            for na in list(edge_data_buffer.keys()):
                if self.output_in_degree[na] == 0:
                    edge_data_buffer.pop(na)

    def forward_(self, *inputs):
        """
        执行前向传播计算（不使用真实数据，用于图构建）
        
        Args:
            *inputs: 输入数据占位符
            
        Returns:
            计算结果占位符
        """
        # 初始化边数据缓冲区
        edge_data_buffer = {}
        outputs = ()
        
        # 设置输入数据
        for idx, na in enumerate(self.input_name):
            edge_data_buffer[na] = inputs[idx]
            
        length = len(self.ops)
        
        # 依次执行每个操作
        for (cc, op_na) in zip(range(length), self.ops):
            op = self.ops[op_na]
            inputs = (edge_data_buffer[na] for na in op.inputs)
            outputs = op.forward_(*inputs)
            
            # 处理输出结果
            if "graph" in outputs:
                outputs, graph = outputs["tensor"], outputs["graph"]
                do_graph = True
            elif "parameters" in outputs:
                outputs, parameters = outputs["tensor"], outputs["parameters"]
                do_graph = False
                
            # 更新入度
            for idx, inp_na in enumerate(op.inputs):
                self.output_in_degree[inp_na] -= 1
                
            # 保存输出结果
            for idx, out_na in enumerate(op.outputs):
                if len(op.outputs) == 1:
                    edge_data_buffer[out_na] = outputs
                    continue
                edge_data_buffer[out_na] = outputs[idx]
                
            # 清理无用的边数据
            for na in list(edge_data_buffer.keys()):
                if self.output_in_degree[na] == 0:
                    edge_data_buffer.pop(na)

