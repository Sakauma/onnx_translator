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
from pathlib import Path
import nn

def _default_tensor_ops_lib_path():
    """优先解析已安装包内的后端，源码树运行时再回退到仓库构建产物。"""
    package_lib = Path(__file__).resolve().with_name("tensor_ops.so")
    repo_lib = Path(__file__).resolve().parent.parent / "tensor_ops.so"
    for candidate in (package_lib, repo_lib):
        if candidate.exists():
            return str(candidate)
    return str(repo_lib)


TENSOR_OPS_LIB_PATH = os.environ.get("TENSOR_OPS_LIB", _default_tensor_ops_lib_path())
# TENSOR_OPS_LIB 允许测试指向刚构建好的共享库，而不需要修改导入代码。
# 默认路径优先使用 wheel 内的 nn/tensor_ops.so，开发环境回退到仓库根目录 tensor_ops.so。

class CTensor(ctypes.Structure):
    """``tensor_ops.h::Tensor`` 的 ctypes ABI 镜像。

    该结构及其 ``shape``、``data`` 缓冲区均由 C 端 ``create_tensor`` 分配，
    Python 只在指针有效期内读写内容，并负责最终调用 ``free_tensor``。
    """
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
    "float8_e4m3fnuz": 18,
    "float8_e5m2fnuz": 19,
    "uint4": 20,
    "int2": 21,
    "uint2": 22,
    "float4_e2m1": 23,
    "float8_e8m0": 24,
}

# 低位宽浮点和整数在 NumPy 中没有等价 dtype，因此数组保存的是未打包的原始编码位；
# 进行数值计算前必须由算子辅助函数显式解码，不能把这些 uint8/int8 当作真实数值。
DTYPE_TO_NUMPY = {
    "float8_e4m3": np.uint8, 
    "float8_e5m2": np.uint8,
    "float8_e4m3fnuz": np.uint8,
    "float8_e5m2fnuz": np.uint8,
    "float16": np.float16,
    "bfloat16": np.uint16,
    "float32": np.float32,
    "float64": np.float64,
    "int4": np.int8,
    "uint4": np.uint8,
    "int2": np.int8,
    "uint2": np.uint8,
    "float4_e2m1": np.uint8,
    "float8_e8m0": np.uint8,
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

# 反向映射只适用于存储类型唯一的常规 dtype。bfloat16 与 uint16 共用 NumPy
# 存储类型，低位宽类型也共用 uint8/int8，因此这些逻辑类型必须由调用方显式指定。
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
    18: "float8_e4m3fnuz",
    19: "float8_e5m2",
    20: "float8_e5m2fnuz",
    21: "uint4",
    22: "int4",
    23: "float4_e2m1",
    24: "float8_e8m0",
    25: "uint2",
    26: "int2",
}

class Tensor:
    """执行阶段使用的张量值，由逻辑 dtype、形状和 NumPy 数据共同组成。

    传入 ``data`` 时构造器不会复制、重排或校验其形状和存储 dtype，调用方必须
    保证它与 ``size``/``dtype`` 一致；跨 C ABI 前会由算子路径建立连续副本。
    """
    
    def __init__(self, *size, dtype="float32", data=None):
        """创建张量值。

        Args:
            *size: 张量形状；单个 list 与逐维位置参数等价。
            dtype: 项目内部逻辑 dtype 名称。
            data: 可选的已有数据；缺省时按逻辑 dtype 创建全零数组。
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
    """图构建和形状推断阶段使用的纯元数据张量，不分配数值缓冲区。"""
    
    def __init__(self, *size, dtype="float32"):
        """记录占位形状和逻辑 dtype，形状参数规则与 :class:`Tensor` 一致。

        Args:
            *size: 张量的维度大小。
            dtype: 项目内部逻辑 dtype 名称。
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
    """内部算子的公共基类，统一 C 后端加载和张量桥接协议。

    子类的 ``forward`` 处理真实 :class:`Tensor`，``forward_`` 只传播
    :class:`Tensor_` 元数据。两条路径都由 :class:`Graph` 负责按 ONNX 边名接线。
    """
    _lib = None
    _lib_initialized = False

    @classmethod
    def _get_lib(cls):
        """加载进程级 C 后端并声明 ctypes 签名。

        ``CDLL`` 缓存在基类上，余弦查找表也只初始化一次。声明 ``argtypes``
        是 ABI 边界的一部分，可避免 Python 整数或指针被 ctypes 错误截断。

        Returns:
            ctypes.CDLL: 已初始化的 C 库实例。
        """
        if cls._lib is None:
            # 加载C库
            if not os.path.exists(TENSOR_OPS_LIB_PATH):
                raise FileNotFoundError(
                    f"C backend library not found: {TENSOR_OPS_LIB_PATH}. Run `make` first."
                )
            cls._lib = ctypes.CDLL(TENSOR_OPS_LIB_PATH)
            
            # create_tensor 返回 C 拥有的指针；缺少 restype 时 ctypes 会按 int 截断地址。
            cls._lib.create_tensor.restype = ctypes.POINTER(CTensor)
            
            # 基础 ABI 是所有构建版本共有的；后面的扩展量化符号允许旧库缺省。
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
    
    def _execute_unary(self, input_tensor, c_func_name):
        """执行形状不变的一元 C 算子，并将 C 输出复制回 Python 所有权。"""
        # C 张量拥有独立缓冲区；连续化后再复制可避免把带 stride 的 NumPy 视图误传给 C。
        in_data = np.ascontiguousarray(input_tensor.data)# 确保数据连续
        input_c = self._numpy_to_ctensor(in_data, input_tensor.dtype)
        
        # 2. 准备输出 (优先使用 self.dtype，否则沿用输入类型)
        out_dtype = self.dtype if self.dtype else input_tensor.dtype
        out_shape = (ctypes.c_int * len(input_tensor.size))(*input_tensor.size)
        output_c = self.lib.create_tensor(out_shape, len(input_tensor.size), DTYPE_MAP[out_dtype])
        
        # 3. 动态调用 C 函数
        getattr(self.lib, c_func_name)(input_c, output_c)
        
        # _ctensor_to_numpy 返回独立副本，所以随后释放 input_c/output_c 不会悬空。
        out_data = self._ctensor_to_numpy(output_c, out_dtype)
        self.lib.free_tensor(input_c)# 调用C函数释放内存
        self.lib.free_tensor(output_c)
        
        return Tensor(*input_tensor.size, dtype=out_dtype, data=out_data)
    
    def _execute_binary(self, input_a, input_b, c_func_name):
        """先在 Python 侧实现 ONNX 广播，再调用要求同形输入的二元 C 算子。"""
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
    
    def _execute_ternary(self, in_a, in_b, in_c, c_func_name, extra_int_arg=None):
        """执行带三路广播输入的 C 算子，主要服务量化/反量化路径。"""
        try:
            a_bc, b_bc, c_bc = np.broadcast_arrays(in_a.data, in_b.data, in_c.data)
        except ValueError as e:
            print(f"Broadcasting error in ternary op: {in_a.size}, {in_b.size}, {in_c.size}")
            raise e
            
        out_shape = a_bc.shape
        out_dtype = self.dtype if self.dtype else "float32"
        # 广播视图通常不连续，且逻辑 dtype 可能不同；逐路恢复存储 dtype 后再复制到 C。
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

    def forward(self, input):
        """真实张量执行入口，由具体算子覆盖。"""
        pass

    def forward_(self, input):
        """元数据推断入口，由具体算子覆盖且不得读取真实数值缓冲区。"""
        pass

    def _numpy_to_ctensor(self, arr: np.ndarray, dtype: str) -> ctypes.POINTER(CTensor):
        """把 NumPy 内容复制到新分配的 C 张量。

        返回值及其 shape/data 缓冲区归调用方所有，必须且只能调用一次
        ``free_tensor``；本函数不会让 C 张量借用 ``arr`` 的生命周期。

        Args:
            arr: 与 ``dtype`` 存储宽度一致的连续 NumPy 数组。
            dtype: 项目内部逻辑 dtype 名称。
            
        Returns:
            ctypes.POINTER(CTensor): C张量指针
        """
        # create_tensor 会复制 shape，因此局部 ctypes 数组在返回后无需继续存活。
        shape = (ctypes.c_int * len(arr.shape))(*arr.shape)
        # 创建C张量
        c_tensor = self.lib.create_tensor(shape, len(arr.shape), DTYPE_MAP[dtype])
        # 复制数据
        data_size = arr.size * arr.itemsize
        ctypes.memmove(c_tensor.contents.data, arr.ctypes.data, data_size)
        return c_tensor

    def _ctensor_to_numpy(self, c_tensor: ctypes.POINTER(CTensor), dtype: str) -> np.ndarray:
        """复制 C 张量内容为独立 NumPy 数组。

        中间的 ``frombuffer`` 视图借用 C 内存，返回前的 ``copy`` 是所有权边界；
        调用方可在本函数返回后立即释放 ``c_tensor``。

        Args:
            c_tensor: 仍然有效的 C 张量指针。
            dtype: 用于解释缓冲区元素宽度的逻辑 dtype。
            
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
    """按拓扑顺序执行内部算子，并以 ONNX 边名管理中间结果生命周期。

    ``ops`` 必须已经拓扑排序。图不会重新排序节点，而是用消费者计数在最后一次
    使用后释放 Python 边引用；显式图输出始终保留到本次执行结束。
    """
    
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

    def update(self, ops):
        """重建算子顺序、唯一名称和每条边的消费者计数。

        Args:
            ops: 已按依赖拓扑排序的操作节点列表。
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

            # output_in_degree 实际保存“剩余消费者数”，同时计入控制流捕获的外层边。
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

    def _init_edge_data(self, inputs):
        """按声明顺序绑定图输入；位置和数量都是公开调用契约。"""
        if len(inputs) != len(self.input_name):
            raise ValueError(
                f"Graph expects {len(self.input_name)} inputs, got {len(inputs)}"
            )
        return {na: inputs[idx] for idx, na in enumerate(self.input_name)}

    @staticmethod
    def _extract_tensor_result(outputs):
        """兼容算子直接返回张量或返回包含 ``tensor`` 的扩展结果字典。"""
        if isinstance(outputs, dict):
            if "tensor" in outputs:
                return outputs["tensor"]
            raise KeyError("operator result dict does not contain 'tensor'")
        return outputs

    @staticmethod
    def _normalize_multi_output(outputs, idx):
        """按 ONNX 输出槽位提取多输出结果，并拒绝静默丢失必需输出。"""
        if isinstance(outputs, (list, tuple)):
            if idx < len(outputs):
                return outputs[idx]
            raise IndexError(f"operator returned {len(outputs)} outputs, index {idx} requested")
        if idx == 0:
            return outputs
        raise TypeError(f"operator should return a list/tuple for multiple outputs, got {type(outputs)}")

    @staticmethod
    def _op_consumed_edges(op):
        """返回显式输入和控制流子图捕获的外层边，去除空槽位与重复名称。"""
        names = []
        for name in list(op.inputs) + list(getattr(op, "outer_scope_names", [])):
            if name and name not in names:
                names.append(name)
        return names

    def _collect_graph_outputs(self, edge_data_buffer):
        """按声明顺序收集输出；单输出保持历史上的标量式返回协议。"""
        missing = [name for name in self.output_name if name not in edge_data_buffer]
        if missing:
            raise KeyError(f"Graph output(s) not produced: {missing}")
        outputs = [edge_data_buffer[name] for name in self.output_name]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def forward(self, *inputs):
        """执行真实张量路径。

        ONNX 用空字符串保留缺省可选输入的位置，此处将其转换为 ``None``，但不
        改变后续参数索引。多输出结果同样按原始输出槽位分配，空输出名只跳过存储。
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

            # 删除的是 Python 引用；C 输出已在算子返回前复制，不依赖已释放的 C 缓冲区。
            for na in list(edge_data_buffer.keys()):
                if edge_usage.get(na, 0) == 0 and na not in protected_outputs:
                    edge_data_buffer.pop(na)

        return self._collect_graph_outputs(edge_data_buffer)

    def forward_(self, *inputs):
        """执行只传播 :class:`Tensor_` 的形状推断路径。

        该路径保留较宽松的历史多输出兼容行为，最终仍由 ``_collect_graph_outputs``
        检查所有声明的图输出是否实际生成。
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
