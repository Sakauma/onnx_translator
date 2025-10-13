import sys
from collections import OrderedDict
import ctypes
import numpy as np
from typing import List, Union
import os


class CTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_int)),
        ("ndim", ctypes.c_int),
        ("size", ctypes.c_size_t),
        ("dtype", ctypes.c_int)
    ]


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


# Data type mapping
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

# onnx data type mapping
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
    11: "double",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16"
}


class Tensor:
    def __init__(self, *size, dtype="float32", data=None):
        self.size = size[0] if (isinstance(size[0], list) and len(size) == 1) else size
        self.data_size = 1
        for s in self.size:
            self.data_size *= s
        self.dtype=dtype
        if data is not None:
            self.data = data
        else:
            np_dtype = DTYPE_TO_NUMPY[dtype]
            self.data = np.zeros(self.size, dtype=np_dtype)


class Tensor_:
    def __init__(self, *size, dtype= "float32"):
        self.size = size[0] if (isinstance(size[0], list) and len(size) == 1) else size
        self.data_size = 1
        for s in self.size:
            self.data_size *= s
        self.dtype = dtype


class Ops:
    _lib = None
    _lib_initialized = False

    @classmethod
    def _get_lib(cls):
        if cls._lib is None:
            cls._lib = ctypes.CDLL('./tensor_ops.so')
            cls._lib.create_tensor.restype = ctypes.POINTER(CTensor)
            cls._lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
            cls._lib.free_tensor.argtypes = [ctypes.POINTER(CTensor)]
            cls._lib.relu_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            cls._lib.cos_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
            cls._lib.init_cos_lut.argtypes = []
            cls._lib.init_cos_lut()
            cls._lib_initialized = True
        return cls._lib

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.parameters = {}
        self.name = None
        self.lib = self._get_lib()

    def forward(self, input):
        pass

    def forward_(self, input):
        pass

    def _numpy_to_ctensor(self, arr: np.ndarray, dtype: str) -> ctypes.POINTER(CTensor):
        shape = (ctypes.c_int * len(arr.shape))(*arr.shape)
        c_tensor = self.lib.create_tensor(shape, len(arr.shape), DTYPE_MAP[dtype])
        # Copy data
        data_size = arr.size * arr.itemsize
        ctypes.memmove(c_tensor.contents.data, arr.ctypes.data, data_size)
        return c_tensor

    def _ctensor_to_numpy(self, c_tensor: ctypes.POINTER(CTensor), dtype: str) -> np.ndarray:
        # Get shape
        shape = [c_tensor.contents.shape[i] for i in range(c_tensor.contents.ndim)]
        # Create numpy array from C data
        np_dtype = DTYPE_TO_NUMPY[dtype]
        arr = np.frombuffer(
            (ctypes.c_byte * (c_tensor.contents.size * np.dtype(np_dtype).itemsize)).from_address(c_tensor.contents.data),
            dtype=np_dtype
        ).reshape(shape)
        return arr.copy()


class Graph:
    def __init__(self, ops, input_name, output_name=None, model_name=None):
        self.input_name = input_name if isinstance(input_name, list) else [input_name]
        self.output_name = output_name if isinstance(output_name, list) else [output_name]
        self.ops = OrderedDict()
        self.update(ops)
        self.model_name = model_name

    def update(self, ops):
        name_dict = {}
        self.output_in_degree = {na: 0 for na in self.input_name}
        for op in ops:
            name = str(op.__class__).split("'")[1].split(".")[-1]
            if name not in name_dict:
                name_dict[name] = 0
            else:
                name_dict[name] += 1
            if not op.name:
                op.name = name + ".%d" % name_dict[name]
                self.ops[op.name] = op
            for i in op.inputs:
                if i in self.output_in_degree:
                    self.output_in_degree[i] += 1
                for o in op.outputs:
                    if o not in self.output_in_degree:
                        self.output_in_degree[o] = 0
                    else:
                        print("output edge name %s repeat!!!" % o)
                        sys.exit()
                if not self.output_name[0]:
                    for na in self.output_in_degree:
                        if self.output_in_degree[na] == 0:
                            self.output_name.append(na)
                            self.output_in_degree[na] = 1
                            self.output_name = self.output_name[1:]

    def forward(self, *inputs):
        edge_data_buffer = {}
        outputs = ()
        for idx, na in enumerate(self.input_name):
            edge_data_buffer[na] = inputs[idx]
        length = len(self.ops)
        for (cc, op_na) in zip(range(length), self.ops):
            op = self.ops[op_na]
            inputs = (edge_data_buffer[na] for na in op.inputs)
            outputs = op.forward(*inputs)
        if "graph" in outputs:
            outputs, graph = outputs["tensor"], outputs["graph"]
            do_graph = True
        elif "parameters" in outputs:
            outputs, parameters = outputs["tensor"], outputs["parameters"]
            do_graph = False
        for idx, inp_na in enumerate(op.inputs):
            self.output_in_degree[inp_na] -= 1
        for idx, out_na in enumerate(op.outputs):
            if len(op.outputs) == 1:
                edge_data_buffer[out_na] = outputs
                continue
            edge_data_buffer[out_na] = outputs[idx]
        for na in list(edge_data_buffer.keys()):
            if self.output_in_degree[na] == 0:
                edge_data_buffer.pop(na)

    def forward_(self, *inputs):
        edge_data_buffer = {}
        outputs = ()
        for idx, na in enumerate(self.input_name):
            edge_data_buffer[na] = inputs[idx]
        length = len(self.ops)
        for (cc, op_na) in zip(range(length), self.ops):
            op = self.ops[op_na]
            inputs = (edge_data_buffer[na] for na in op.inputs)
            outputs = op.forward_(*inputs)
        if "graph" in outputs:
            outputs, graph = outputs["tensor"], outputs["graph"]
            do_graph = True
        elif "parameters" in outputs:
            outputs, parameters = outputs["tensor"], outputs["parameters"]
            do_graph = False
        for idx, inp_na in enumerate(op.inputs):
            self.output_in_degree[inp_na] -= 1
        for idx, out_na in enumerate(op.outputs):
            if len(op.outputs) == 1:
                edge_data_buffer[out_na] = outputs
                continue
        edge_data_buffer[out_na] = outputs[idx]
        for na in list(edge_data_buffer.keys()):
            if self.output_in_degree[na] == 0:
                edge_data_buffer.pop(na)

