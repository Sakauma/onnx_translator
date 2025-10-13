from nn import Ops
from nn import Tensor, Tensor_
import nn
import ctypes
import numpy as np
from typing import List, Union
import os


class RELU(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(RELU, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input: Tensor) -> Tensor:
        """RELU function implementation using C backend, computing with real data"""
        # Convert input to C tensor
        input_c = self._numpy_to_ctensor(input.data, self.dtype)
        # Create output C tensor
        output_shape = (ctypes.c_int * len(input.size))(*input.size)
        output_c = self.lib.create_tensor(output_shape, len(input.size), nn.DTYPE_MAP[self.dtype])
        # Call C function
        self.lib.relu_forward(input_c, output_c)
        # Convert back to numpy and create output tensor
        output_data = self._ctensor_to_numpy(output_c, self.dtype)
        output_tensor = Tensor(*input.size, dtype=self.dtype, data=output_data)
        # Clean up
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)
        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, input: Tensor_) -> Tensor_:
        """RELU function implementation using python implementation, computing without real data"""
        output_tensor = input
        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values


class COS(Ops):
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(COS, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    def forward(self, input: Tensor) -> Tensor:
        """COS function implementation using C backend, computing with real data"""
        # Convert input to C tensor
        input_c = self._numpy_to_ctensor(input.data, self.dtype)
        # Create output C tensor
        output_shape = (ctypes.c_int * len(input.size))(*input.size)
        output_c = self.lib.create_tensor(output_shape, len(input.size), nn.DTYPE_MAP[self.dtype])
        # Call C function
        self.lib.cos_forward(input_c, output_c)
        # Convert back to numpy and create output tensor
        output_data = self._ctensor_to_numpy(output_c, self.dtype)
        output_tensor = Tensor(*input.size, dtype=self.dtype, data=output_data)
        # Clean up
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)
        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

    def forward_(self, input: Tensor_) -> Tensor_:
        """cos function implementation using python implementation, computing without real data"""
        output_tensor = input
        values = {"tensor": output_tensor,
                  "parameters": None,
                  "graph": None}
        self.parameters = {"values": values}
        return values

