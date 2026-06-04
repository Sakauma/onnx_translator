# /**
#   ******************************************************************************
#   * @file        elementwise_activation_extra.py
#   * @author      Egor Izmaylov
#   * @brief       保存 `elementwise_activation_extra` 分组中的 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *

class Elu(Ops):
    # 初始化 `Elu` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, alpha=1.0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.elu_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]

    # 执行 `Elu` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.elu_forward(x_c, out_c, ctypes.c_float(self.alpha))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # 执行 `Elu` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Selu(Ops):
    # 初始化 `Selu` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, alpha=1.67326, gamma=1.0507, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.gamma = gamma
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.selu_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float, ctypes.c_float]

    # 执行 `Selu` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.selu_forward(x_c, out_c, ctypes.c_float(self.alpha), ctypes.c_float(self.gamma))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # 执行 `Selu` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class LeakyRelu(Ops):
    # 初始化 `LeakyRelu` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, alpha=0.01, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.leaky_relu_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]

    # 执行 `LeakyRelu` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.leaky_relu_forward(x_c, out_c, ctypes.c_float(self.alpha))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # 执行 `LeakyRelu` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class ThresholdedRelu(Ops):
    # 初始化 `ThresholdedRelu` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, alpha=1.0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.thresholded_relu_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]

    # 执行 `ThresholdedRelu` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.thresholded_relu_forward(x_c, out_c, ctypes.c_float(self.alpha))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # 执行 `ThresholdedRelu` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class HardSigmoid(Ops):
    # 初始化 `HardSigmoid` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, alpha=0.2, beta=0.5, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.beta = beta
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.hard_sigmoid_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float, ctypes.c_float]

    # 执行 `HardSigmoid` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.hard_sigmoid_forward(x_c, out_c, ctypes.c_float(self.alpha), ctypes.c_float(self.beta))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # 执行 `HardSigmoid` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Celu(Ops):
    # 初始化 `Celu` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, alpha=1.0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.celu_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]

    # 执行 `Celu` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.celu_forward(x_c, out_c, ctypes.c_float(self.alpha))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # 执行 `Celu` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Shrink(Ops):
    # 初始化 `Shrink` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, bias=0.0, lambd=0.5, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.bias = bias
        self.lambd = lambd
        self.dtype = dtype
        self.version = version
        if self.lib: self.lib.shrink_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float, ctypes.c_float]

    # 执行 `Shrink` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        self.lib.shrink_forward(x_c, out_c, ctypes.c_float(self.bias), ctypes.c_float(self.lambd))
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}
    
    # 执行 `Shrink` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Softplus(Ops):
    # 初始化 `Softplus` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Softplus` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x): return {"tensor": self._execute_unary(x, "softplus_forward"), "parameters": None}
    # 执行 `Softplus` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Softsign(Ops):
    # 初始化 `Softsign` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Softsign` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x): return {"tensor": self._execute_unary(x, "softsign_forward"), "parameters": None}
    # 执行 `Softsign` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class HardSwish(Ops):
    # 初始化 `HardSwish` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `HardSwish` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x): return {"tensor": self._execute_unary(x, "hard_swish_forward"), "parameters": None}
    # 执行 `HardSwish` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Acos(Ops):
    # 初始化 `Acos` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Acos` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x): return {"tensor": self._execute_unary(x, "acos_forward"), "parameters": None}
    # 执行 `Acos` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Asin(Ops):
    # 初始化 `Asin` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Asin` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x): return {"tensor": self._execute_unary(x, "asin_forward"), "parameters": None}
    # 执行 `Asin` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Cosh(Ops):
    # 初始化 `Cosh` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Cosh` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x): return {"tensor": self._execute_unary(x, "cosh_forward"), "parameters": None}
    # 执行 `Cosh` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Sinh(Ops):
    # 初始化 `Sinh` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Sinh` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x): return {"tensor": self._execute_unary(x, "sinh_forward"), "parameters": None}
    # 执行 `Sinh` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Asinh(Ops):
    # 初始化 `Asinh` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Asinh` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x): return {"tensor": self._execute_unary(x, "asinh_forward"), "parameters": None}
    # 执行 `Asinh` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Acosh(Ops):
    # 初始化 `Acosh` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Acosh` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x): return {"tensor": self._execute_unary(x, "acosh_forward"), "parameters": None}
    # 执行 `Acosh` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Atanh(Ops):
    # 初始化 `Atanh` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Atanh` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x): return {"tensor": self._execute_unary(x, "atanh_forward"), "parameters": None}
    # 执行 `Atanh` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Gelu(Ops):
    # 初始化 `Gelu` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Gelu` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x): return {"tensor": self._execute_unary(x, "gelu_forward"), "parameters": None}
    # 执行 `Gelu` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Mish(Ops):
    # 初始化 `Mish` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
    # 执行 `Mish` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x): return {"tensor": self._execute_unary(x, "mish_forward"), "parameters": None}
    # 执行 `Mish` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x): return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class Binarizer(Ops):
    # 初始化 `Binarizer` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, threshold=0.0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.threshold = threshold
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.binarizer_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_float]

    # 执行 `Binarizer` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        out_tensor = Tensor(*x.size, dtype=self.dtype)
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        out_c = self._numpy_to_ctensor(out_tensor.data, self.dtype)
        
        self.lib.binarizer_forward(x_c, out_c, ctypes.c_float(self.threshold))
        
        out_tensor.data = self._ctensor_to_numpy(out_c, self.dtype)
        self.lib.free_tensor(x_c); self.lib.free_tensor(out_c)
        return {"tensor": out_tensor, "parameters": None}

    # 执行 `Binarizer` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None}


class DynamicQuantizeLinear(Ops):
    # 初始化 `DynamicQuantizeLinear` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="uint8", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = "uint8"
        self.version = version
        if self.lib:
            self.lib.dynamic_quantize_linear_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), 
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)
            ]

    # 执行 `DynamicQuantizeLinear` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        # DynamicQuantizeLinear 的 y_scale 和 y_zero_point 是标量输出，不能建成一维长度 1 张量。
        y = Tensor(*x.size, dtype="uint8")
        y_scale = Tensor(dtype="float32")
        y_zp = Tensor(dtype="uint8")
        
        x_c = self._numpy_to_ctensor(x.data, x.dtype)
        y_c = self._numpy_to_ctensor(y.data, "uint8")
        scale_c = self._numpy_to_ctensor(y_scale.data, "float32")
        zp_c = self._numpy_to_ctensor(y_zp.data, "uint8")
        
        self.lib.dynamic_quantize_linear_forward(x_c, y_c, scale_c, zp_c)
        
        y.data = self._ctensor_to_numpy(y_c, "uint8")
        y_scale.data = self._ctensor_to_numpy(scale_c, "float32")
        y_zp.data = self._ctensor_to_numpy(zp_c, "uint8")
        
        self.lib.free_tensor(x_c); self.lib.free_tensor(y_c); self.lib.free_tensor(scale_c); self.lib.free_tensor(zp_c)
        
        return {"tensor": [y, y_scale, y_zp], "parameters": None}

    # 执行 `DynamicQuantizeLinear` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        return {
            "tensor": [Tensor_(*x.size, dtype="uint8"), Tensor_(dtype="float32"), Tensor_(dtype="uint8")],
            "parameters": None
        }
