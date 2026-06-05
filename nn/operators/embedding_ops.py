# /**
#   ******************************************************************************
#   * @file        embedding_ops.py
#   * @author      Egor Izmaylov
#   * @brief       保存位置编码和嵌入变换相关的 ONNX 算子实现。
#   * @details     2026.06.05  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *


class RotaryEmbedding(Ops):
    # 初始化 `RotaryEmbedding` 的构造参数，保存 RoPE 维度、头数、交错布局和输出 dtype。
    def __init__(
        self,
        inputs,
        outputs,
        num_heads=0,
        rotary_embedding_dim=0,
        interleaved=0,
        dtype="float32",
        version="23",
    ):
        super().__init__(inputs, outputs)
        self.num_heads = int(num_heads or 0)
        self.rotary_embedding_dim = int(rotary_embedding_dim or 0)
        self.interleaved = int(interleaved or 0)
        self.dtype = dtype
        self.version = version
        self._has_rotary_embedding_c_backend = False
        if self.lib:
            try:
                self.lib.rotary_embedding_forward.argtypes = [
                    ctypes.POINTER(CTensor),
                    ctypes.POINTER(CTensor),
                    ctypes.POINTER(CTensor),
                    ctypes.POINTER(CTensor),
                    ctypes.POINTER(CTensor),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                self._has_rotary_embedding_c_backend = True
            except AttributeError:
                self._has_rotary_embedding_c_backend = False

    # 根据输入 rank、num_heads 和 rotary_embedding_dim 解析 RoPE 使用的内部形状。
    def _resolve_layout(self, x_shape, cos_shape, sin_shape, position_shape=None):
        if len(x_shape) == 4:
            batch_size, num_heads, sequence_length, head_size = map(int, x_shape)
        elif len(x_shape) == 3:
            batch_size, sequence_length, hidden_size = map(int, x_shape)
            if self.num_heads <= 0:
                raise ValueError("RotaryEmbedding requires num_heads for 3D input")
            if hidden_size % self.num_heads != 0:
                raise ValueError(f"RotaryEmbedding hidden_size {hidden_size} is not divisible by num_heads {self.num_heads}")
            num_heads = self.num_heads
            head_size = hidden_size // num_heads
        else:
            raise ValueError(f"RotaryEmbedding expects 3D or 4D input, got shape {x_shape}")

        rotary_dim = self.rotary_embedding_dim if self.rotary_embedding_dim > 0 else head_size
        if rotary_dim <= 0 or rotary_dim > head_size:
            raise ValueError(f"RotaryEmbedding rotary_embedding_dim {rotary_dim} is invalid for head_size {head_size}")
        if rotary_dim % 2 != 0:
            raise ValueError("RotaryEmbedding rotary_embedding_dim must be even")
        rotary_half = rotary_dim // 2

        if tuple(cos_shape)[-1] != rotary_half:
            raise ValueError(f"Last dimension of cos_cache ({tuple(cos_shape)[-1]}) must equal rotary_embedding_dim/2 ({rotary_half})")
        if tuple(sin_shape)[-1] != rotary_half:
            raise ValueError(f"Last dimension of sin_cache ({tuple(sin_shape)[-1]}) must equal rotary_embedding_dim/2 ({rotary_half})")
        if position_shape is not None:
            if tuple(position_shape) != (batch_size, sequence_length):
                raise ValueError(f"RotaryEmbedding position_ids shape must be {(batch_size, sequence_length)}, got {position_shape}")
            if len(cos_shape) != 2 or len(sin_shape) != 2:
                raise ValueError("RotaryEmbedding expects 2D cos/sin cache when position_ids is provided")
        else:
            expected_cache = (batch_size, sequence_length, rotary_half)
            if tuple(cos_shape) != expected_cache or tuple(sin_shape) != expected_cache:
                raise ValueError(f"RotaryEmbedding expects 3D cos/sin cache {expected_cache} when position_ids is omitted")
        return batch_size, sequence_length, num_heads, head_size, rotary_dim

    # 执行 `RotaryEmbedding` 的真实张量计算路径，按官方 RoPE 公式旋转每个 head 的前缀维度。
    def forward(self, x, cos_cache, sin_cache, position_ids=None):
        x_data = _tensor_data_as_numeric(x)
        cos_data = _tensor_data_as_numeric(cos_cache)
        sin_data = _tensor_data_as_numeric(sin_cache)
        position_shape = None if position_ids is None else position_ids.size
        self._resolve_layout(x_data.shape, cos_data.shape, sin_data.shape, position_shape)

        if (
            self._has_rotary_embedding_c_backend
            and x.dtype in nn.DTYPE_MAP
            and cos_cache.dtype in nn.DTYPE_MAP
            and sin_cache.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
            and (position_ids is None or position_ids.dtype in nn.DTYPE_MAP)
        ):
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            cos_c = self._numpy_to_ctensor(np.ascontiguousarray(cos_cache.data), cos_cache.dtype)
            sin_c = self._numpy_to_ctensor(np.ascontiguousarray(sin_cache.data), sin_cache.dtype)
            pos_c = None if position_ids is None else self._numpy_to_ctensor(np.ascontiguousarray(position_ids.data), position_ids.dtype)
            output_shape_c = (ctypes.c_int * len(x.size))(*x.size)
            out_c = self.lib.create_tensor(output_shape_c, len(x.size), nn.DTYPE_MAP[self.dtype])
            self.lib.rotary_embedding_forward(
                x_c,
                cos_c,
                sin_c,
                pos_c,
                out_c,
                ctypes.c_int(self.num_heads),
                ctypes.c_int(self.rotary_embedding_dim),
                ctypes.c_int(self.interleaved),
            )
            out_data = self._ctensor_to_numpy(out_c, self.dtype)
            for c_tensor in (x_c, cos_c, sin_c, pos_c, out_c):
                if c_tensor is not None:
                    self.lib.free_tensor(c_tensor)
            return {"tensor": Tensor(*x.size, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

        original_shape = x_data.shape
        if x_data.ndim == 4:
            work = np.transpose(x_data, (0, 2, 1, 3)).astype(np.float32, copy=False)
        else:
            batch_size, sequence_length, hidden_size = x_data.shape
            head_size = hidden_size // self.num_heads
            work = x_data.reshape(batch_size, sequence_length, self.num_heads, head_size).astype(np.float32, copy=False)

        head_size = work.shape[-1]
        rotary_dim = self.rotary_embedding_dim if self.rotary_embedding_dim > 0 else head_size
        rotary_half = rotary_dim // 2
        x_rotate = work[..., :rotary_dim]
        x_not_rotate = work[..., rotary_dim:]

        cos_values = cos_data.astype(np.float32, copy=False)
        sin_values = sin_data.astype(np.float32, copy=False)
        if position_ids is not None:
            positions = np.asarray(position_ids.data, dtype=np.int64)
            cos_values = cos_values[positions]
            sin_values = sin_values[positions]
        cos_values = np.expand_dims(cos_values, axis=2)
        sin_values = np.expand_dims(sin_values, axis=2)

        if self.interleaved:
            x1 = x_rotate[..., 0::2]
            x2 = x_rotate[..., 1::2]
        else:
            x1 = x_rotate[..., :rotary_half]
            x2 = x_rotate[..., rotary_half:]
        real = cos_values * x1 - sin_values * x2
        imag = sin_values * x1 + cos_values * x2
        if self.interleaved:
            rotated = np.stack((real, imag), axis=-1).reshape(x_rotate.shape)
        else:
            rotated = np.concatenate((real, imag), axis=-1)
        output = np.concatenate((rotated, x_not_rotate), axis=-1)
        if len(original_shape) == 3:
            output = output.reshape(original_shape)
        else:
            output = np.transpose(output, (0, 2, 1, 3))
        out_data = _cast_numeric_to_dtype(output, self.dtype)
        return {"tensor": Tensor(*original_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}

    # 执行 `RotaryEmbedding` 的形状推断路径，只生成与输入同形状的 `Tensor_` 元数据。
    def forward_(self, x, cos_cache, sin_cache, position_ids=None):
        position_shape = None if position_ids is None else position_ids.size
        self._resolve_layout(x.size, cos_cache.size, sin_cache.size, position_shape)
        return {"tensor": Tensor_(*x.size, dtype=self.dtype), "parameters": None, "graph": None}
