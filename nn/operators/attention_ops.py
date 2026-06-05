# /**
#   ******************************************************************************
#   * @file        attention_ops.py
#   * @author      Egor Izmaylov
#   * @brief       实现 Attention 算子的官方语义、C 后端主路径和形状推断。
#   * @details     2026.06.05  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *


class Attention(Ops):
    # 初始化 Attention 的头数、缩放、mask、softcap 和中间输出控制属性。
    def __init__(
        self,
        inputs,
        outputs,
        q_num_heads=None,
        kv_num_heads=None,
        scale=None,
        is_causal=0,
        softmax_precision=None,
        softcap=None,
        qk_matmul_output_mode=0,
        dtype="float32",
        version="24",
    ):
        super().__init__(inputs, outputs)
        self.q_num_heads = None if q_num_heads is None else int(q_num_heads)
        self.kv_num_heads = None if kv_num_heads is None else int(kv_num_heads)
        self.scale = None if scale is None else float(scale)
        self.is_causal = int(is_causal or 0)
        self.softmax_precision = None if softmax_precision is None else int(softmax_precision)
        self.softcap = None if softcap is None else float(softcap)
        self.qk_matmul_output_mode = int(qk_matmul_output_mode or 0)
        self.dtype = dtype
        self.version = version
        self._has_attention_c_backend = False
        if self.lib:
            try:
                self.lib.attention_forward.argtypes = [
                    ctypes.POINTER(CTensor),
                    ctypes.POINTER(CTensor),
                    ctypes.POINTER(CTensor),
                    ctypes.POINTER(CTensor),
                    ctypes.POINTER(CTensor),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                    ctypes.c_int,
                    ctypes.c_float,
                ]
                self._has_attention_c_backend = True
            except AttributeError:
                self._has_attention_c_backend = False

    # 判断当前调用是否请求了 present key/value 或 qk_matmul_output 等可选输出。
    def _needs_aux_outputs(self):
        return any(bool(name) for name in list(self.outputs)[1:])

    # 解析 Q/K/V 的 3D 或 4D 布局，并推导 Y、present cache 和 QK 中间张量形状。
    def _resolve_shapes(self, q_shape, k_shape, v_shape, past_key_shape=None, past_value_shape=None):
        if len(q_shape) != len(k_shape) or len(q_shape) != len(v_shape) or len(q_shape) not in {3, 4}:
            raise ValueError(f"Attention expects Q/K/V to be all 3D or all 4D, got {q_shape}, {k_shape}, {v_shape}")
        batch_size = int(q_shape[0])
        if int(k_shape[0]) != batch_size or int(v_shape[0]) != batch_size:
            raise ValueError("Attention Q/K/V batch dimensions must match")

        input_rank = len(q_shape)
        if input_rank == 3:
            if self.q_num_heads is None or self.kv_num_heads is None:
                raise ValueError("Attention requires q_num_heads and kv_num_heads for 3D Q/K/V")
            q_num_heads = int(self.q_num_heads)
            kv_num_heads = int(self.kv_num_heads)
            if q_num_heads <= 0 or kv_num_heads <= 0:
                raise ValueError("Attention q_num_heads and kv_num_heads must be positive")
            q_sequence_length = int(q_shape[1])
            kv_sequence_length = int(k_shape[1])
            if int(v_shape[1]) != kv_sequence_length:
                raise ValueError("Attention K and V sequence dimensions must match")
            if int(q_shape[2]) % q_num_heads != 0:
                raise ValueError("Attention Q hidden size must be divisible by q_num_heads")
            if int(k_shape[2]) % kv_num_heads != 0 or int(v_shape[2]) % kv_num_heads != 0:
                raise ValueError("Attention K/V hidden sizes must be divisible by kv_num_heads")
            head_size = int(q_shape[2]) // q_num_heads
            k_head_size = int(k_shape[2]) // kv_num_heads
            v_head_size = int(v_shape[2]) // kv_num_heads
            if head_size != k_head_size:
                raise ValueError("Attention Q and K head sizes must match")
            output_shape = (batch_size, q_sequence_length, q_num_heads * v_head_size)
        else:
            q_num_heads = int(q_shape[1])
            kv_num_heads = int(k_shape[1])
            q_sequence_length = int(q_shape[2])
            kv_sequence_length = int(k_shape[2])
            head_size = int(q_shape[3])
            if int(k_shape[3]) != head_size:
                raise ValueError("Attention Q and K head sizes must match")
            if int(v_shape[1]) != kv_num_heads or int(v_shape[2]) != kv_sequence_length:
                raise ValueError("Attention K and V head/sequence dimensions must match")
            v_head_size = int(v_shape[3])
            output_shape = (batch_size, q_num_heads, q_sequence_length, v_head_size)

        if q_num_heads <= 0 or kv_num_heads <= 0 or q_num_heads % kv_num_heads != 0:
            raise ValueError(f"Attention requires q_num_heads % kv_num_heads == 0, got {q_num_heads}/{kv_num_heads}")

        past_sequence_length = 0
        if past_key_shape is not None or past_value_shape is not None:
            if past_key_shape is None or past_value_shape is None:
                raise ValueError("Attention past_key and past_value must be provided together")
            expected_key_prefix = (batch_size, kv_num_heads)
            expected_value_prefix = (batch_size, kv_num_heads)
            if tuple(past_key_shape[:2]) != expected_key_prefix or int(past_key_shape[3]) != head_size:
                raise ValueError(f"Attention past_key shape is invalid: {past_key_shape}")
            if tuple(past_value_shape[:2]) != expected_value_prefix or int(past_value_shape[3]) != v_head_size:
                raise ValueError(f"Attention past_value shape is invalid: {past_value_shape}")
            if int(past_key_shape[2]) != int(past_value_shape[2]):
                raise ValueError("Attention past_key and past_value sequence dimensions must match")
            past_sequence_length = int(past_key_shape[2])

        total_sequence_length = past_sequence_length + kv_sequence_length
        present_key_shape = (batch_size, kv_num_heads, total_sequence_length, head_size)
        present_value_shape = (batch_size, kv_num_heads, total_sequence_length, v_head_size)
        qk_shape = (batch_size, q_num_heads, q_sequence_length, total_sequence_length)
        return {
            "input_rank": input_rank,
            "batch_size": batch_size,
            "q_num_heads": q_num_heads,
            "kv_num_heads": kv_num_heads,
            "q_sequence_length": q_sequence_length,
            "kv_sequence_length": kv_sequence_length,
            "head_size": head_size,
            "v_head_size": v_head_size,
            "output_shape": output_shape,
            "present_key_shape": present_key_shape,
            "present_value_shape": present_value_shape,
            "qk_shape": qk_shape,
        }

    # 执行 Attention 的 C 后端主路径，仅在 4D、无 cache 辅助输出时进入。
    def _forward_c(self, q, k, v, attn_mask, layout):
        q_c = self._numpy_to_ctensor(np.ascontiguousarray(q.data), q.dtype)
        k_c = self._numpy_to_ctensor(np.ascontiguousarray(k.data), k.dtype)
        v_c = self._numpy_to_ctensor(np.ascontiguousarray(v.data), v.dtype)
        mask_c = None if attn_mask is None else self._numpy_to_ctensor(np.ascontiguousarray(attn_mask.data), attn_mask.dtype)
        output_shape_c = (ctypes.c_int * len(layout["output_shape"]))(*layout["output_shape"])
        out_c = self.lib.create_tensor(output_shape_c, len(layout["output_shape"]), nn.DTYPE_MAP[self.dtype])
        self.lib.attention_forward(
            q_c,
            k_c,
            v_c,
            mask_c,
            out_c,
            ctypes.c_int(layout["q_num_heads"]),
            ctypes.c_int(layout["kv_num_heads"]),
            ctypes.c_float(-1.0 if self.scale is None else self.scale),
            ctypes.c_int(self.is_causal),
            ctypes.c_float(0.0 if self.softcap is None else self.softcap),
        )
        out_data = self._ctensor_to_numpy(out_c, self.dtype)
        for c_tensor in (q_c, k_c, v_c, mask_c, out_c):
            if c_tensor is not None:
                self.lib.free_tensor(c_tensor)
        return Tensor(*layout["output_shape"], dtype=self.dtype, data=out_data)

    # 执行 Attention 的真实张量计算路径，完整语义回落到 ONNX 官方 reference。
    def forward(self, q, k, v, attn_mask=None, past_key=None, past_value=None, nonpad_kv_seqlen=None):
        layout = self._resolve_shapes(
            q.size,
            k.size,
            v.size,
            None if past_key is None else past_key.size,
            None if past_value is None else past_value.size,
        )
        can_use_c = (
            self._has_attention_c_backend
            and layout["input_rank"] == 4
            and past_key is None
            and past_value is None
            and nonpad_kv_seqlen is None
            and not self._needs_aux_outputs()
            and q.dtype in nn.DTYPE_MAP
            and k.dtype in nn.DTYPE_MAP
            and v.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
            and (attn_mask is None or attn_mask.dtype in nn.DTYPE_MAP)
        )
        if can_use_c:
            out_tensor = self._forward_c(q, k, v, attn_mask, layout)
            return {"tensor": out_tensor, "parameters": None, "graph": None}

        from onnx.reference.ops.op_attention import _compute_attention

        y, present_key, present_value, qk_output = _compute_attention(
            _tensor_data_as_numeric(q),
            _tensor_data_as_numeric(k),
            _tensor_data_as_numeric(v),
            attn_mask=None if attn_mask is None else _tensor_data_as_numeric(attn_mask),
            past_key=None if past_key is None else _tensor_data_as_numeric(past_key),
            past_value=None if past_value is None else _tensor_data_as_numeric(past_value),
            nonpad_kv_seqlen=None if nonpad_kv_seqlen is None else np.asarray(nonpad_kv_seqlen.data, dtype=np.int64),
            scale=self.scale,
            is_causal=bool(self.is_causal),
            q_num_heads=self.q_num_heads,
            kv_num_heads=self.kv_num_heads,
            softmax_precision=self.softmax_precision,
            softcap=self.softcap,
            qk_matmul_output_mode=self.qk_matmul_output_mode,
        )
        y_tensor = Tensor(*y.shape, dtype=self.dtype, data=_cast_numeric_to_dtype(y, self.dtype))
        if not self._needs_aux_outputs():
            return {"tensor": y_tensor, "parameters": None, "graph": None}

        key_tensor = Tensor(*present_key.shape, dtype=k.dtype, data=_cast_numeric_to_dtype(present_key, k.dtype))
        value_tensor = Tensor(*present_value.shape, dtype=v.dtype, data=_cast_numeric_to_dtype(present_value, v.dtype))
        qk_tensor = Tensor(*qk_output.shape, dtype=self.dtype, data=_cast_numeric_to_dtype(qk_output, self.dtype))
        return {"tensor": [y_tensor, key_tensor, value_tensor, qk_tensor], "parameters": None, "graph": None}

    # 执行 Attention 的形状推断路径，按可选输出名称返回对应 Tensor_。
    def forward_(self, q, k, v, attn_mask=None, past_key=None, past_value=None, nonpad_kv_seqlen=None):
        layout = self._resolve_shapes(
            q.size,
            k.size,
            v.size,
            None if past_key is None else past_key.size,
            None if past_value is None else past_value.size,
        )
        y_tensor = Tensor_(*layout["output_shape"], dtype=self.dtype)
        if not self._needs_aux_outputs():
            return {"tensor": y_tensor, "parameters": None, "graph": None}
        return {
            "tensor": [
                y_tensor,
                Tensor_(*layout["present_key_shape"], dtype=k.dtype),
                Tensor_(*layout["present_value_shape"], dtype=v.dtype),
                Tensor_(*layout["qk_shape"], dtype=self.dtype),
            ],
            "parameters": None,
            "graph": None,
        }
