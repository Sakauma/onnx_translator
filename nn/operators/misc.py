"""文件功能：保存未归入主要算子域的兼容算子实现。
作者：Egor Izmaylov
时间：2026-06-02
"""

from .common import *

class CConvParams(ctypes.Structure):
    # ctypes 结构体字段顺序必须与 C 后端保持一致；卷积参数变更时需要同步 tensor_ops/tensor_ops.h。
    _fields_ = [
        ("pads", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("dilations", ctypes.POINTER(ctypes.c_int)),
        ("group", ctypes.c_int)
    ]


class CPoolParams(ctypes.Structure):
    # 池化算子将空间参数以原始 int 数组传入 C 后端，避免逐元素计算时回调 Python。
    _fields_ = [
        ("pads", ctypes.POINTER(ctypes.c_int)),
        ("strides", ctypes.POINTER(ctypes.c_int)),
        ("dilations", ctypes.POINTER(ctypes.c_int)),
        ("kernel_shape", ctypes.POINTER(ctypes.c_int))
    ]


class CReduceParams(ctypes.Structure):
    # 归约算子共享同一参数结构，因为它们主要差异在累积规则，axes/keepdims 语义基本一致。
    _fields_ = [
        ("axes", ctypes.POINTER(ctypes.c_int)),
        ("num_axes", ctypes.c_int),
        ("keepdims", ctypes.c_int)
    ]


class RNN(Ops):
    # 初始化 `RNN` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(
        self,
        inputs,
        outputs,
        hidden_size=None,
        direction="forward",
        activations=None,
        activation_alpha=None,
        activation_beta=None,
        clip=None,
        layout=0,
        dtype="float32",
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.hidden_size = hidden_size
        self.direction = direction
        self.activations = list(activations or ["Tanh", "Tanh"])
        self.activation_alpha = list(activation_alpha or [])
        self.activation_beta = list(activation_beta or [])
        self.clip = clip
        self.layout = layout
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.rnn_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
            ]

    # 封装 `_direction_code` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _direction_code(self):
        return {"forward": 0, "reverse": 1, "bidirectional": 2}[self.direction]

    # 封装 `_activation_buffers` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _activation_buffers(self):
        codes = []
        for activation in self.activations:
            name = activation.decode("utf-8") if isinstance(activation, bytes) else activation
            key = str(name).lower()
            if key not in _ACTIVATION_CODES:
                raise ValueError(f"Unsupported recurrent activation {activation!r}")
            codes.append(_ACTIVATION_CODES[key])
        if not codes:
            codes = [0]
        alphas = [np.nan] * len(codes)
        betas = [np.nan] * len(codes)
        for idx, value in enumerate(self.activation_alpha[: len(codes)]):
            alphas[idx] = float(value)
        for idx, value in enumerate(self.activation_beta[: len(codes)]):
            betas[idx] = float(value)
        return (
            (ctypes.c_int * len(codes))(*codes),
            (ctypes.c_float * len(codes))(*alphas),
            (ctypes.c_float * len(codes))(*betas),
            len(codes),
        )

    # 封装 `_optional_ctensor` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _optional_ctensor(self, tensor):
        if tensor is None:
            return None
        return self._numpy_to_ctensor(np.ascontiguousarray(tensor.data), tensor.dtype)

    # 封装 `_c_supported` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _c_supported(self, *tensors):
        return self.lib is not None and self.dtype in nn.DTYPE_MAP and all(
            tensor is None or tensor.dtype in nn.DTYPE_MAP for tensor in tensors
        )

    # 执行 `RNN` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, w, r, b=None, sequence_lens=None, initial_h=None):
        x_time = _recurrent_time_major(np.asarray(x.data), self.layout)
        seq_len, batch_size = x_time.shape[0], x_time.shape[1]
        num_dirs = w.data.shape[0]
        hidden = self.hidden_size or r.data.shape[-1]
        if self._c_supported(x, w, r, b, sequence_lens, initial_h):
            y_shape = (batch_size, seq_len, num_dirs, hidden) if self.layout == 1 else (seq_len, num_dirs, batch_size, hidden)
            y_h_shape = (num_dirs, batch_size, hidden)
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            w_c = self._numpy_to_ctensor(np.ascontiguousarray(w.data), w.dtype)
            r_c = self._numpy_to_ctensor(np.ascontiguousarray(r.data), r.dtype)
            b_c = self._optional_ctensor(b)
            seq_c = self._optional_ctensor(sequence_lens)
            init_c = self._optional_ctensor(initial_h)
            y_shape_c = (ctypes.c_int * 4)(*y_shape)
            yh_shape_c = (ctypes.c_int * 3)(*y_h_shape)
            y_c = self.lib.create_tensor(y_shape_c, 4, nn.DTYPE_MAP[self.dtype])
            yh_c = self.lib.create_tensor(yh_shape_c, 3, nn.DTYPE_MAP[self.dtype])
            act_codes, act_alpha, act_beta, act_count = self._activation_buffers()
            self.lib.rnn_forward(
                x_c, w_c, r_c, b_c, seq_c, init_c, y_c, yh_c,
                ctypes.c_int(hidden), ctypes.c_int(self._direction_code()), ctypes.c_int(self.layout),
                act_codes, act_alpha, act_beta, ctypes.c_int(act_count),
                ctypes.c_float(0.0 if self.clip is None else self.clip), ctypes.c_int(self.clip is not None),
            )
            y_data = self._ctensor_to_numpy(y_c, self.dtype)
            yh_data = self._ctensor_to_numpy(yh_c, self.dtype)
            for c_tensor in (x_c, w_c, r_c, b_c, seq_c, init_c, y_c, yh_c):
                if c_tensor is not None:
                    self.lib.free_tensor(c_tensor)
            outputs = (Tensor(*y_shape, dtype=self.dtype, data=y_data), Tensor(*y_h_shape, dtype=self.dtype, data=yh_data))
            return {"tensor": outputs[0] if len(self.outputs) == 1 else outputs, "parameters": None}

        bias = b.data if b is not None else np.zeros((num_dirs, 2 * hidden), dtype=x_time.dtype)
        h_prev = initial_h.data.copy() if initial_h is not None else np.zeros((num_dirs, batch_size, hidden), dtype=x_time.dtype)
        y = np.zeros((seq_len, num_dirs, batch_size, hidden), dtype=x_time.dtype)
        for direction_index in range(num_dirs):
            reverse = self.direction == "reverse" or (self.direction == "bidirectional" and direction_index == 1)
            time_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
            act = _activation_at(self.activations, self.activation_alpha, self.activation_beta, direction_index, "Tanh")
            wb, rb = np.split(bias[direction_index], 2)
            h_t = h_prev[direction_index]
            for t in time_indices:
                pre = x_time[t] @ w.data[direction_index].T + h_t @ r.data[direction_index].T + wb + rb
                h_new = act(_clip_if_needed(pre, self.clip))
                active = _sequence_mask(sequence_lens, t, batch_size)
                h_t = np.where(active, h_new, h_t)
                y[t, direction_index] = h_t
            h_prev[direction_index] = h_t
        y = _recurrent_output_layout(y, self.layout).astype(nn.DTYPE_TO_NUMPY.get(self.dtype, x_time.dtype), copy=False)
        y_h = h_prev.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, x_time.dtype), copy=False)
        outputs = (Tensor(*y.shape, dtype=self.dtype, data=y), Tensor(*y_h.shape, dtype=self.dtype, data=y_h))
        return {"tensor": outputs[0] if len(self.outputs) == 1 else outputs, "parameters": None}

    # 执行 `RNN` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, w, r, b=None, sequence_lens=None, initial_h=None):
        num_dirs = _num_directions(self.direction)
        hidden = self.hidden_size or r.size[-1]
        if self.layout == 1:
            batch_size, seq_len = x.size[0], x.size[1]
            y_shape = (batch_size, seq_len, num_dirs, hidden)
        else:
            seq_len, batch_size = x.size[0], x.size[1]
            y_shape = (seq_len, num_dirs, batch_size, hidden)
        y = Tensor_(*y_shape, dtype=self.dtype)
        y_h = Tensor_(num_dirs, batch_size, hidden, dtype=self.dtype)
        return {"tensor": y if len(self.outputs) == 1 else (y, y_h), "parameters": None}


class GRU(RNN):
    # 初始化 `GRU` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(
        self,
        inputs,
        outputs,
        hidden_size=None,
        direction="forward",
        activations=None,
        activation_alpha=None,
        activation_beta=None,
        clip=None,
        layout=0,
        linear_before_reset=0,
        dtype="float32",
        version="17",
    ):
        super().__init__(inputs, outputs, hidden_size, direction, activations or ["Sigmoid", "Tanh"] * _num_directions(direction), activation_alpha, activation_beta, clip, layout, dtype, version)
        self.linear_before_reset = linear_before_reset
        if self.lib:
            self.lib.gru_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
            ]

    # 执行 `GRU` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, w, r, b=None, sequence_lens=None, initial_h=None):
        x_time = _recurrent_time_major(np.asarray(x.data), self.layout)
        seq_len, batch_size = x_time.shape[0], x_time.shape[1]
        num_dirs = w.data.shape[0]
        hidden = self.hidden_size or r.data.shape[-1]
        if self._c_supported(x, w, r, b, sequence_lens, initial_h):
            y_shape = (batch_size, seq_len, num_dirs, hidden) if self.layout == 1 else (seq_len, num_dirs, batch_size, hidden)
            y_h_shape = (num_dirs, batch_size, hidden)
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            w_c = self._numpy_to_ctensor(np.ascontiguousarray(w.data), w.dtype)
            r_c = self._numpy_to_ctensor(np.ascontiguousarray(r.data), r.dtype)
            b_c = self._optional_ctensor(b)
            seq_c = self._optional_ctensor(sequence_lens)
            init_c = self._optional_ctensor(initial_h)
            y_shape_c = (ctypes.c_int * 4)(*y_shape)
            yh_shape_c = (ctypes.c_int * 3)(*y_h_shape)
            y_c = self.lib.create_tensor(y_shape_c, 4, nn.DTYPE_MAP[self.dtype])
            yh_c = self.lib.create_tensor(yh_shape_c, 3, nn.DTYPE_MAP[self.dtype])
            act_codes, act_alpha, act_beta, act_count = self._activation_buffers()
            self.lib.gru_forward(
                x_c, w_c, r_c, b_c, seq_c, init_c, y_c, yh_c,
                ctypes.c_int(hidden), ctypes.c_int(self._direction_code()), ctypes.c_int(self.layout),
                ctypes.c_int(self.linear_before_reset), act_codes, act_alpha, act_beta,
                ctypes.c_int(act_count), ctypes.c_float(0.0 if self.clip is None else self.clip),
                ctypes.c_int(self.clip is not None),
            )
            y_data = self._ctensor_to_numpy(y_c, self.dtype)
            yh_data = self._ctensor_to_numpy(yh_c, self.dtype)
            for c_tensor in (x_c, w_c, r_c, b_c, seq_c, init_c, y_c, yh_c):
                if c_tensor is not None:
                    self.lib.free_tensor(c_tensor)
            outputs = (Tensor(*y_shape, dtype=self.dtype, data=y_data), Tensor(*y_h_shape, dtype=self.dtype, data=yh_data))
            return {"tensor": outputs[0] if len(self.outputs) == 1 else outputs, "parameters": None}

        bias = b.data if b is not None else np.zeros((num_dirs, 6 * hidden), dtype=x_time.dtype)
        h_prev = initial_h.data.copy() if initial_h is not None else np.zeros((num_dirs, batch_size, hidden), dtype=x_time.dtype)
        y = np.zeros((seq_len, num_dirs, batch_size, hidden), dtype=x_time.dtype)
        for direction_index in range(num_dirs):
            reverse = self.direction == "reverse" or (self.direction == "bidirectional" and direction_index == 1)
            time_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
            f = _activation_at(self.activations, self.activation_alpha, self.activation_beta, direction_index * 2, "Sigmoid")
            g = _activation_at(self.activations, self.activation_alpha, self.activation_beta, direction_index * 2 + 1, "Tanh")
            wz, wr, wh = np.split(w.data[direction_index], 3)
            rz, rr, rh = np.split(r.data[direction_index], 3)
            wbz, wbr, wbh, rbz, rbr, rbh = np.split(bias[direction_index], 6)
            h_t = h_prev[direction_index]
            for t in time_indices:
                z = f(_clip_if_needed(x_time[t] @ wz.T + h_t @ rz.T + wbz + rbz, self.clip))
                reset = f(_clip_if_needed(x_time[t] @ wr.T + h_t @ rr.T + wbr + rbr, self.clip))
                if self.linear_before_reset:
                    h_candidate = g(_clip_if_needed(x_time[t] @ wh.T + reset * (h_t @ rh.T + rbh) + wbh, self.clip))
                else:
                    h_candidate = g(_clip_if_needed(x_time[t] @ wh.T + (reset * h_t) @ rh.T + wbh + rbh, self.clip))
                h_new = (1.0 - z) * h_candidate + z * h_t
                active = _sequence_mask(sequence_lens, t, batch_size)
                h_t = np.where(active, h_new, h_t)
                y[t, direction_index] = h_t
            h_prev[direction_index] = h_t
        y = _recurrent_output_layout(y, self.layout).astype(nn.DTYPE_TO_NUMPY.get(self.dtype, x_time.dtype), copy=False)
        y_h = h_prev.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, x_time.dtype), copy=False)
        outputs = (Tensor(*y.shape, dtype=self.dtype, data=y), Tensor(*y_h.shape, dtype=self.dtype, data=y_h))
        return {"tensor": outputs[0] if len(self.outputs) == 1 else outputs, "parameters": None}


class LSTM(RNN):
    # 初始化 `LSTM` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(
        self,
        inputs,
        outputs,
        hidden_size=None,
        direction="forward",
        activations=None,
        activation_alpha=None,
        activation_beta=None,
        clip=None,
        layout=0,
        input_forget=0,
        dtype="float32",
        version="17",
    ):
        super().__init__(inputs, outputs, hidden_size, direction, activations or ["Sigmoid", "Tanh", "Tanh"] * _num_directions(direction), activation_alpha, activation_beta, clip, layout, dtype, version)
        self.input_forget = input_forget
        if self.lib:
            self.lib.lstm_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_int,
            ]

    # 执行 `LSTM` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x, w, r, b=None, sequence_lens=None, initial_h=None, initial_c=None, p=None):
        x_time = _recurrent_time_major(np.asarray(x.data), self.layout)
        seq_len, batch_size = x_time.shape[0], x_time.shape[1]
        num_dirs = w.data.shape[0]
        hidden = self.hidden_size or r.data.shape[-1]
        if self._c_supported(x, w, r, b, sequence_lens, initial_h, initial_c, p):
            y_shape = (batch_size, seq_len, num_dirs, hidden) if self.layout == 1 else (seq_len, num_dirs, batch_size, hidden)
            y_h_shape = (num_dirs, batch_size, hidden)
            x_c = self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype)
            w_c = self._numpy_to_ctensor(np.ascontiguousarray(w.data), w.dtype)
            r_c = self._numpy_to_ctensor(np.ascontiguousarray(r.data), r.dtype)
            b_c = self._optional_ctensor(b)
            seq_c = self._optional_ctensor(sequence_lens)
            init_h_c = self._optional_ctensor(initial_h)
            init_c_c = self._optional_ctensor(initial_c)
            p_c = self._optional_ctensor(p)
            y_shape_c = (ctypes.c_int * 4)(*y_shape)
            yh_shape_c = (ctypes.c_int * 3)(*y_h_shape)
            y_c = self.lib.create_tensor(y_shape_c, 4, nn.DTYPE_MAP[self.dtype])
            yh_c = self.lib.create_tensor(yh_shape_c, 3, nn.DTYPE_MAP[self.dtype])
            yc_c = self.lib.create_tensor(yh_shape_c, 3, nn.DTYPE_MAP[self.dtype])
            act_codes, act_alpha, act_beta, act_count = self._activation_buffers()
            self.lib.lstm_forward(
                x_c, w_c, r_c, b_c, seq_c, init_h_c, init_c_c, p_c,
                y_c, yh_c, yc_c, ctypes.c_int(hidden), ctypes.c_int(self._direction_code()),
                ctypes.c_int(self.layout), ctypes.c_int(self.input_forget), act_codes,
                act_alpha, act_beta, ctypes.c_int(act_count),
                ctypes.c_float(0.0 if self.clip is None else self.clip), ctypes.c_int(self.clip is not None),
            )
            y_data = self._ctensor_to_numpy(y_c, self.dtype)
            yh_data = self._ctensor_to_numpy(yh_c, self.dtype)
            yc_data = self._ctensor_to_numpy(yc_c, self.dtype)
            for c_tensor in (x_c, w_c, r_c, b_c, seq_c, init_h_c, init_c_c, p_c, y_c, yh_c, yc_c):
                if c_tensor is not None:
                    self.lib.free_tensor(c_tensor)
            outputs = (
                Tensor(*y_shape, dtype=self.dtype, data=y_data),
                Tensor(*y_h_shape, dtype=self.dtype, data=yh_data),
                Tensor(*y_h_shape, dtype=self.dtype, data=yc_data),
            )
            selected = tuple(value for name, value in zip(self.outputs, outputs) if name)
            return {"tensor": selected[0] if len(selected) == 1 else selected, "parameters": None}

        bias = b.data if b is not None else np.zeros((num_dirs, 8 * hidden), dtype=x_time.dtype)
        peepholes = p.data if p is not None else np.zeros((num_dirs, 3 * hidden), dtype=x_time.dtype)
        h_prev = initial_h.data.copy() if initial_h is not None else np.zeros((num_dirs, batch_size, hidden), dtype=x_time.dtype)
        c_prev = initial_c.data.copy() if initial_c is not None else np.zeros((num_dirs, batch_size, hidden), dtype=x_time.dtype)
        y = np.zeros((seq_len, num_dirs, batch_size, hidden), dtype=x_time.dtype)
        for direction_index in range(num_dirs):
            reverse = self.direction == "reverse" or (self.direction == "bidirectional" and direction_index == 1)
            time_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
            f_act = _activation_at(self.activations, self.activation_alpha, self.activation_beta, direction_index * 3, "Sigmoid")
            g_act = _activation_at(self.activations, self.activation_alpha, self.activation_beta, direction_index * 3 + 1, "Tanh")
            h_act = _activation_at(self.activations, self.activation_alpha, self.activation_beta, direction_index * 3 + 2, "Tanh")
            p_i, p_o, p_f = np.split(peepholes[direction_index], 3)
            h_t = h_prev[direction_index]
            c_t = c_prev[direction_index]
            bias_sum = np.add(*np.split(bias[direction_index], 2))
            for t in time_indices:
                i, o, forget, c = np.split(x_time[t] @ w.data[direction_index].T + h_t @ r.data[direction_index].T + bias_sum, 4, axis=-1)
                i = f_act(_clip_if_needed(i + p_i * c_t, self.clip))
                forget = 1.0 - i if self.input_forget else f_act(_clip_if_needed(forget + p_f * c_t, self.clip))
                c_bar = g_act(_clip_if_needed(c, self.clip))
                c_new = forget * c_t + i * c_bar
                o = f_act(_clip_if_needed(o + p_o * c_new, self.clip))
                h_new = o * h_act(c_new)
                active = _sequence_mask(sequence_lens, t, batch_size)
                h_t = np.where(active, h_new, h_t)
                c_t = np.where(active, c_new, c_t)
                y[t, direction_index] = h_t
            h_prev[direction_index] = h_t
            c_prev[direction_index] = c_t
        y = _recurrent_output_layout(y, self.layout).astype(nn.DTYPE_TO_NUMPY.get(self.dtype, x_time.dtype), copy=False)
        y_h = h_prev.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, x_time.dtype), copy=False)
        y_c = c_prev.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, x_time.dtype), copy=False)
        outputs = (
            Tensor(*y.shape, dtype=self.dtype, data=y),
            Tensor(*y_h.shape, dtype=self.dtype, data=y_h),
            Tensor(*y_c.shape, dtype=self.dtype, data=y_c),
        )
        selected = tuple(value for name, value in zip(self.outputs, outputs) if name)
        return {"tensor": selected[0] if len(selected) == 1 else selected, "parameters": None}

    # 执行 `LSTM` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x, w, r, b=None, sequence_lens=None, initial_h=None, initial_c=None, p=None):
        num_dirs = _num_directions(self.direction)
        hidden = self.hidden_size or r.size[-1]
        if self.layout == 1:
            batch_size, seq_len = x.size[0], x.size[1]
            y_shape = (batch_size, seq_len, num_dirs, hidden)
        else:
            seq_len, batch_size = x.size[0], x.size[1]
            y_shape = (seq_len, num_dirs, batch_size, hidden)
        y = Tensor_(*y_shape, dtype=self.dtype)
        y_h = Tensor_(num_dirs, batch_size, hidden, dtype=self.dtype)
        y_c = Tensor_(num_dirs, batch_size, hidden, dtype=self.dtype)
        selected = tuple(value for name, value in zip(self.outputs, (y, y_h, y_c)) if name)
        return {"tensor": selected[0] if len(selected) == 1 else selected, "parameters": None}


class Clip(Ops):
    # 初始化 `Clip` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype, version="17"):
        super(Clip, self).__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version
        
        if self.lib:
             self.lib.clip_forward.argtypes = [
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor), 
                ctypes.POINTER(nn.CTensor), ctypes.POINTER(nn.CTensor)
            ]

    # 执行 `Clip` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input: Tensor, min_val: Tensor = None, max_val: Tensor = None) -> dict:
        # 1. 准备广播列表
        # 注意：ONNX 中 min/max 是可选的，可能为 None
        broadcast_list = [input.data]
        if min_val is not None: broadcast_list.append(min_val.data)
        if max_val is not None: broadcast_list.append(max_val.data)

        # 2. 执行广播 (numpy 会自动处理标量与张量的广播)
        try:
            broadcasted = np.broadcast_arrays(*broadcast_list)
        except ValueError:
             raise ValueError(f"Clip inputs shape mismatch: input={input.size}, "
                              f"min={min_val.size if min_val else 'None'}, "
                              f"max={max_val.size if max_val else 'None'}")

        # 3. 提取广播后的数据
        # broadcasted[0] 对应 input
        input_data = np.ascontiguousarray(broadcasted[0])
        
        idx = 1
        min_data = None
        if min_val is not None:
            min_data = np.ascontiguousarray(broadcasted[idx])
            idx += 1
            
        max_data = None
        if max_val is not None:
            max_data = np.ascontiguousarray(broadcasted[idx])
        
        # 4. 准备 C Tensor
        # 此时 input_data, min_data, max_data 的 shape 应该完全一致
        out_shape = input_data.shape
        if self.lib is None or self.dtype not in nn.DTYPE_MAP:
            out_data = input_data
            if min_data is not None:
                out_data = np.maximum(out_data, min_data)
            if max_data is not None:
                out_data = np.minimum(out_data, max_data)
            out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None, "graph": None}
        
        input_c = self._numpy_to_ctensor(input_data, self.dtype) # 使用广播后的数据
        output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
        output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
        
        min_c = self._numpy_to_ctensor(min_data, min_val.dtype if min_val else "float32") if min_data is not None else ctypes.POINTER(nn.CTensor)()
        max_c = self._numpy_to_ctensor(max_data, max_val.dtype if max_val else "float32") if max_data is not None else ctypes.POINTER(nn.CTensor)()
        
        # 5. 调用 C 函数
        self.lib.clip_forward(input_c, output_c, min_c, max_c)
        
        out_data = self._ctensor_to_numpy(output_c, self.dtype)
        
        # 6. 资源释放
        self.lib.free_tensor(input_c)
        self.lib.free_tensor(output_c)
        if min_data is not None: self.lib.free_tensor(min_c)
        if max_data is not None: self.lib.free_tensor(max_c)

        out_tensor = Tensor(*out_shape, dtype=self.dtype, data=out_data)
        return {"tensor": out_tensor, "parameters": None, "graph": None}

    # 执行 `Clip` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input: Tensor_, min_val: Tensor_ = None, max_val: Tensor_ = None) -> dict:
        # 图推断模式：简单假设输出形状与输入一致（实际应考虑广播）
        # 假设 min/max 是标量或可广播，输出 shape 由 input 主导
        try:
            shapes = [input.size]
            if min_val: shapes.append(min_val.size)
            if max_val: shapes.append(max_val.size)
            out_shape = np.broadcast_shapes(*shapes)
        except:
            out_shape = input.size
        output_tensor = Tensor_(*out_shape, dtype=self.dtype)
        return {"tensor": output_tensor, "parameters": None, "graph": None}
