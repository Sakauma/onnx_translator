# /**
#   ******************************************************************************
#   * @file        test_operator_recurrent_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用独立公式验证 RNN、GRU、LSTM 的 ONNX17 循环语义。
#   * @details     2026.06.04  V1.0.0  创建
#   * @details     2026.06.14  V1.0.1  补充 sequence_lens 为 0 时保持初始状态的边界验证
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from conftest import _disable_c_backend
from operator_test_context import *  # noqa: F401,F403


# 计算 Sigmoid 激活，供 GRU/LSTM 的独立参考公式复用。
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 按 ONNX layout 属性把输入规整为 time-major，便于循环公式统一处理。
def _to_time_major(x, layout):
    return np.swapaxes(x, 0, 1) if layout == 1 else x


# 按 ONNX layout 属性把 time-major 的 Y 输出恢复为算子约定布局。
def _from_time_major(y, layout):
    return np.transpose(y, (2, 0, 1, 3)) if layout == 1 else y


# 根据 direction 属性返回某个方向实际访问的时间步顺序。
def _time_indices(seq_len, direction, direction_index):
    reverse = direction == "reverse" or (direction == "bidirectional" and direction_index == 1)
    return range(seq_len - 1, -1, -1) if reverse else range(seq_len)


# 根据 sequence_lens 判断某个 batch 在当前物理时间步是否仍参与状态更新。
def _active_mask(sequence_lens, t, batch_size):
    if sequence_lens is None:
        return np.ones((batch_size, 1), dtype=bool)
    return (np.asarray(sequence_lens).reshape(-1) > t).reshape(batch_size, 1)


# 用 ONNX 官方 RNN 公式独立计算 Y/Y_h，覆盖 direction、layout、sequence_lens 和 initial_h。
def _rnn_reference(x, w, r, b=None, sequence_lens=None, initial_h=None, direction="forward", layout=0):
    x_time = _to_time_major(np.asarray(x, dtype=np.float64), layout)
    seq_len, batch_size = x_time.shape[0], x_time.shape[1]
    num_dirs, hidden = w.shape[0], r.shape[-1]
    bias = np.zeros((num_dirs, 2 * hidden), dtype=np.float64) if b is None else np.asarray(b, dtype=np.float64)
    h_state = (
        np.zeros((num_dirs, batch_size, hidden), dtype=np.float64)
        if initial_h is None
        else np.asarray(initial_h, dtype=np.float64).copy()
    )
    y = np.zeros((seq_len, num_dirs, batch_size, hidden), dtype=np.float64)
    for direction_index in range(num_dirs):
        wb, rb = np.split(bias[direction_index], 2)
        for t in _time_indices(seq_len, direction, direction_index):
            pre = x_time[t] @ w[direction_index].T + h_state[direction_index] @ r[direction_index].T + wb + rb
            h_new = np.tanh(pre)
            active = _active_mask(sequence_lens, t, batch_size)
            h_state[direction_index] = np.where(active, h_new, h_state[direction_index])
            y[t, direction_index] = h_state[direction_index]
    return _from_time_major(y, layout), h_state


# 用 ONNX 官方 GRU 公式独立计算 Y/Y_h，显式覆盖 linear_before_reset 分支。
def _gru_reference(
    x,
    w,
    r,
    b=None,
    sequence_lens=None,
    initial_h=None,
    direction="forward",
    layout=0,
    linear_before_reset=0,
):
    x_time = _to_time_major(np.asarray(x, dtype=np.float64), layout)
    seq_len, batch_size = x_time.shape[0], x_time.shape[1]
    num_dirs, hidden = w.shape[0], r.shape[-1]
    bias = np.zeros((num_dirs, 6 * hidden), dtype=np.float64) if b is None else np.asarray(b, dtype=np.float64)
    h_state = (
        np.zeros((num_dirs, batch_size, hidden), dtype=np.float64)
        if initial_h is None
        else np.asarray(initial_h, dtype=np.float64).copy()
    )
    y = np.zeros((seq_len, num_dirs, batch_size, hidden), dtype=np.float64)
    for direction_index in range(num_dirs):
        wz, wr, wh = np.split(w[direction_index], 3)
        rz, rr, rh = np.split(r[direction_index], 3)
        wbz, wbr, wbh, rbz, rbr, rbh = np.split(bias[direction_index], 6)
        for t in _time_indices(seq_len, direction, direction_index):
            h_prev = h_state[direction_index]
            z = _sigmoid(x_time[t] @ wz.T + h_prev @ rz.T + wbz + rbz)
            reset = _sigmoid(x_time[t] @ wr.T + h_prev @ rr.T + wbr + rbr)
            if linear_before_reset:
                h_candidate = np.tanh(x_time[t] @ wh.T + reset * (h_prev @ rh.T + rbh) + wbh)
            else:
                h_candidate = np.tanh(x_time[t] @ wh.T + (reset * h_prev) @ rh.T + wbh + rbh)
            h_new = (1.0 - z) * h_candidate + z * h_prev
            active = _active_mask(sequence_lens, t, batch_size)
            h_state[direction_index] = np.where(active, h_new, h_prev)
            y[t, direction_index] = h_state[direction_index]
    return _from_time_major(y, layout), h_state


# 用 ONNX 官方 LSTM 公式独立计算 Y/Y_h/Y_c，覆盖 peephole 和 input_forget 语义。
def _lstm_reference(
    x,
    w,
    r,
    b=None,
    sequence_lens=None,
    initial_h=None,
    initial_c=None,
    p=None,
    direction="forward",
    layout=0,
    input_forget=0,
):
    x_time = _to_time_major(np.asarray(x, dtype=np.float64), layout)
    seq_len, batch_size = x_time.shape[0], x_time.shape[1]
    num_dirs, hidden = w.shape[0], r.shape[-1]
    bias = np.zeros((num_dirs, 8 * hidden), dtype=np.float64) if b is None else np.asarray(b, dtype=np.float64)
    peepholes = np.zeros((num_dirs, 3 * hidden), dtype=np.float64) if p is None else np.asarray(p, dtype=np.float64)
    h_state = (
        np.zeros((num_dirs, batch_size, hidden), dtype=np.float64)
        if initial_h is None
        else np.asarray(initial_h, dtype=np.float64).copy()
    )
    c_state = (
        np.zeros((num_dirs, batch_size, hidden), dtype=np.float64)
        if initial_c is None
        else np.asarray(initial_c, dtype=np.float64).copy()
    )
    y = np.zeros((seq_len, num_dirs, batch_size, hidden), dtype=np.float64)
    for direction_index in range(num_dirs):
        p_i, p_o, p_f = np.split(peepholes[direction_index], 3)
        bias_sum = np.add(*np.split(bias[direction_index], 2))
        for t in _time_indices(seq_len, direction, direction_index):
            h_prev = h_state[direction_index]
            c_prev = c_state[direction_index]
            gates = x_time[t] @ w[direction_index].T + h_prev @ r[direction_index].T + bias_sum
            i, o, forget, c = np.split(gates, 4, axis=-1)
            i = _sigmoid(i + p_i * c_prev)
            forget = 1.0 - i if input_forget else _sigmoid(forget + p_f * c_prev)
            c_new = forget * c_prev + i * np.tanh(c)
            o = _sigmoid(o + p_o * c_new)
            h_new = o * np.tanh(c_new)
            active = _active_mask(sequence_lens, t, batch_size)
            h_state[direction_index] = np.where(active, h_new, h_prev)
            c_state[direction_index] = np.where(active, c_new, c_prev)
            y[t, direction_index] = h_state[direction_index]
    return _from_time_major(y, layout), h_state, c_state


# 构造 Tensor，避免每个断言重复 dtype、shape 和 data 样板。
def _tensor(data, dtype):
    return Tensor(*data.shape, dtype=dtype, data=data)


# 将 float32 数值转换为 bfloat16 的 uint16 位模式，匹配 Tensor 内部存储。
def _bf16_bits(values):
    data = np.asarray(values, dtype=np.float32)
    bits = data.view(np.uint32)
    lsb = (bits >> 16) & 1
    guard = (bits >> 15) & 1
    sticky = (bits & 0x7FFF) != 0
    rounded = bits + ((guard & (sticky | lsb)).astype(np.uint32) << 16)
    rounded = np.where(np.isnan(data), bits, rounded)
    return (rounded >> 16).astype(np.uint16)


# 将 bfloat16 的 uint16 位模式解码成 float32，便于和独立循环公式比较。
def _bf16_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


# 构造 bfloat16 Tensor，同时返回实际可参与公式计算的解码值。
def _bf16_tensor(values):
    bits = _bf16_bits(values)
    return _tensor(bits, "bfloat16"), _bf16_to_float32(bits)


# 比较 bfloat16 Tensor 的位模式输出和独立公式结果。
def _assert_bf16_tensor_matches(actual, expected):
    assert actual.dtype == "bfloat16"
    assert actual.data.dtype == np.uint16
    np.testing.assert_array_equal(actual.data, _bf16_bits(np.asarray(expected, dtype=np.float32)))


# 验证 C 后端循环算子在混合精度和 sequence_lens 场景下符合独立 ONNX 公式。
def test_c_backend_recurrent_ops_match_independent_onnx_formulas():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.array(
        [
            [[0.5, -0.2], [0.1, 0.4]],
            [[1.0, 0.3], [-0.3, 0.2]],
            [[0.2, 0.7], [0.8, -0.5]],
        ],
        dtype=np.float64,
    )
    sequence_lens = np.array([3, 2], dtype=np.int32)
    rnn_w = np.array([[[0.1, 0.2], [-0.2, 0.3]], [[-0.3, 0.4], [0.2, 0.1]]], dtype=np.float64)
    rnn_r = np.array([[[0.5, 0.1], [0.2, 0.4]], [[0.2, -0.1], [0.3, 0.2]]], dtype=np.float64)
    rnn_b = np.array([[0.1, -0.1, 0.05, 0.02], [0.0, 0.1, -0.03, 0.04]], dtype=np.float64)
    rnn_initial = np.array([[[0.1, 0.0], [0.0, 0.2]], [[-0.1, 0.1], [0.2, -0.2]]], dtype=np.float64)
    expected_y, expected_h = _rnn_reference(
        x, rnn_w, rnn_r, rnn_b, sequence_lens, rnn_initial, direction="bidirectional"
    )
    actual_y, actual_h = RNN(
        ["x", "w", "r", "b", "seq", "init"], ["y", "yh"], hidden_size=2, direction="bidirectional", dtype="float64"
    ).forward(
        _tensor(x, "float64"),
        _tensor(rnn_w, "float64"),
        _tensor(rnn_r, "float64"),
        _tensor(rnn_b, "float64"),
        _tensor(sequence_lens, "int32"),
        _tensor(rnn_initial, "float64"),
    )["tensor"]
    np.testing.assert_allclose(actual_y.data, expected_y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_h.data, expected_h, rtol=1e-12, atol=1e-12)

    gru_x = np.swapaxes(x.astype(np.float32), 0, 1)
    gru_w = np.linspace(-0.4, 0.5, 12, dtype=np.float32).reshape(1, 6, 2)
    gru_r = np.linspace(0.3, -0.2, 12, dtype=np.float32).reshape(1, 6, 2)
    gru_b = np.linspace(-0.2, 0.2, 12, dtype=np.float32).reshape(1, 12)
    gru_initial = np.array([[[0.1, -0.1], [0.2, 0.0]]], dtype=np.float32)
    expected_y, expected_h = _gru_reference(
        gru_x, gru_w, gru_r, gru_b, sequence_lens, gru_initial, direction="reverse", layout=1, linear_before_reset=1
    )
    actual_y, actual_h = GRU(
        ["x", "w", "r", "b", "seq", "init"],
        ["y", "yh"],
        hidden_size=2,
        direction="reverse",
        layout=1,
        linear_before_reset=1,
        dtype="float32",
    ).forward(
        _tensor(gru_x, "float32"),
        _tensor(gru_w, "float32"),
        _tensor(gru_r, "float32"),
        _tensor(gru_b, "float32"),
        _tensor(sequence_lens, "int32"),
        _tensor(gru_initial, "float32"),
    )["tensor"]
    np.testing.assert_allclose(actual_y.data, expected_y.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_h.data, expected_h.astype(np.float32), rtol=1e-6, atol=1e-6)

    lstm_x = x.astype(np.float16)
    lstm_w = np.linspace(-0.3, 0.4, 16, dtype=np.float16).reshape(1, 8, 2)
    lstm_r = np.linspace(0.2, -0.2, 16, dtype=np.float16).reshape(1, 8, 2)
    lstm_b = np.linspace(-0.1, 0.1, 16, dtype=np.float16).reshape(1, 16)
    lstm_initial_h = np.array([[[0.1, 0.0], [-0.1, 0.2]]], dtype=np.float16)
    lstm_initial_c = np.array([[[0.0, 0.2], [0.1, -0.1]]], dtype=np.float16)
    peepholes = np.linspace(-0.05, 0.05, 6, dtype=np.float16).reshape(1, 6)
    expected_y, expected_h, expected_c = _lstm_reference(
        lstm_x,
        lstm_w,
        lstm_r,
        lstm_b,
        sequence_lens,
        lstm_initial_h,
        lstm_initial_c,
        peepholes,
        input_forget=1,
    )
    actual_y, actual_h, actual_c = LSTM(
        ["x", "w", "r", "b", "seq", "h", "c", "p"],
        ["y", "yh", "yc"],
        hidden_size=2,
        input_forget=1,
        dtype="float16",
    ).forward(
        _tensor(lstm_x, "float16"),
        _tensor(lstm_w, "float16"),
        _tensor(lstm_r, "float16"),
        _tensor(lstm_b, "float16"),
        _tensor(sequence_lens, "int32"),
        _tensor(lstm_initial_h, "float16"),
        _tensor(lstm_initial_c, "float16"),
        _tensor(peepholes, "float16"),
    )["tensor"]
    np.testing.assert_allclose(actual_y.data, expected_y.astype(np.float16), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(actual_h.data, expected_h.astype(np.float16), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(actual_c.data, expected_c.astype(np.float16), rtol=1e-3, atol=1e-3)


# 验证 sequence_lens 为 0 的 batch 不执行任何时间步，并保持 initial_h/initial_c 状态。
def test_c_backend_recurrent_zero_sequence_lens_preserve_initial_state():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x = np.array(
        [
            [[0.5, -0.2], [0.1, 0.4]],
            [[1.0, 0.3], [-0.3, 0.2]],
            [[0.2, 0.7], [0.8, -0.5]],
        ],
        dtype=np.float32,
    )
    sequence_lens = np.array([0, 2], dtype=np.int32)

    rnn_w = np.array([[[0.1, 0.2], [-0.2, 0.3]]], dtype=np.float32)
    rnn_r = np.array([[[0.5, 0.1], [0.2, 0.4]]], dtype=np.float32)
    rnn_b = np.array([[0.1, -0.1, 0.05, 0.02]], dtype=np.float32)
    rnn_initial = np.array([[[0.25, -0.5], [0.0, 0.2]]], dtype=np.float32)
    expected_y, expected_h = _rnn_reference(x, rnn_w, rnn_r, rnn_b, sequence_lens, rnn_initial)
    actual_y, actual_h = RNN(
        ["x", "w", "r", "b", "seq", "init"], ["y", "yh"], hidden_size=2, dtype="float32"
    ).forward(
        _tensor(x, "float32"),
        _tensor(rnn_w, "float32"),
        _tensor(rnn_r, "float32"),
        _tensor(rnn_b, "float32"),
        _tensor(sequence_lens, "int32"),
        _tensor(rnn_initial, "float32"),
    )["tensor"]
    np.testing.assert_allclose(actual_y.data, expected_y.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_h.data, expected_h.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        actual_y.data[:, 0, 0, :],
        np.broadcast_to(rnn_initial[0, 0], actual_y.data[:, 0, 0, :].shape),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(actual_h.data[0, 0], rnn_initial[0, 0], rtol=1e-6, atol=1e-6)

    gru_w = np.linspace(-0.4, 0.5, 12, dtype=np.float32).reshape(1, 6, 2)
    gru_r = np.linspace(0.3, -0.2, 12, dtype=np.float32).reshape(1, 6, 2)
    gru_b = np.linspace(-0.2, 0.2, 12, dtype=np.float32).reshape(1, 12)
    gru_initial = np.array([[[0.25, -0.125], [0.1, -0.2]]], dtype=np.float32)
    expected_y, expected_h = _gru_reference(x, gru_w, gru_r, gru_b, sequence_lens, gru_initial, linear_before_reset=1)
    actual_y, actual_h = GRU(
        ["x", "w", "r", "b", "seq", "init"],
        ["y", "yh"],
        hidden_size=2,
        linear_before_reset=1,
        dtype="float32",
    ).forward(
        _tensor(x, "float32"),
        _tensor(gru_w, "float32"),
        _tensor(gru_r, "float32"),
        _tensor(gru_b, "float32"),
        _tensor(sequence_lens, "int32"),
        _tensor(gru_initial, "float32"),
    )["tensor"]
    np.testing.assert_allclose(actual_y.data, expected_y.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_h.data, expected_h.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        actual_y.data[:, 0, 0, :],
        np.broadcast_to(gru_initial[0, 0], actual_y.data[:, 0, 0, :].shape),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(actual_h.data[0, 0], gru_initial[0, 0], rtol=1e-6, atol=1e-6)

    lstm_w = np.linspace(-0.3, 0.4, 16, dtype=np.float32).reshape(1, 8, 2)
    lstm_r = np.linspace(0.2, -0.2, 16, dtype=np.float32).reshape(1, 8, 2)
    lstm_b = np.linspace(-0.1, 0.1, 16, dtype=np.float32).reshape(1, 16)
    lstm_initial_h = np.array([[[0.25, -0.125], [0.1, -0.2]]], dtype=np.float32)
    lstm_initial_c = np.array([[[-0.4, 0.3], [0.05, -0.1]]], dtype=np.float32)
    peepholes = np.linspace(-0.05, 0.05, 6, dtype=np.float32).reshape(1, 6)
    expected_y, expected_h, expected_c = _lstm_reference(
        x,
        lstm_w,
        lstm_r,
        lstm_b,
        sequence_lens,
        lstm_initial_h,
        lstm_initial_c,
        peepholes,
        input_forget=1,
    )
    actual_y, actual_h, actual_c = LSTM(
        ["x", "w", "r", "b", "seq", "h", "c", "p"],
        ["y", "yh", "yc"],
        hidden_size=2,
        input_forget=1,
        dtype="float32",
    ).forward(
        _tensor(x, "float32"),
        _tensor(lstm_w, "float32"),
        _tensor(lstm_r, "float32"),
        _tensor(lstm_b, "float32"),
        _tensor(sequence_lens, "int32"),
        _tensor(lstm_initial_h, "float32"),
        _tensor(lstm_initial_c, "float32"),
        _tensor(peepholes, "float32"),
    )["tensor"]
    np.testing.assert_allclose(actual_y.data, expected_y.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_h.data, expected_h.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_c.data, expected_c.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        actual_y.data[:, 0, 0, :],
        np.broadcast_to(lstm_initial_h[0, 0], actual_y.data[:, 0, 0, :].shape),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(actual_h.data[0, 0], lstm_initial_h[0, 0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_c.data[0, 0], lstm_initial_c[0, 0], rtol=1e-6, atol=1e-6)


# 验证循环算子的 Python fallback 会解码 bfloat16 输入，并把 Y/Y_h/Y_c 按位编码回 bfloat16。
def test_python_recurrent_fallback_bfloat16_decodes_and_encodes_bit_storage(monkeypatch):
    _disable_c_backend(monkeypatch)

    x, x_values = _bf16_tensor(np.array([[[0.5]], [[-0.25]]], dtype=np.float32))
    rnn_w, rnn_w_values = _bf16_tensor(np.array([[[0.25]]], dtype=np.float32))
    rnn_r, rnn_r_values = _bf16_tensor(np.array([[[0.1]]], dtype=np.float32))
    rnn_b, rnn_b_values = _bf16_tensor(np.zeros((1, 2), dtype=np.float32))
    initial_h, initial_h_values = _bf16_tensor(np.array([[[0.125]]], dtype=np.float32))
    expected_y, expected_h = _rnn_reference(x_values, rnn_w_values, rnn_r_values, rnn_b_values, initial_h=initial_h_values)
    actual_y, actual_h = RNN(
        ["x", "w", "r", "b", "", "h"], ["y", "yh"], hidden_size=1, dtype="bfloat16"
    ).forward(x, rnn_w, rnn_r, rnn_b, None, initial_h)["tensor"]
    _assert_bf16_tensor_matches(actual_y, expected_y)
    _assert_bf16_tensor_matches(actual_h, expected_h)

    gru_w, gru_w_values = _bf16_tensor(np.array([[[0.2], [-0.15], [0.3]]], dtype=np.float32))
    gru_r, gru_r_values = _bf16_tensor(np.array([[[0.1], [0.05], [-0.2]]], dtype=np.float32))
    gru_b, gru_b_values = _bf16_tensor(np.zeros((1, 6), dtype=np.float32))
    expected_y, expected_h = _gru_reference(x_values, gru_w_values, gru_r_values, gru_b_values, initial_h=initial_h_values)
    actual_y, actual_h = GRU(
        ["x", "w", "r", "b", "", "h"], ["y", "yh"], hidden_size=1, dtype="bfloat16"
    ).forward(x, gru_w, gru_r, gru_b, None, initial_h)["tensor"]
    _assert_bf16_tensor_matches(actual_y, expected_y)
    _assert_bf16_tensor_matches(actual_h, expected_h)

    lstm_w, lstm_w_values = _bf16_tensor(np.array([[[0.2], [0.1], [-0.15], [0.3]]], dtype=np.float32))
    lstm_r, lstm_r_values = _bf16_tensor(np.array([[[0.05], [-0.1], [0.2], [0.15]]], dtype=np.float32))
    lstm_b, lstm_b_values = _bf16_tensor(np.zeros((1, 8), dtype=np.float32))
    initial_c, initial_c_values = _bf16_tensor(np.array([[[0.05]]], dtype=np.float32))
    peepholes, peephole_values = _bf16_tensor(np.array([[0.01, -0.02, 0.03]], dtype=np.float32))
    expected_y, expected_h, expected_c = _lstm_reference(
        x_values,
        lstm_w_values,
        lstm_r_values,
        lstm_b_values,
        initial_h=initial_h_values,
        initial_c=initial_c_values,
        p=peephole_values,
    )
    actual_y, actual_h, actual_c = LSTM(
        ["x", "w", "r", "b", "", "h", "c", "p"], ["y", "yh", "yc"], hidden_size=1, dtype="bfloat16"
    ).forward(x, lstm_w, lstm_r, lstm_b, None, initial_h, initial_c, peepholes)["tensor"]
    _assert_bf16_tensor_matches(actual_y, expected_y)
    _assert_bf16_tensor_matches(actual_h, expected_h)
    _assert_bf16_tensor_matches(actual_c, expected_c)
