# /**
#   ******************************************************************************
#   * @file        test_operator_spectral_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用独立 FFT 公式验证 DFT、STFT 的 ONNX17 混合精度语义。
#   * @details     2026.06.04  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from operator_test_context import *  # noqa: F401,F403


# 将 float32 数值转换为 bfloat16 的 uint16 位模式，匹配 Tensor 的内部存储方式。
def _bf16_bits(values):
    data = np.asarray(values, dtype=np.float32)
    bits = data.view(np.uint32)
    lsb = (bits >> 16) & 1
    guard = (bits >> 15) & 1
    sticky = (bits & 0x7FFF) != 0
    rounded = bits + ((guard & (sticky | lsb)).astype(np.uint32) << 16)
    rounded = np.where(np.isnan(data), bits, rounded)
    return (rounded >> 16).astype(np.uint16)


# 将 bfloat16 的 uint16 位模式解码为 float32，便于按数值容差比较 C 后端输出。
def _bf16_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


# 从 ONNX 的末尾复数维编码恢复 NumPy complex 数组。
def _to_complex(values, dtype):
    data = _bf16_to_float32(values) if dtype == "bfloat16" else np.asarray(values)
    if data.shape[-1] == 1:
        return np.squeeze(data, axis=-1).astype(np.complex128)
    return data[..., 0].astype(np.complex128) + 1j * data[..., 1].astype(np.complex128)


# 将 NumPy complex 结果编码回 ONNX 末尾复数维格式。
def _from_complex(values, real_only=False):
    if real_only:
        return np.real(values)[..., np.newaxis]
    return np.stack([np.real(values), np.imag(values)], axis=-1)


# 对 bfloat16 输出解码，对其他 dtype 直接返回 NumPy 数组，用于统一断言入口。
def _output_values(tensor):
    if tensor.dtype == "bfloat16":
        return _bf16_to_float32(tensor.data)
    return tensor.data


# 验证 C 后端 DFT 在 float64、float16 和 bfloat16 下符合独立 FFT 公式。
def test_c_backend_dft_mixed_precision_matches_independent_fft():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    complex_signal = np.array([[[1.0, 0.5], [2.0, -1.0], [-1.0, 0.25], [0.5, 0.0]]], dtype=np.float64)
    complex_tensor = Tensor(*complex_signal.shape, dtype="float64", data=complex_signal)
    complex_out = DFT(["x"], ["y"], axis=1, onesided=0, dtype="float64").forward(complex_tensor)["tensor"]
    expected_complex = _from_complex(np.fft.fft(_to_complex(complex_signal, "float64"), n=4, axis=1))
    np.testing.assert_allclose(complex_out.data, expected_complex, rtol=1e-12, atol=1e-12)

    real_signal = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float16)
    real_tensor = Tensor(*real_signal.shape, dtype="float16", data=real_signal)
    real_out = DFT(["x"], ["y"], axis=1, onesided=1, dtype="float16").forward(real_tensor)["tensor"]
    expected_real = _from_complex(np.fft.fft(_to_complex(real_signal, "float16"), n=4, axis=1)[:, :3])
    np.testing.assert_allclose(real_out.data, expected_real.astype(np.float16), rtol=1e-3, atol=1e-3)

    bf16_signal_values = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    bf16_tensor = Tensor(*bf16_signal_values.shape, dtype="bfloat16", data=_bf16_bits(bf16_signal_values))
    bf16_out = DFT(["x"], ["y"], axis=1, onesided=1, dtype="bfloat16").forward(bf16_tensor)["tensor"]
    expected_bf16 = _from_complex(np.fft.fft(bf16_signal_values.squeeze(-1), n=4, axis=1)[:, :3])
    np.testing.assert_allclose(_output_values(bf16_out), expected_bf16.astype(np.float32), rtol=1e-2, atol=1e-2)


# 验证 DFT 在高维输入的中间轴上执行变换时，C 后端坐标映射符合独立 FFT 公式。
def test_c_backend_dft_high_rank_middle_axis_matches_independent_fft():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    signal_values = np.linspace(-2.0, 2.5, 2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4, 1)
    signal = Tensor(*signal_values.shape, dtype="float32", data=signal_values)
    dft_length = Tensor(dtype="int64", data=np.array(5, dtype=np.int64))
    actual = DFT(["x", "dft_length"], ["y"], axis=1, onesided=0, dtype="float32").forward(signal, dft_length)["tensor"]

    expected = _from_complex(np.fft.fft(signal_values.squeeze(-1), n=5, axis=1))
    np.testing.assert_allclose(actual.data, expected.astype(np.float32), rtol=1e-5, atol=1e-5)


# 验证 C 后端 STFT 在 bfloat16 输入、窗口和输出组合下正确读取低精度位模式。
def test_c_backend_stft_bfloat16_matches_independent_fft():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    signal_values = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    window_values = np.array([1.0, 0.5], dtype=np.float32)
    signal = Tensor(*signal_values.shape, dtype="bfloat16", data=_bf16_bits(signal_values))
    window = Tensor(*window_values.shape, dtype="bfloat16", data=_bf16_bits(window_values))
    actual = STFT(["x", "step", "window", "length"], ["y"], onesided=1, dtype="bfloat16").forward(
        signal,
        Tensor(dtype="int64", data=np.array(2, dtype=np.int64)),
        window,
        Tensor(dtype="int64", data=np.array(2, dtype=np.int64)),
    )["tensor"]

    frames = np.stack([signal_values[..., 0:2, :], signal_values[..., 2:4, :]], axis=-3)
    weighted = frames * window_values.reshape((1, 1, 2, 1))
    expected = _from_complex(np.fft.fft(weighted.squeeze(-1), n=2, axis=-1)[..., :2])
    np.testing.assert_allclose(_output_values(actual), expected.astype(np.float32), rtol=1e-2, atol=1e-2)


# 验证 STFT 在多前缀维输入上逐前缀独立切帧，并沿每帧长度维执行 DFT。
def test_c_backend_stft_high_rank_prefix_matches_independent_fft():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    signal_values = np.linspace(-1.5, 2.0, 2 * 2 * 6, dtype=np.float32).reshape(2, 2, 6, 1)
    window_values = np.array([1.0, 0.5, -0.25, 0.75], dtype=np.float32)
    signal = Tensor(*signal_values.shape, dtype="float32", data=signal_values)
    window = Tensor(*window_values.shape, dtype="float32", data=window_values)
    actual = STFT(["x", "step", "window", "length"], ["y"], onesided=1, dtype="float32").forward(
        signal,
        Tensor(dtype="int64", data=np.array(2, dtype=np.int64)),
        window,
        Tensor(dtype="int64", data=np.array(4, dtype=np.int64)),
    )["tensor"]

    frames = np.stack([signal_values[..., 0:4, :], signal_values[..., 2:6, :]], axis=-3)
    weighted = frames * window_values.reshape((1, 1, 1, 4, 1))
    expected = _from_complex(np.fft.fft(weighted.squeeze(-1), n=4, axis=-1)[..., :3])
    np.testing.assert_allclose(actual.data, expected.astype(np.float32), rtol=1e-5, atol=1e-5)
