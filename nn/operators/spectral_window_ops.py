"""文件功能：保存 `spectral_window_ops` 分组中的 ONNX 算子实现。
作者：Egor Izmaylov
时间：2026-06-02
"""

from .common import *

class MelWeightMatrix(Ops):
    # 初始化 `MelWeightMatrix` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, output_datatype=1, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = nn.onnx_dtype_mapping.get(output_datatype, "float32") if isinstance(output_datatype, int) else output_datatype
        self.version = version
        if self.lib:
            self.lib.mel_weight_matrix_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
            ]

    # 封装 `_scalar` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _scalar(value):
        return np.asarray(value.data).item()

    # 封装 `_mel` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _mel(frequency):
        return 2595.0 * np.log10(1.0 + frequency / 700.0)

    # 封装 `_hz` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _hz(mel):
        return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)

    # 执行 `MelWeightMatrix` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz):
        bins = int(self._scalar(num_mel_bins))
        dft_len = int(self._scalar(dft_length))
        rate = int(self._scalar(sample_rate))
        lower = float(self._scalar(lower_edge_hertz))
        upper = float(self._scalar(upper_edge_hertz))
        if bins < 0 or dft_len < 0 or rate <= 0 or upper < lower:
            raise ValueError("Invalid MelWeightMatrix parameters")

        num_spectrogram_bins = dft_len // 2 + 1
        if self.lib is not None and self.dtype in nn.DTYPE_MAP:
            output_shape_c = (ctypes.c_int * 2)(num_spectrogram_bins, bins)
            output_c = self.lib.create_tensor(output_shape_c, 2, nn.DTYPE_MAP[self.dtype])
            scalar_ctensors = [
                self._numpy_to_ctensor(np.ascontiguousarray(np.asarray(tensor.data).reshape(())), tensor.dtype)
                for tensor in (num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)
            ]
            self.lib.mel_weight_matrix_forward(*scalar_ctensors, output_c)
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            for c_tensor in scalar_ctensors:
                self.lib.free_tensor(c_tensor)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

        mel_points = np.linspace(self._mel(lower), self._mel(upper), bins + 2)
        hz_points = self._hz(mel_points)
        frequency_bins = np.floor((dft_len + 1) * hz_points / rate).astype(int)

        output = np.zeros((num_spectrogram_bins, bins), dtype=np.float64)
        for i in range(bins):
            left, center, right = frequency_bins[i], frequency_bins[i + 1], frequency_bins[i + 2]
            left = max(left, 0)
            center = min(max(center, 0), num_spectrogram_bins - 1)
            right = min(max(right, 0), num_spectrogram_bins)
            if center == left:
                output[center, i] = 1.0
            else:
                for j in range(left, center + 1):
                    output[j, i] = float(j - left) / float(center - left)
            if right > center:
                for j in range(center, right):
                    output[j, i] = float(right - j) / float(right - center)

        out_dtype = nn.DTYPE_TO_NUMPY.get(self.dtype, np.float32)
        out_data = output.astype(out_dtype)
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `MelWeightMatrix` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz):
        if isinstance(num_mel_bins, Tensor) and isinstance(dft_length, Tensor):
            out_shape = (int(num_mel_bins.data.item()) // 1, int(dft_length.data.item()) // 2 + 1)
            out_shape = (out_shape[1], out_shape[0])
        else:
            out_shape = (1, 1)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}


class DFT(Ops):
    # 初始化 `DFT` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=1, inverse=0, onesided=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.inverse = bool(inverse)
        self.onesided = bool(onesided)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.dft_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]

    # 封装 `_optional_length` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _optional_length(dft_length, default):
        if dft_length is None:
            return int(default)
        return int(np.asarray(dft_length.data).item())

    # 封装 `_as_complex` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _as_complex(data):
        if data.shape[-1] == 1:
            return np.squeeze(data, axis=-1).astype(np.complex128)
        if data.shape[-1] == 2:
            return data[..., 0].astype(np.complex128) + 1j * data[..., 1].astype(np.complex128)
        raise ValueError(f"DFT expects the last dimension to be 1 or 2, got {data.shape[-1]}")

    # 封装 `_from_complex` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _from_complex(data, dtype, real_only=False):
        if real_only:
            out = np.real(data)[..., np.newaxis]
        else:
            out = np.stack([np.real(data), np.imag(data)], axis=-1)
        return out.astype(nn.DTYPE_TO_NUMPY.get(dtype, np.float32), copy=False)

    # 封装 `_output_shape` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _output_shape(self, input_shape, dft_length=None):
        if len(input_shape) < 1:
            raise ValueError("DFT expects rank >= 1")
        if input_shape[-1] not in (1, 2):
            raise ValueError(f"DFT expects the last dimension to be 1 or 2, got {input_shape[-1]}")
        axis = self.axis % len(input_shape)
        if axis == len(input_shape) - 1:
            raise ValueError("DFT axis cannot refer to the trailing complex component dimension")
        default_len = 2 * (input_shape[axis] - 1) if self.inverse and self.onesided else input_shape[axis]
        fft_len = int(default_len if dft_length is None else dft_length)
        out_shape = list(input_shape)
        if self.inverse:
            out_shape[axis] = fft_len
            out_shape[-1] = 1 if self.onesided else 2
        else:
            out_shape[axis] = fft_len // 2 + 1 if self.onesided else fft_len
            out_shape[-1] = 2
        return tuple(out_shape)

    # 执行 `DFT` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input, dft_length=None):
        data = np.asarray(input.data)
        axis = self.axis % data.ndim
        if axis == data.ndim - 1:
            raise ValueError("DFT axis cannot refer to the trailing complex component dimension")
        default_len = 2 * (data.shape[axis] - 1) if self.inverse and self.onesided else data.shape[axis]
        fft_len = self._optional_length(dft_length, default_len)
        out_shape = self._output_shape(data.shape, fft_len)
        if self.lib is not None and input.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.dft_forward(
                input_c,
                output_c,
                ctypes.c_int(self.axis),
                ctypes.c_int(int(self.inverse)),
                ctypes.c_int(int(self.onesided)),
                ctypes.c_int(fft_len),
            )
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        complex_data = self._as_complex(data)
        if self.inverse:
            if self.onesided:
                transformed = np.fft.irfft(complex_data, n=fft_len, axis=axis)
                out_data = self._from_complex(transformed, self.dtype, real_only=True)
            else:
                transformed = np.fft.ifft(complex_data, n=fft_len, axis=axis)
                out_data = self._from_complex(transformed, self.dtype)
        else:
            transformed = np.fft.fft(complex_data, n=fft_len, axis=axis)
            out_data = self._from_complex(transformed, self.dtype)
            if self.onesided:
                slices = [slice(None)] * out_data.ndim
                slices[axis] = slice(0, fft_len // 2 + 1)
                out_data = out_data[tuple(slices)]
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `DFT` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input, dft_length=None):
        length = None
        if isinstance(dft_length, Tensor):
            length = int(dft_length.data.item())
        return {"tensor": Tensor_(*self._output_shape(input.size, length), dtype=self.dtype), "parameters": None}


class STFT(Ops):
    # 初始化 `STFT` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, onesided=1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.onesided = bool(onesided)
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.stft_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]

    # 封装 `_scalar` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _scalar(value):
        return int(np.asarray(value.data).item())

    # 封装 `_frame_length` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _frame_length(signal, window=None, frame_length=None):
        if frame_length is not None:
            return STFT._scalar(frame_length)
        if window is not None:
            return int(window.data.shape[0])
        return int(signal.data.shape[-2])

    # 执行 `STFT` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, signal, frame_step, window=None, frame_length=None):
        data = np.asarray(signal.data)
        if data.ndim < 2 or data.shape[-1] not in (1, 2):
            raise ValueError(f"STFT expects input shape [..., signal_length, 1|2], got {signal.size}")
        step = self._scalar(frame_step)
        length = self._frame_length(signal, window, frame_length)
        if step <= 0 or length <= 0:
            raise ValueError("STFT frame_step and frame_length must be positive")
        n_frames = 1 + (data.shape[-2] - length) // step
        if n_frames < 0:
            raise ValueError("STFT frame_length cannot exceed signal length")
        bins = length // 2 + 1 if self.onesided else length
        out_shape = tuple(data.shape[:-2]) + (n_frames, bins, 2)
        if self.lib is not None and signal.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            signal_c = self._numpy_to_ctensor(np.ascontiguousarray(signal.data), signal.dtype)
            window_c = None
            if window is not None:
                if window.dtype not in nn.DTYPE_MAP:
                    self.lib.free_tensor(signal_c)
                    window_c = None
                else:
                    window_c = self._numpy_to_ctensor(np.ascontiguousarray(window.data), window.dtype)
            if window is None or window_c is not None:
                output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
                output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
                self.lib.stft_forward(
                    signal_c,
                    window_c,
                    output_c,
                    ctypes.c_int(step),
                    ctypes.c_int(length),
                    ctypes.c_int(int(self.onesided)),
                )
                out_data = self._ctensor_to_numpy(output_c, self.dtype)
                self.lib.free_tensor(signal_c)
                if window_c is not None:
                    self.lib.free_tensor(window_c)
                self.lib.free_tensor(output_c)
                return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        frame_list = []
        for frame_idx in range(n_frames):
            start = frame_idx * step
            stop = start + length
            frame = data[..., start:stop, :]
            if frame.shape[-2] < length:
                pad_width = [(0, 0)] * frame.ndim
                pad_width[-2] = (0, length - frame.shape[-2])
                frame = np.pad(frame, pad_width, mode="constant")
            frame_list.append(frame)

        frames = np.stack(frame_list, axis=-3)
        if window is None:
            win = np.ones((length,), dtype=data.dtype)
        else:
            win = np.asarray(window.data, dtype=data.dtype)
        window_shape = (1,) * (frames.ndim - 2) + (length, 1)
        weighted = frames * win.reshape(window_shape)
        out_data = DFT([], [], axis=-2, onesided=int(self.onesided), dtype=self.dtype).forward(
            Tensor(*weighted.shape, dtype=signal.dtype, data=weighted)
        )["tensor"].data
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `STFT` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, signal, frame_step, window=None, frame_length=None):
        step = int(frame_step.data.item()) if isinstance(frame_step, Tensor) else 1
        if isinstance(frame_length, Tensor):
            length = int(frame_length.data.item())
        elif isinstance(window, Tensor):
            length = int(window.data.shape[0])
        elif hasattr(window, "size") and window is not None:
            length = int(window.size[0])
        else:
            length = int(signal.size[-2])
        n_frames = 1 + (signal.size[-2] - length) // step
        bins = length // 2 + 1 if self.onesided else length
        out_shape = tuple(signal.size[:-2]) + (n_frames, bins, 2)
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}


class HannWindow(Ops):
    # 初始化 `HannWindow` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, periodic=1, output_datatype=1, version="17"):
        super().__init__(inputs, outputs)
        self.periodic = periodic
        self.dtype = nn.onnx_dtype_mapping.get(output_datatype, "float32")
        self.version = version
        if self.lib:
            self.lib.hann_window_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]

    # 执行 `HannWindow` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, size):
        out_data = _window_values_c_first(self, size, "hann_window_forward", "hann")
        out_shape = out_data.shape
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `HannWindow` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, size):
        return {"tensor": Tensor_(*_window_output_shape(size), dtype=self.dtype), "parameters": None}


class HammingWindow(Ops):
    # 初始化 `HammingWindow` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, periodic=1, output_datatype=1, version="17"):
        super().__init__(inputs, outputs)
        self.periodic = periodic
        self.dtype = nn.onnx_dtype_mapping.get(output_datatype, "float32")
        self.version = version
        if self.lib:
            self.lib.hamming_window_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]

    # 执行 `HammingWindow` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, size):
        out_data = _window_values_c_first(self, size, "hamming_window_forward", "hamming")
        out_shape = out_data.shape
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `HammingWindow` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, size):
        return {"tensor": Tensor_(*_window_output_shape(size), dtype=self.dtype), "parameters": None}


class BlackmanWindow(Ops):
    # 初始化 `BlackmanWindow` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, periodic=1, output_datatype=1, version="17"):
        super().__init__(inputs, outputs)
        self.periodic = periodic
        self.dtype = nn.onnx_dtype_mapping.get(output_datatype, "float32")
        self.version = version
        if self.lib:
            self.lib.blackman_window_forward.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int]

    # 执行 `BlackmanWindow` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, size):
        out_data = _window_values_c_first(self, size, "blackman_window_forward", "blackman")
        out_shape = out_data.shape
        return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `BlackmanWindow` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, size):
        return {"tensor": Tensor_(*_window_output_shape(size), dtype=self.dtype), "parameters": None}
