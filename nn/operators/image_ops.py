# /**
#   ******************************************************************************
#   * @file        image_ops.py
#   * @author      Egor Izmaylov
#   * @brief       保存图像输入输出相关 ONNX 算子实现。
#   * @details     2026.06.05  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *
import io


class ImageDecoder(Ops):
    # 初始化 `ImageDecoder` 的 pixel_format 属性，输出固定为 uint8 图像张量。
    def __init__(self, inputs, outputs, pixel_format="RGB", version="20"):
        super().__init__(inputs, outputs)
        self.pixel_format = pixel_format or "RGB"
        self.dtype = "uint8"
        self.version = version

    # 执行图像字节流解码，按 ONNX 官方 pixel_format 约定输出 channel-last uint8 张量。
    def forward(self, encoded_stream):
        try:
            import PIL.Image
        except ImportError as exc:
            raise ImportError("Pillow must be installed to use ImageDecoder") from exc

        encoded = np.asarray(encoded_stream.data, dtype=np.uint8)
        try:
            image = PIL.Image.open(io.BytesIO(encoded.tobytes()))
            if self.pixel_format == "BGR":
                decoded = np.asarray(image, dtype=np.uint8)[:, :, ::-1]
            elif self.pixel_format == "RGB":
                decoded = np.asarray(image, dtype=np.uint8)
            elif self.pixel_format == "Grayscale":
                decoded = np.asarray(image.convert("L"), dtype=np.uint8)
                decoded = np.expand_dims(decoded, axis=2)
            else:
                raise ValueError(f"Unsupported ImageDecoder pixel_format {self.pixel_format!r}")
        except ValueError:
            raise
        except Exception:
            decoded = np.empty((0, 0, 0), dtype=np.uint8)

        return {"tensor": Tensor(*decoded.shape, dtype="uint8", data=decoded), "parameters": None, "graph": None}

    # 执行形状推断路径；若已有真实字节流则返回精确 shape，否则返回空图像占位 shape。
    def forward_(self, encoded_stream):
        if isinstance(encoded_stream, Tensor):
            result = self.forward(encoded_stream)["tensor"]
            return {"tensor": Tensor_(*result.size, dtype="uint8"), "parameters": None, "graph": None}
        return {"tensor": Tensor_(0, 0, 0, dtype="uint8"), "parameters": None, "graph": None}
