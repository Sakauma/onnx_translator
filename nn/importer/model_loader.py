# /**
#   ******************************************************************************
#   * @file        model_loader.py
#   * @author      Egor Izmaylov
#   * @brief       统一从模型所在目录解析 ONNX 与相邻 external data。
#   * @details     2026.09.10  V1.0.0  创建
#   ******************************************************************************
# */

from pathlib import Path

import onnx


def load_model(file_path, *, load_external_data=True):
    """使用模型父目录作为 ONNX 官方 external-data 解析基准。"""
    model_path = Path(file_path).expanduser().resolve()
    return onnx.load_model(str(model_path), load_external_data=load_external_data)
