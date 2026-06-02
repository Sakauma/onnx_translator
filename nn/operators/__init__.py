"""文件功能：聚合分域拆分后的 ONNX 算子实现，并为旧导入路径提供统一导出。
作者：Egor Izmaylov
时间：2026-06-02
"""

from __future__ import annotations

from .common import *

from . import elementwise_basic as _elementwise_basic
from .elementwise_basic import *
from . import elementwise_compare_logic as _elementwise_compare_logic
from .elementwise_compare_logic import *
from . import elementwise_activation_extra as _elementwise_activation_extra
from .elementwise_activation_extra import *
from . import quantization as _quantization
from .quantization import *
from . import conv_ops as _conv_ops
from .conv_ops import *
from . import matrix_ops as _matrix_ops
from .matrix_ops import *
from . import pooling_roi as _pooling_roi
from .pooling_roi import *
from . import shape_transform_ops as _shape_transform_ops
from .shape_transform_ops import *
from . import shape_constant_ops as _shape_constant_ops
from .shape_constant_ops import *
from . import index_scatter_ops as _index_scatter_ops
from .index_scatter_ops import *
from . import shape_extra_ops as _shape_extra_ops
from .shape_extra_ops import *
from . import reduce_arg as _reduce_arg
from .reduce_arg import *
from . import sequence_optional_control as _sequence_optional_control
from .sequence_optional_control import *
from . import normalization_ops as _normalization_ops
from .normalization_ops import *
from . import random_ops as _random_ops
from .random_ops import *
from . import loss_ranking_ops as _loss_ranking_ops
from .loss_ranking_ops import *
from . import spectral_window_ops as _spectral_window_ops
from .spectral_window_ops import *
from . import text as _text
from .text import *
from . import misc as _misc
from .misc import *

_MODULES = [_elementwise_basic, _elementwise_compare_logic, _elementwise_activation_extra, _quantization, _conv_ops, _matrix_ops, _pooling_roi, _shape_transform_ops, _shape_constant_ops, _index_scatter_ops, _shape_extra_ops, _reduce_arg, _sequence_optional_control, _normalization_ops, _random_ops, _loss_ranking_ops, _spectral_window_ops, _text, _misc]
_PUBLIC_SYMBOLS = {name: value for name, value in globals().items() if not name.startswith('_')}
_SHARED_SYMBOLS = {name: value for name, value in globals().items() if not name.startswith('__')}
for _module in _MODULES:
    _module.__dict__.update(_SHARED_SYMBOLS)
__all__ = sorted(_PUBLIC_SYMBOLS)
