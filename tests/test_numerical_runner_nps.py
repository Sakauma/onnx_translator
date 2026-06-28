# /**
#   ******************************************************************************
#   * @file        test_numerical_runner_nps.py
#   * @author      Egor Izmaylov
#   * @brief       Covers NPS forward dispatch used by numerical runner plans.
#   * @details     2026.06.28  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import numpy as np

from nn import Tensor
from tools.numerical.runner_nps import run_nps_forward


class _FakeTensor:
    def __init__(self, data):
        self.data = data


class _FakeGenericOp:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def forward(self, *tensors):
        assert "shape_value" not in self.kwargs
        return {"tensor": _FakeTensor(tensors[0].data + 1)}


class _FakeTopKOp:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def forward(self, x, k):
        values = np.asarray([[3.0, 2.0]], dtype=np.float32)
        indices = np.asarray([[1, 0]], dtype=np.int64)
        return {"tensor": [_FakeTensor(values), _FakeTensor(indices)]}


def test_run_nps_forward_strips_plan_only_init_args_for_generic_op():
    tensor = Tensor(2, dtype="float32", data=np.asarray([1.0, 2.0], dtype=np.float32))

    result = run_nps_forward(
        _FakeGenericOp,
        "relu",
        [tensor],
        {"shape_value": [2], "custom_attr": 3},
        "float32",
    )

    np.testing.assert_array_equal(result.output, np.asarray([2.0, 3.0], dtype=np.float32))
    assert result.topk_indices is None


def test_run_nps_forward_returns_topk_indices_side_output():
    x = Tensor(1, 3, dtype="float32", data=np.asarray([[1.0, 3.0, 2.0]], dtype=np.float32))
    k = Tensor(1, dtype="int64", data=np.asarray([2], dtype=np.int64))

    result = run_nps_forward(_FakeTopKOp, "topk", [x, k], {"k_value": 2}, "float32")

    np.testing.assert_array_equal(result.output, np.asarray([[3.0, 2.0]], dtype=np.float32))
    np.testing.assert_array_equal(result.topk_indices, np.asarray([[1, 0]], dtype=np.int64))
