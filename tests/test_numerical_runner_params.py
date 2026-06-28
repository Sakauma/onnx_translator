# /**
#   ******************************************************************************
#   * @file        test_numerical_runner_params.py
#   * @author      Egor Izmaylov
#   * @brief       Covers numerical runner parameter helper functions.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import numpy as np

from tools.numerical.runner_params import (
    normalize_slice_parameters,
    onnx_dtype_id_from_name,
    recurrent_params_binary,
    slice_io_values,
)


def test_slice_helpers_expand_negative_axis_and_reverse_step():
    init_args = {
        "starts_value": [-1],
        "ends_value": [-5],
        "axes_value": [-1],
        "steps_value": [-1],
    }

    starts, ends, axes, steps = slice_io_values(init_args, (2, 3, 4))
    full_starts, full_ends, full_steps = normalize_slice_parameters((2, 3, 4), starts, ends, axes, steps)

    assert full_starts == [0, 0, 3]
    assert full_ends == [2, 3, -1]
    assert full_steps == [1, 1, -1]


def test_recurrent_params_binary_encodes_activation_metadata():
    payload = recurrent_params_binary(
        [2, 3],
        {
            "activations": [b"Relu", "Tanh"],
            "activation_alpha": [0.25],
            "activation_beta": [0.5],
            "clip": 1.25,
        },
    )

    ints = np.frombuffer(payload[: (2 + 2 + 6) * 4], dtype=np.int32)
    floats = np.frombuffer(payload[(2 + 2 + 6) * 4 :], dtype=np.float32)

    assert ints.tolist()[:4] == [2, 3, 2, 1]
    assert ints.tolist()[4:6] == [2, 0]
    assert floats[0] == np.float32(0.25)
    assert floats[6] == np.float32(0.5)
    assert floats[-1] == np.float32(1.25)


def test_onnx_dtype_id_from_name_rejects_unknown_dtype():
    assert onnx_dtype_id_from_name("float32") == 1
    try:
        onnx_dtype_id_from_name("float8_e4m3")
    except ValueError as exc:
        assert "Unsupported ONNX output_datatype" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported window dtype")
