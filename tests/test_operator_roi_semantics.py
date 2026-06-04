# /**
#   ******************************************************************************
#   * @file        test_operator_roi_semantics.py
#   * @author      Egor Izmaylov
#   * @brief       使用独立 ROI 公式验证 MaxRoiPool、RoiAlign 的 ONNX17 混合精度语义。
#   * @details     2026.06.04  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from operator_test_context import *  # noqa: F401,F403


# 计算 MaxRoiPool 的独立参考结果，按 ROI 坐标缩放、量化和池化分箱取最大值。
def _max_roi_pool_reference(x, rois, pooled_shape, spatial_scale=1.0):
    data = np.asarray(x, dtype=np.float64)
    roi_data = np.asarray(rois, dtype=np.float64)
    pooled_h, pooled_w = pooled_shape
    num_rois, channels = roi_data.shape[0], data.shape[1]
    height, width = data.shape[2], data.shape[3]
    out = np.zeros((num_rois, channels, pooled_h, pooled_w), dtype=np.float64)
    for roi_idx, roi in enumerate(roi_data):
        batch = int(roi[0])
        x1 = int(np.rint(roi[1] * spatial_scale))
        y1 = int(np.rint(roi[2] * spatial_scale))
        x2 = int(np.rint(roi[3] * spatial_scale))
        y2 = int(np.rint(roi[4] * spatial_scale))
        roi_w = max(x2 - x1 + 1, 1)
        roi_h = max(y2 - y1 + 1, 1)
        bin_h = float(roi_h) / float(pooled_h)
        bin_w = float(roi_w) / float(pooled_w)
        for ph in range(pooled_h):
            for pw in range(pooled_w):
                hstart = min(max(int(np.floor(ph * bin_h)) + y1, 0), height)
                hend = min(max(int(np.ceil((ph + 1) * bin_h)) + y1, 0), height)
                wstart = min(max(int(np.floor(pw * bin_w)) + x1, 0), width)
                wend = min(max(int(np.ceil((pw + 1) * bin_w)) + x1, 0), width)
                if hend <= hstart or wend <= wstart:
                    out[roi_idx, :, ph, pw] = 0.0
                else:
                    out[roi_idx, :, ph, pw] = np.max(data[batch, :, hstart:hend, wstart:wend], axis=(1, 2))
    return out


# 按 ONNX RoiAlign 参考实现中的边界规则计算一个采样点的四邻点权重和贡献。
def _roi_align_terms(image, y, x):
    height, width = image.shape
    if y < -1.0 or y > height or x < -1.0 or x > width:
        return [0.0, 0.0, 0.0, 0.0]
    y = max(y, 0.0)
    x = max(x, 0.0)
    y_low = int(y)
    x_low = int(x)
    if y_low >= height - 1:
        y_high = y_low = height - 1
        y = float(y_low)
    else:
        y_high = y_low + 1
    if x_low >= width - 1:
        x_high = x_low = width - 1
        x = float(x_low)
    else:
        x_high = x_low + 1
    ly = y - float(y_low)
    lx = x - float(x_low)
    hy = 1.0 - ly
    hx = 1.0 - lx
    return [
        image[y_low, x_low] * hy * hx,
        image[y_low, x_high] * hy * lx,
        image[y_high, x_low] * ly * hx,
        image[y_high, x_high] * ly * lx,
    ]


# 计算 RoiAlign 的独立参考结果，覆盖 avg/max、sampling_ratio 和坐标变换模式。
def _roi_align_reference(
    x,
    rois,
    batch_indices,
    output_height,
    output_width,
    spatial_scale=1.0,
    sampling_ratio=0,
    mode="avg",
    coordinate_transformation_mode="half_pixel",
):
    data = np.asarray(x, dtype=np.float64)
    roi_data = np.asarray(rois, dtype=np.float64)
    batches = np.asarray(batch_indices, dtype=np.int64).reshape(-1)
    num_rois, channels = roi_data.shape[0], data.shape[1]
    out = np.empty((num_rois, channels, output_height, output_width), dtype=np.float64)
    half_pixel = coordinate_transformation_mode == "half_pixel"
    offset = 0.5 if half_pixel else 0.0
    for roi_idx, roi in enumerate(roi_data):
        roi_start_w = roi[0] * spatial_scale - offset
        roi_start_h = roi[1] * spatial_scale - offset
        roi_end_w = roi[2] * spatial_scale - offset
        roi_end_h = roi[3] * spatial_scale - offset
        roi_w = roi_end_w - roi_start_w
        roi_h = roi_end_h - roi_start_h
        if not half_pixel:
            roi_w = max(roi_w, 1.0)
            roi_h = max(roi_h, 1.0)
        bin_h = roi_h / output_height
        bin_w = roi_w / output_width
        grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(roi_h / output_height))
        grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(roi_w / output_width))
        grid_h = max(grid_h, 1)
        grid_w = max(grid_w, 1)
        count = grid_h * grid_w
        for c in range(channels):
            image = data[int(batches[roi_idx]), c]
            for ph in range(output_height):
                for pw in range(output_width):
                    values = []
                    for iy in range(grid_h):
                        yy = roi_start_h + ph * bin_h + (iy + 0.5) * bin_h / grid_h
                        for ix in range(grid_w):
                            xx = roi_start_w + pw * bin_w + (ix + 0.5) * bin_w / grid_w
                            terms = _roi_align_terms(image, yy, xx)
                            values.append(max(terms) if mode == "max" else sum(terms))
                    out[roi_idx, c, ph, pw] = max(values) if mode == "max" else sum(values) / count
    return out


# 构造 Tensor，避免每个断言重复 dtype、shape 和 data 样板。
def _tensor(data, dtype):
    return Tensor(*data.shape, dtype=dtype, data=data)


# 验证 MaxRoiPool 的 C 后端在 float16/float64 输入下符合 ROI 分箱最大值公式。
def test_c_backend_max_roi_pool_mixed_precision_matches_independent_formula():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x64 = (np.arange(2 * 2 * 5 * 5, dtype=np.float64).reshape(2, 2, 5, 5) / 8.0) - 3.0
    rois64 = np.array([[0, 0.0, 0.0, 4.0, 4.0], [1, 1.0, 1.0, 3.0, 4.0]], dtype=np.float64)
    expected64 = _max_roi_pool_reference(x64, rois64, pooled_shape=(2, 3), spatial_scale=1.0)
    actual64 = MaxRoiPool(
        ["x", "rois"], ["y"], pooled_shape=[2, 3], spatial_scale=1.0, dtype="float64"
    ).forward(_tensor(x64, "float64"), _tensor(rois64, "float64"))["tensor"]
    np.testing.assert_allclose(actual64.data, expected64, rtol=1e-12, atol=1e-12)

    x16 = x64.astype(np.float16)
    rois16 = rois64.astype(np.float16)
    expected16 = _max_roi_pool_reference(x16, rois16, pooled_shape=(2, 3), spatial_scale=1.0).astype(np.float16)
    actual16 = MaxRoiPool(
        ["x", "rois"], ["y"], pooled_shape=[2, 3], spatial_scale=1.0, dtype="float16"
    ).forward(_tensor(x16, "float16"), _tensor(rois16, "float16"))["tensor"]
    np.testing.assert_allclose(actual16.data, expected16, rtol=1e-3, atol=1e-3)


# 验证 RoiAlign 的 C 后端在 float16/float64 输入下符合 ONNX 双线性采样公式。
def test_c_backend_roi_align_mixed_precision_matches_independent_formula():
    if not os.path.exists(nn.TENSOR_OPS_LIB_PATH):
        pytest.skip("C backend library is not built")

    x64 = (np.arange(2 * 1 * 4 * 5, dtype=np.float64).reshape(2, 1, 4, 5) / 7.0) - 2.0
    rois64 = np.array([[0.2, 0.1, 3.8, 3.0], [0.5, 0.4, 4.0, 2.6]], dtype=np.float64)
    batch_indices = np.array([0, 1], dtype=np.int64)
    expected64 = _roi_align_reference(
        x64,
        rois64,
        batch_indices,
        output_height=2,
        output_width=3,
        sampling_ratio=0,
        mode="avg",
        coordinate_transformation_mode="half_pixel",
    )
    actual64 = RoiAlign(
        ["x", "rois", "batch"],
        ["y"],
        output_height=2,
        output_width=3,
        sampling_ratio=0,
        mode="avg",
        coordinate_transformation_mode="half_pixel",
        dtype="float64",
    ).forward(_tensor(x64, "float64"), _tensor(rois64, "float64"), _tensor(batch_indices, "int64"))["tensor"]
    np.testing.assert_allclose(actual64.data, expected64, rtol=1e-12, atol=1e-12)

    x16 = x64.astype(np.float16)
    rois16 = rois64.astype(np.float16)
    expected16 = _roi_align_reference(
        x16,
        rois16,
        batch_indices,
        output_height=1,
        output_width=2,
        sampling_ratio=2,
        mode="max",
        coordinate_transformation_mode="output_half_pixel",
    ).astype(np.float16)
    actual16 = RoiAlign(
        ["x", "rois", "batch"],
        ["y"],
        output_height=1,
        output_width=2,
        sampling_ratio=2,
        mode="max",
        coordinate_transformation_mode="output_half_pixel",
        dtype="float16",
    ).forward(_tensor(x16, "float16"), _tensor(rois16, "float16"), _tensor(batch_indices, "int64"))["tensor"]
    np.testing.assert_allclose(actual16.data, expected16, rtol=1e-3, atol=1e-3)
