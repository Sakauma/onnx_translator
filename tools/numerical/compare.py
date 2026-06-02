# /**
#   ******************************************************************************
#   * @file        compare.py
#   * @author      Egor Izmaylov
#   * @brief       比较 C 后端输出与 CUDA 参考输出，生成误差和失败掩码。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import numpy as np

from .dtype import get_dtype_limits, to_float32


def check_accuracy(nps_val, cuda_val, atol, rtol, dtype):
    """
    严谨的验证逻辑：支持数值对比、溢出判定和 NaN 匹配
    """
    min_limit, max_limit, is_saturating = get_dtype_limits(dtype)
    nan_match = np.isnan(nps_val) & np.isnan(cuda_val)
    inf_match = np.isinf(nps_val) & np.isinf(cuda_val) & (np.sign(nps_val) == np.sign(cuda_val))
    
    cuda_finite = np.isfinite(cuda_val)
    gt_overflow_pos = cuda_finite & (cuda_val > max_limit)
    gt_overflow_neg = cuda_finite & (cuda_val < min_limit)
    
    if is_saturating:
        overflow_pos_match = gt_overflow_pos & (nps_val == max_limit)
        overflow_neg_match = gt_overflow_neg & (nps_val == min_limit)
    else:
        overflow_pos_match = gt_overflow_pos & (nps_val == np.inf)
        overflow_neg_match = gt_overflow_neg & (nps_val == -np.inf)
        
    logic_pass = nan_match | inf_match | overflow_pos_match | overflow_neg_match
    valid_numeric_mask = np.isfinite(nps_val) & np.isfinite(cuda_val)
    current_max_abs = 0.0
    current_max_rel = 0.0
    numeric_pass_mask = np.zeros_like(nps_val, dtype=bool)
    
    if np.any(valid_numeric_mask):
        # 提取数值
        v_nps = nps_val[valid_numeric_mask]
        v_cuda = cuda_val[valid_numeric_mask]
        
        # 计算误差
        diff = np.abs(v_nps - v_cuda)
        ref = np.abs(v_cuda) + 1e-12 # 防止除零
        rel = diff / ref
        
        current_max_abs = np.max(diff)
        current_max_rel = np.max(rel)
        
        tolerance = atol + rtol * np.abs(v_cuda)
        is_close = diff <= tolerance
        
        numeric_pass_mask[valid_numeric_mask] = is_close

    final_pass = logic_pass | numeric_pass_mask
    fail_mask = ~final_pass
    
    if np.all(final_pass):
        if not np.any(valid_numeric_mask):
            print(f"     ⚠️  Warning: Pass but all values were NaN/Inf/Overflow matched.")
        return True, current_max_abs, current_max_rel, None
    else:
        numeric_fail = fail_mask & valid_numeric_mask
        
        fail_abs = -1.0
        fail_rel = -1.0
        
        if np.any(numeric_fail):
            diff = np.abs(nps_val[numeric_fail] - cuda_val[numeric_fail])
            fail_abs = np.max(diff)
            ref = np.abs(cuda_val[numeric_fail]) + 1e-12
            fail_rel = np.max(diff / ref)
        elif np.any(fail_mask):
            fail_abs = -999.0
            fail_rel = -999.0
            
        return False, fail_abs, fail_rel, fail_mask
