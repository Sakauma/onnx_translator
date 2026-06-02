"""文件功能：兼容旧命令的数值正确性验证入口，实际逻辑位于 tools.numerical 包。
作者：Egor Izmaylov
时间：2026-06-02
"""

from tools.numerical.compare import check_accuracy
from tools.numerical.cuda import CUDA_VERIFY_DIR, run_cuda_ground_truth
from tools.numerical.data import generate_random_data, random_uniform_like_reference
from tools.numerical.dtype import (
    bfloat16_bits_to_float32,
    decode_float8_e4m3,
    decode_float8_e5m2,
    float32_to_bfloat16_bits,
    get_dtype_limits,
    to_float32,
)
from tools.numerical.runner import verify_op
from tools.numerical.cli import main


if __name__ == "__main__":
    main()
