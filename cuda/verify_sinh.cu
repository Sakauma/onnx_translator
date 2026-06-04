/**
  ******************************************************************************
  * @file        verify_sinh.cu
  * @author      Egor Izmaylov
  * @brief       提供 sinh 算子的 CUDA 参考验证程序，供数值正确性脚本与 C 后端结果对比。
  * @details     2026.06.05  V1.0.0  创建
  ******************************************************************************
  * @attention
  ******************************************************************************
*/

#define VERIFY_OP_NAME sinh
#define VERIFY_EXPR(x) sinhf(x)
#include "verify_unary_math.cuh"
