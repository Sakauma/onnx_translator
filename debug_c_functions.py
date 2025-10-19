import ctypes
import numpy as np

# 加载库
lib = ctypes.CDLL('./libtensor_ops.so')

# 定义C函数参数类型
lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
lib.create_tensor.restype = ctypes.c_void_p

lib.free_tensor.argtypes = [ctypes.c_void_p]
lib.add_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

print("✅ 库加载成功")

# 创建测试张量
shape = (ctypes.c_int * 1)(4)
tensor1_ptr = lib.create_tensor(shape, 1, 1)  # 1 对应 float32
tensor2_ptr = lib.create_tensor(shape, 1, 1)
output_tensor_ptr = lib.create_tensor(shape, 1, 1)

print("✅ 张量创建成功")


# Tensor结构体定义
class TensorStruct(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.c_void_p),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('ndim', ctypes.c_int),
        ('size', ctypes.c_size_t),
        ('dtype', ctypes.c_int)
    ]


# 将指针转换为Tensor结构体
tensor1 = ctypes.cast(tensor1_ptr, ctypes.POINTER(TensorStruct)).contents
tensor2 = ctypes.cast(tensor2_ptr, ctypes.POINTER(TensorStruct)).contents
output_tensor = ctypes.cast(output_tensor_ptr, ctypes.POINTER(TensorStruct)).contents

print(f"DEBUG: 张量1信息 - data: {tensor1.data}, size: {tensor1.size}")
print(f"DEBUG: 张量2信息 - data: {tensor2.data}, size: {tensor2.size}")

# 创建测试数据
data1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
data2 = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

print(f"DEBUG: 输入数据1: {data1}")
print(f"DEBUG: 输入数据2: {data2}")

# 方法1：直接通过指针设置数据
if tensor1.data:
    # 将void指针转换为float指针
    float_array1 = (ctypes.c_float * 4).from_address(tensor1.data)
    for i in range(4):
        float_array1[i] = data1[i]
    print("✅ 数据1设置完成")

if tensor2.data:
    float_array2 = (ctypes.c_float * 4).from_address(tensor2.data)
    for i in range(4):
        float_array2[i] = data2[i]
    print("✅ 数据2设置完成")

# 验证数据是否正确设置
print("DEBUG: 验证设置的数据:")
if tensor1.data:
    float_array1_check = (ctypes.c_float * 4).from_address(tensor1.data)
    check_data1 = [float_array1_check[i] for i in range(4)]
    print(f"  张量1数据: {check_data1}")

if tensor2.data:
    float_array2_check = (ctypes.c_float * 4).from_address(tensor2.data)
    check_data2 = [float_array2_check[i] for i in range(4)]
    print(f"  张量2数据: {check_data2}")

# 调用add_forward
try:
    lib.add_forward(tensor1_ptr, tensor2_ptr, output_tensor_ptr)
    print("✅ add_forward 调用成功")

    # 读取结果
    if output_tensor.data:
        float_array_output = (ctypes.c_float * 4).from_address(output_tensor.data)
        result_data = np.array([float_array_output[i] for i in range(4)], dtype=np.float32)
        print(f"✅ 计算结果: {result_data}")
        print(f"✅ 期望结果: {data1 + data2}")
        print(f"✅ 结果正确: {np.allclose(result_data, data1 + data2)}")
    else:
        print("❌ 输出张量数据指针为空")

except Exception as e:
    print(f"❌ add_forward 调用失败: {e}")
    import traceback

    traceback.print_exc()

# 清理
lib.free_tensor(tensor1_ptr)
lib.free_tensor(tensor2_ptr)
lib.free_tensor(output_tensor_ptr)