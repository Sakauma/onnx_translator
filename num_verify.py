import numpy as np
from nn import Tensor
from nn.Operators import RESHAPE  # 假设存在RESHAPE算子类


def run_reshape_verification(op_name="reshape", nps_operator_class=RESHAPE,
                             input_shape=(1, 3, 128, 128), target_shape=(1, 3, 64, 256),
                             dtype="float32"):
    """
    Reshape算子的NumPy基准验证函数，对比NPS reshape与NumPy原生reshape结果
    """
    print(f"\n--- NumPy Benchmark Verification for [{op_name.upper()}] Operator ---")
    print(f"Input shape: {input_shape} -> Target shape: {target_shape}")

    # 1. 生成随机输入数据张量
    np_input = np.random.randn(*input_shape).astype(dtype)
    tensor_input = Tensor(*input_shape, dtype=dtype, data=np_input)
    print(f"Input data shape: {input_shape}, dtype: {dtype}")

    # 2. 生成目标形状张量（需要是整数类型）
    np_target_shape = np.array(target_shape, dtype="int64")
    tensor_target_shape = Tensor(np_target_shape.shape[0], dtype="int64", data=np_target_shape)
    print(f"Target shape tensor: {target_shape}")

    # 3. 获取NPS RESHAPE算子计算结果
    print(f"[NPS] Running {op_name.upper()} operator...")

    try:
        # 尝试创建reshape算子
        nps_op = nps_operator_class(inputs=['x', 'shape'], outputs=['y'], dtype=dtype)
        print("RESHAPE operator created successfully")

        # 尝试调用forward方法
        print("Calling forward method...")
        forward_result = nps_op.forward(tensor_input, tensor_target_shape)
        print("Forward method completed")

        # 检查返回结果的结构
        print(f"Forward result type: {type(forward_result)}")
        if isinstance(forward_result, dict):
            print(f"Forward result keys: {forward_result.keys()}")
            if "tensor" in forward_result:
                nps_result_tensor = forward_result["tensor"]
                print(f"Result tensor type: {type(nps_result_tensor)}")
                if hasattr(nps_result_tensor, 'data'):
                    nps_result = nps_result_tensor.data
                else:
                    print("Result tensor has no 'data' attribute")
                    # 尝试其他可能的属性
                    print(f"Result tensor attributes: {dir(nps_result_tensor)}")
                    return
            else:
                print("No 'tensor' key in forward result")
                return
        else:
            print("Forward result is not a dictionary")
            # 可能forward直接返回张量
            if hasattr(forward_result, 'data'):
                nps_result = forward_result.data
            else:
                print("Forward result has no 'data' attribute")
                return

        print(f"[NPS] Reshape calculation complete. Output shape: {nps_result.shape}")

    except Exception as e:
        print(f"Error during NPS reshape operation: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 获取NumPy基准计算结果（原生reshape）
    print(f"[NumPy] Running reference {op_name.upper()} calculation...")
    np_result = np_input.reshape(target_shape)  # NumPy原生reshape作为基准
    print(f"[NumPy] Reshape calculation complete. Output shape: {np_result.shape}")

    # 5. 对比结果
    print("[COMPARE] Comparing NPS RESHAPE result vs NumPy reshape result...")

    # 检查形状是否一致
    if nps_result.shape != np_result.shape:
        print(f"❌ FAILED: Shape mismatch! NPS: {nps_result.shape}, NumPy: {np_result.shape}")
        return

    # 检查数据是否一致（reshape应该保持数据不变，只是改变形状）
    if np.allclose(nps_result, np_result, atol=1e-6, rtol=1e-6):
        print(f"✅ SUCCESS: {op_name.upper()} operator matches NumPy benchmark!")
        # 验证元素总数是否保持不变
        original_elements = np.prod(input_shape)
        reshaped_elements = np.prod(target_shape)
        if original_elements == reshaped_elements:
            print(f"✅ Element count preserved: {original_elements}")
        else:
            print(f"⚠️  Element count changed: {original_elements} -> {reshaped_elements}")
    else:
        print(f"❌ FAILED: {op_name.upper()} operator differs from NumPy benchmark!")
        diff = np.abs(nps_result - np_result)
        print(f"  - Max absolute difference: {diff.max()}")
        print(f"  - Mean absolute difference: {diff.mean()}")


def test_simple_reshape():
    """测试一个更简单的reshape用例"""
    print(f"\n{'=' * 60}")
    print("Testing Simple Reshape")
    print(f"{'=' * 60}")

    # 使用更简单的形状
    input_shape = (2, 3)
    target_shape = (3, 2)
    dtype = "float32"

    np_input = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    tensor_input = Tensor(*input_shape, dtype=dtype, data=np_input)

    np_target_shape = np.array(target_shape, dtype="int64")
    tensor_target_shape = Tensor(np_target_shape.shape[0], dtype="int64", data=np_target_shape)

    print(f"Input data:\n{np_input}")
    print(f"Target shape: {target_shape}")

    try:
        nps_op = RESHAPE(inputs=['x', 'shape'], outputs=['y'], dtype=dtype)
        result = nps_op.forward(tensor_input, tensor_target_shape)

        if isinstance(result, dict) and "tensor" in result:
            nps_result = result["tensor"].data
        else:
            nps_result = result.data

        print(f"NPS reshape result:\n{nps_result}")

        # NumPy基准
        np_result = np_input.reshape(target_shape)
        print(f"NumPy reshape result:\n{np_result}")

        if np.array_equal(nps_result, np_result):
            print("✅ Simple reshape test PASSED!")
        else:
            print("❌ Simple reshape test FAILED!")

    except Exception as e:
        print(f"Error in simple reshape test: {e}")
        import traceback
        traceback.print_exc()


def test_reshape_with_different_dtype():
    """测试使用不同的数据类型"""
    print(f"\n{'=' * 60}")
    print("Testing Reshape with Different Data Types")
    print(f"{'=' * 60}")

    input_shape = (2, 4)
    target_shape = (4, 2)

    # 测试float32
    np_input_float32 = np.random.randn(*input_shape).astype("float32")
    tensor_input_float32 = Tensor(*input_shape, dtype="float32", data=np_input_float32)
    np_target_shape = np.array(target_shape, dtype="int64")
    tensor_target_shape = Tensor(np_target_shape.shape[0], dtype="int64", data=np_target_shape)

    print("Testing float32...")
    try:
        nps_op = RESHAPE(inputs=['x', 'shape'], outputs=['y'], dtype="float32")
        result = nps_op.forward(tensor_input_float32, tensor_target_shape)

        if isinstance(result, dict) and "tensor" in result:
            nps_result = result["tensor"].data
        else:
            nps_result = result.data

        np_result = np_input_float32.reshape(target_shape)

        if np.array_equal(nps_result, np_result):
            print("✅ float32 reshape test PASSED!")
        else:
            print("❌ float32 reshape test FAILED!")

    except Exception as e:
        print(f"Error in float32 reshape test: {e}")


if __name__ == "__main__":
    # 先运行简单测试
    test_simple_reshape()

    # 再运行完整测试
    # run_reshape_verification()

    # 测试不同数据类型
    # test_reshape_with_different_dtype()