import sys
import time

print("=" * 50)
print("系統資訊")
print("=" * 50)
print(f"Python 版本: {sys.version}")
print(f"Python 路徑: {sys.executable}")

print("\n" + "=" * 50)
print("TensorFlow 基本資訊")
print("=" * 50)

try:
    import tensorflow as tf
    print(f"TensorFlow 版本: {tf.__version__}")
    print(f"TensorFlow 安裝路徑: {tf.__file__}")
except Exception as e:
    print(f"TensorFlow 載入失敗: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("GPU 偵測")
print("=" * 50)

# 列出所有 GPU 設備
gpus = tf.config.list_physical_devices("GPU")
print(f"偵測到的 GPU 數量: {len(gpus)}")

if len(gpus) > 0:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    print("GPU 可用")
else:
    print("沒有偵測到 GPU，將使用 CPU")

print("\n" + "=" * 50)
print("CUDA 和 cuDNN 資訊")
print("=" * 50)
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(
    f"GPU available: {tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)}"
)

print("\n" + "=" * 50)
print("簡單運算測試")
print("=" * 50)

# 簡單的張量運算測試
try:
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print("矩陣 A:")
    print(a.numpy())
    print("\n矩陣 B:")
    print(b.numpy())
    print("\nA × B =")
    print(c.numpy())
    print("張量運算測試成功")
except Exception as e:
    print(f"張量運算測試失敗: {e}")

print("\n" + "=" * 50)
print("建立簡單神經網路測試")
print("=" * 50)

try:
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense

    # 建立簡單的模型
    inputs = Input(shape=(10,))
    x = Dense(5, activation="relu")(inputs)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adam", loss="binary_crossentropy")
    print("神經網路模型建立成功")
    print("\n模型架構:")
    model.summary()

except Exception as e:
    print(f"神經網路測試失敗: {e}")

print("\n" + "=" * 50)
print("GPU 運算速度測試（可選）")
print("=" * 50)

# CPU 測試
with tf.device("/CPU:0"):
    start = time.time()
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    c = tf.matmul(a, b)
    cpu_time = time.time() - start
    print(f"CPU 計算時間: {cpu_time:.4f} 秒")

# GPU 測試（如果有的話）
if len(gpus) > 0:
    with tf.device("/GPU:0"):
        start = time.time()
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        c = tf.matmul(a, b)
        gpu_time = time.time() - start
        print(f"GPU 計算時間: {gpu_time:.4f} 秒")
        print(f"GPU 加速比: {cpu_time / gpu_time:.2f}x")
else:
    print("無 GPU，跳過 GPU 測試")

print("\n" + "=" * 50)
print("測試完成")
print("=" * 50)
