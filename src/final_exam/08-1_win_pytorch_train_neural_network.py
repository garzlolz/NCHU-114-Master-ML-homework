import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_IMAGE_DATA_FORMAT"] = "channels_last"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from scipy import sparse 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 使用 Keras 3 語法取代 tensorflow.keras
# ==========================================
import keras
from keras import Model, Input, regularizers
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

# 設定中文字體
from utils.cross_platform_config import set_matplotlib_font
font_name = set_matplotlib_font()

print("使用字型：", font_name)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False

print(f"目前使用的 Keras 後端: {keras.backend.backend()}")


def build_keras_model(input_dim, num_classes, learning_rate=0.0003):
    """
    建立 Keras 神經網路模型 (PyTorch Backend)。
    架構：512 -> 256 -> 128 -> 64 -> Softmax
    """
    inputs = Input(shape=(input_dim,), name="input_features")

    # L2 正則化
    l2_reg = regularizers.l2(0.0005)

    # 第一層: 512 neurons
    x = Dense(512, name="dense_512", kernel_regularizer=l2_reg)(inputs)
    x = BatchNormalization(name="batchnorm_1")(x)
    x = Activation("relu", name="activation_1")(x)
    x = Dropout(0.45, name="dropout_1")(x)

    # 第二層: 256 neurons
    x = Dense(256, name="dense_256", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization(name="batchnorm_2")(x)
    x = Activation("relu", name="activation_2")(x)
    x = Dropout(0.4, name="dropout_2")(x)

    # 第三層: 128 neurons
    x = Dense(128, name="dense_128", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization(name="batchnorm_3")(x)
    x = Activation("relu", name="activation_3")(x)
    x = Dropout(0.4, name="dropout_3")(x)

    # 第四層: 64 neurons
    x = Dense(64, name="dense_64", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization(name="batchnorm_4")(x)
    x = Activation("relu", name="activation_4")(x)
    x = Dropout(0.3, name="dropout_4")(x)

    outputs = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="product_classifier_keras")

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )

    return model


def main():
    print("=" * 70)
    print("神經網路模型訓練 (Keras 3 + PyTorch Backend)")
    print("=" * 70)

    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/result_images", exist_ok=True)

    # ==================== 1. 讀取資料 ====================
    print("\n步驟 1: 讀取資料")
    traditional_file = "output/models/traditional_models.pkl"
    
    if not os.path.exists(traditional_file):
        print(f"錯誤: 找不到 {traditional_file}")
        return

    with open(traditional_file, "rb") as f:
        trad_data = pickle.load(f)

    X_train_smote = trad_data["X_train_smote"]
    y_train_smote = trad_data["y_train_smote"]
    X_test = trad_data["X_test"]
    y_test = trad_data["y_test"]
    le = trad_data["label_encoder"]

    # 將稀疏矩陣轉換為 Dense Array
    if sparse.issparse(X_train_smote):
        print("正在將訓練集轉換為 Dense Array (加速 PyTorch 訓練)...")
        X_train_smote = X_train_smote.toarray().astype('float32')
        
    if sparse.issparse(X_test):
        print("正在將測試集轉換為 Dense Array...")
        X_test = X_test.toarray().astype('float32')

    print(f"訓練集: {X_train_smote.shape}")
    print(f"測試集: {X_test.shape}")
    
    y_train_keras = to_categorical(y_train_smote, num_classes=len(le.classes_))

    # ==================== 2. Grid Search ====================
    print("\n" + "=" * 70)
    print("步驟 2: Grid Search (L2 Regularized)")
    print("=" * 70)

    learning_rates = [0.00026, 0.00028, 0.0003]
    batch_sizes = [16, 24, 28]

    keras_histories = {}
    keras_accuracies = {}
    keras_train_times = {}

    best_acc = 0
    best_params = {}
    best_model = None
    best_y_pred = None
    
    X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
        X_train_smote, y_train_keras, test_size=0.1, random_state=42, stratify=y_train_smote
    )

    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n--- Training: lr={lr}, bs={bs} ---")
            
            model = build_keras_model(X_train_smote.shape[1], len(le.classes_), lr)

            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0
            )
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True, min_delta=0.0005
            )

            start = time.time()
            
            history = model.fit(
                X_train_sub, y_train_sub,
                batch_size=bs,
                epochs=300,
                validation_data=(X_valid, y_valid),
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            train_time = time.time() - start
            
            probs = model.predict(X_test, verbose=0)
            preds = np.argmax(probs, axis=1)
            acc = accuracy_score(y_test, preds)
            
            print(f"結果 -> 準確率: {acc:.2%}, 耗時: {train_time:.1f}s")
            
            key = (lr, bs)
            keras_accuracies[key] = acc
            keras_histories[key] = history
            keras_train_times[key] = train_time
            
            if acc > best_acc:
                best_acc = acc
                best_params = {'lr': lr, 'bs': bs}
                best_model = model
                best_y_pred = preds

    # ==================== 3. 輸出結果 ====================
    print("\n" + "=" * 70)
    print("Keras Grid Search 結果彙總")
    print("=" * 70)

    for (lr, bs), acc in sorted(keras_accuracies.items()):
        print(f"  lr={lr:8g}, batch_size={bs:3d}: {acc:.2%}")

    print(f"\n最佳組合: lr={best_params['lr']}, bs={best_params['bs']}, 準確率={best_acc:.2%}")

    # ==================== 4. 儲存結果 ====================
    print("\n步驟 4: 儲存訓練結果")
    best_model.save("output/models/best_keras_model.keras")
    
    keras_results = {
        "best_model": None, # 避免 pickle 錯誤，這裡不存 model 物件
        "best_lr": best_params['lr'],
        "best_bs": best_params['bs'],
        "best_accuracy": best_acc,
        "best_y_pred": best_y_pred,
        "histories": keras_histories,
        "accuracies": keras_accuracies,
        "train_times": keras_train_times,
        "X_test": X_test,
        "y_test": y_test,
        "label_encoder": le,
    }
    
    with open("output/models/keras_results.pkl", "wb") as f:
        pickle.dump(keras_results, f)
        
    print("模型與結果已儲存。")

    # ==================== 5. 視覺化 ====================
    print("\n步驟 5: 生成圖表")
    
    # 5.1 Training History
    best_hist = keras_histories[(best_params['lr'], best_params['bs'])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(best_hist.history["loss"], label="Train Loss")
    axes[0].plot(best_hist.history["val_loss"], label="Val Loss")
    axes[0].set_title(f"Loss Curve (lr={best_params['lr']}, bs={best_params['bs']})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(best_hist.history["accuracy"], label="Train Acc")
    axes[1].plot(best_hist.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("output/result_images/keras_training_history.png", dpi=300)
    plt.close()
    
    # 5.2 Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, best_y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix (PyTorch)\nAccuracy: {best_acc:.2%}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("output/result_images/keras_confusion_matrix.png", dpi=300)
    plt.close()
    
    # 5.3 Batch Size Comparison
    plt.figure(figsize=(8, 5))
    bs_list = sorted(batch_sizes)
    acc_list = [keras_accuracies[(best_params['lr'], bs)] for bs in bs_list]
    plt.plot(bs_list, acc_list, marker="o")
    plt.xticks(bs_list)
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.title(f"Batch Size Comparison (lr={best_params['lr']})")
    for x, yv in zip(bs_list, acc_list):
        plt.text(x, yv + 0.001, f"{yv:.2%}", ha="center", va="bottom")
    plt.grid(True, alpha=0.3)
    plt.savefig("output/result_images/keras_batchsize_comparison.png", dpi=300)
    plt.close()

    print("圖表已生成完成。")

if __name__ == "__main__":
    main()