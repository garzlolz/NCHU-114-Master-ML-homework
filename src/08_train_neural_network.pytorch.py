# ==========================================
# 這隻程式碼嘗試使用 Keras 3 語法取代 tensorflow.keras
# 原因: 因為 TensorFlow 在 RTX50 系列顯示卡上的支援較差
# pytorch 對新顯示卡支援較好，因此改用 Keras 3 + PyTorch Backend
# 主要差異在於 import 的方式不同
# 其餘程式碼介面基本上跟 tensorflow.keras 相同
# ==========================================
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
from imblearn.over_sampling import SMOTE  # 新增這行

import keras
from keras import Model, Input
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


def build_keras_model(input_dim, num_classes, learning_rate):
    """
    建立 Keras 神經網路模型 (PyTorch Backend)。
    架構： 512 -> 256 -> 128 -> 64 -> Softmax
    """
    inputs = Input(shape=(input_dim,), name="input_features")
    # 第一層: 512
    x = Dense(512, name="dense_512")(inputs)
    x = BatchNormalization(name="batchnorm_0")(x)
    x = Activation("relu", name="activation_0")(x)
    x = Dropout(0.4, name="dropout_0")(x)

    # 第二層: 256
    x = Dense(256, name="dense_256")(x)
    x = BatchNormalization(name="batchnorm_1")(x)
    x = Activation("relu", name="activation_1")(x)
    x = Dropout(0.4, name="dropout_1")(x)

    # 第三層: 128
    x = Dense(128, name="dense_128")(x)
    x = BatchNormalization(name="batchnorm_2")(x)
    x = Activation("relu", name="activation_2")(x)
    x = Dropout(0.4, name="dropout_2")(x)

    # 第四層: 64
    x = Dense(64, name="dense_64")(x)
    x = BatchNormalization(name="batchnorm_3")(x)
    x = Activation("relu", name="activation_3")(x)
    x = Dropout(0.4, name="dropout_3")(x)

    outputs = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="product_classifier_keras")
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("=" * 70)
    print("神經網路模型訓練 (Keras 3 + PyTorch Backend)")
    print("=" * 70)

    TARGET_SEED = 821407
    keras.utils.set_random_seed(TARGET_SEED)

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

    # 載入原始訓練集（未 SMOTE）
    X_train_orig = trad_data["X_train"]
    y_train_orig = trad_data["y_train"]
    X_test = trad_data["X_test"]
    y_test = trad_data["y_test"]
    le = trad_data["label_encoder"]

    # 轉為 Dense Array
    if sparse.issparse(X_train_orig):
        X_train_orig = X_train_orig.toarray().astype("float32")
    if sparse.issparse(X_test):
        X_test = X_test.toarray().astype("float32")

    print(f"原始訓練集: {X_train_orig.shape}")
    print(f"測試集: {X_test.shape}")

    # ==================== 2. 分割驗證集 ====================
    print("\n步驟 2: 分割驗證集（在 SMOTE 之前）")

    # 先從原始訓練集分出驗證集
    X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
        X_train_orig,
        y_train_orig,
        test_size=0.2,
        random_state=TARGET_SEED,
        stratify=y_train_orig,
    )

    print(f"訓練子集（未 SMOTE）: {X_train_sub.shape}")
    print(f"驗證集: {X_valid.shape}")

    # ==================== 3. 對訓練子集做 SMOTE ====================
    print("\n步驟 3: 對訓練子集做 SMOTE")

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_sub, y_train_sub)

    print(f"SMOTE 後訓練集: {X_train_smote.shape}")

    # 轉換為 one-hot encoding
    y_train_keras = to_categorical(y_train_smote, num_classes=len(le.classes_))
    y_valid_keras = to_categorical(y_valid, num_classes=len(le.classes_))

    # ==================== 4. 訓練模型 ====================
    print("\n步驟 4: 開始訓練")

    lr = 0.00025
    bs = 16

    print(f"\n--- Training: Seed={TARGET_SEED}, lr={lr}, bs={bs} ---")

    model = build_keras_model(X_train_smote.shape[1], len(le.classes_), lr)

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,  # 增加耐心
        restore_best_weights=True,
        min_delta=0.001,
        verbose=1,
    )

    start = time.time()

    history = model.fit(
        X_train_smote,  # SMOTE 後的訓練子集
        y_train_keras,
        batch_size=bs,
        epochs=100,
        validation_data=(X_valid, y_valid_keras),  # 驗證集（未 SMOTE）
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    train_time = time.time() - start

    # ==================== 5. 評估模型 ====================
    print("\n步驟 5: 評估模型")

    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)
    final_val_loss = history.history["val_loss"][-1]

    print(f"\n測試集準確率: {acc:.2%}")
    print(f"最終 Validation Loss: {final_val_loss:.4f}")
    print(f"訓練耗時: {train_time:.1f}s")

    # ==================== 6. 儲存結果 ====================
    print("\n步驟 6: 儲存訓練結果")

    model.save("output/models/best_keras_model.keras")

    keras_results = {
        "best_model": None,
        "best_lr": lr,
        "best_bs": bs,
        "best_seed": TARGET_SEED,
        "best_accuracy": acc,
        "best_y_pred": preds,
        "history": history.history,
        "train_time": train_time,
        "X_test": X_test,
        "y_test": y_test,
        "label_encoder": le,
    }

    with open("output/models/keras_results.pkl", "wb") as f:
        pickle.dump(keras_results, f)

    print("模型與結果已儲存。")

    # ==================== 7. 視覺化 ====================
    print("\n步驟 7: 生成圖表")

    # 7.1 Training History
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="訓練集損失", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="驗證集損失", linewidth=2)
    axes[0].set_title(f"損失曲線 (Seed={TARGET_SEED})", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="訓練集準確率", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="驗證集準確率", linewidth=2)
    axes[1].set_title("準確率曲線", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/result_images/keras_training_history.png", dpi=300)
    plt.close()

    # 7.2 Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title(f"混淆矩陣 (Seed={TARGET_SEED})\n測試準確率: {acc:.2%}", fontsize=16)
    plt.xlabel("預測類別")
    plt.ylabel("實際類別")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("output/result_images/keras_confusion_matrix.png", dpi=300)
    plt.close()

    print("圖表已生成完成。")


if __name__ == "__main__":
    main()
