import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 設定中文字體
from utils.cross_platform_config import set_matplotlib_font

font_name = set_matplotlib_font()
print("使用字型：", font_name)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False


def build_keras_model(input_dim, num_classes, learning_rate):
    """
    建立 Keras 神經網路模型 (TensorFlow Backend)。
    架構：512 -> 256 -> 128 -> 64 -> Softmax
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
    x = Dropout(0.35, name="dropout_1")(x)

    # 第三層: 128
    x = Dense(128, name="dense_128")(x)
    x = BatchNormalization(name="batchnorm_2")(x)
    x = Activation("relu", name="activation_2")(x)
    x = Dropout(0.3, name="dropout_2")(x)

    # 第四層: 64
    x = Dense(64, name="dense_64")(x)
    x = BatchNormalization(name="batchnorm_3")(x)
    x = Activation("relu", name="activation_3")(x)
    x = Dropout(0.25, name="dropout_3")(x)

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
    print("神經網路模型訓練 (Keras + TensorFlow Backend)")
    print("=" * 70)

    TARGET_SEED = 42  # 使用固定種子以便重現

    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/result_images", exist_ok=True)

    # ==================== 1. 讀取資料 ====================
    print("\n步驟 1: 讀取資料")
    print("-" * 70)

    traditional_file = "output/models/traditional_models.pkl"
    if not os.path.exists(traditional_file):
        print(f"錯誤: 找不到 {traditional_file}")
        print("請先執行 07_train_traditional_model.py")
        return

    with open(traditional_file, "rb") as f:
        trad_data = pickle.load(f)

    # 載入原始訓練集（未 SMOTE）
    X_train_orig = trad_data["X_train"]
    y_train_orig = trad_data["y_train"]
    X_test = trad_data["X_test"]
    y_test = trad_data["y_test"]
    le = trad_data["label_encoder"]

    print(f"原始訓練集: {X_train_orig.shape}")
    print(f"測試集: {X_test.shape}")
    print(f"類別數: {len(le.classes_)}")

    # ==================== 2. 分割驗證集 ====================
    print("\n步驟 2: 分割驗證集（在 SMOTE 之前）")
    print("-" * 70)

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
    print("-" * 70)

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_sub, y_train_sub)

    print(f"SMOTE 後訓練集: {X_train_smote.shape}")

    # 轉換為 one-hot encoding
    y_train_keras = to_categorical(y_train_smote, num_classes=len(le.classes_))
    y_valid_keras = to_categorical(y_valid, num_classes=len(le.classes_))

    # ==================== 4. 訓練模型 ====================
    print("\n步驟 4: 開始訓練")
    print("=" * 70)

    lr = 0.00025
    bs = 20

    print(f"\n參數設定: Seed={TARGET_SEED}, lr={lr}, bs={bs}")
    print("-" * 70)

    model = build_keras_model(X_train_smote.shape[1], len(le.classes_), lr)

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        min_delta=0.001,
        verbose=1,
    )

    start_time = time.time()

    history = model.fit(
        X_train_smote,
        y_train_keras,
        batch_size=bs,
        epochs=100,
        validation_data=(X_valid, y_valid_keras),
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    train_time = time.time() - start_time

    # ==================== 5. 評估模型 ====================
    print("\n步驟 5: 評估模型")
    print("-" * 70)

    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)
    final_val_loss = history.history["val_loss"][-1]

    print(f"\n測試集準確率: {acc:.2%}")
    print(f"最終 Validation Loss: {final_val_loss:.4f}")
    print(f"訓練耗時: {train_time:.1f} 秒")

    # ==================== 6. 儲存結果 ====================
    print("\n步驟 6: 儲存訓練結果")
    print("-" * 70)

    # 儲存 .keras 模型檔案
    keras_model_file = "output/models/best_keras_model.keras"
    model.save(keras_model_file)
    print(f"Keras 模型已儲存: {keras_model_file}")

    # 儲存結果 pickle（與 PyTorch 版本結構一致）
    keras_results = {
        "best_model": None,  # 不儲存模型物件，使用 .keras 檔案
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

    keras_pkl_file = "output/models/keras_results.pkl"
    with open(keras_pkl_file, "wb") as f:
        pickle.dump(keras_results, f)
    print(f"結果 pickle 已儲存: {keras_pkl_file}")

    # ==================== 7. 視覺化 ====================
    print("\n步驟 7: 生成圖表")
    print("-" * 70)

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
    plt.savefig(
        "output/result_images/keras_training_history.png", dpi=300, bbox_inches="tight"
    )
    print("訓練歷史圖已儲存: output/result_images/keras_training_history.png")
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
    plt.savefig(
        "output/result_images/keras_confusion_matrix.png", dpi=300, bbox_inches="tight"
    )
    print("混淆矩陣已儲存: output/result_images/keras_confusion_matrix.png")
    plt.close()

    print("\n" + "=" * 70)
    print("神經網路訓練完成")
    print("=" * 70)
    print("\n生成的檔案:")
    print(f"  - {keras_pkl_file}")
    print(f"  - {keras_model_file}")
    print("  - output/result_images/keras_training_history.png")
    print("  - output/result_images/keras_confusion_matrix.png")


if __name__ == "__main__":
    main()
