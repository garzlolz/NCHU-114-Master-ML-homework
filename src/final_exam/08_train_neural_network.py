import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

print("使用字型：", font_name)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False


def build_keras_model(input_dim, num_classes, learning_rate=0.0003):
    """
    建立 Keras 神經網路模型。
    架構：1024 -> 512 -> 256 -> 128 -> Softmax
    """
    inputs = Input(shape=(input_dim,), sparse=True, name="input_features")

    # # 第一層: 1024 neurons
    x = Dense(1024, name="dense_1024")(inputs)
    x = BatchNormalization(name="batchnorm_0")(x)
    x = Activation("relu", name="activation_0")(x)
    x = Dropout(0.45, name="dropout_0")(x)

    # 第二層: 512 neurons
    x = Dense(512, name="dense_512",)(x)
    x = BatchNormalization(name="batchnorm_1")(x)
    x = Activation("relu", name="activation_1")(x)
    x = Dropout(0.45, name="dropout_1")(x)

    # 第三層: 256 neurons
    x = Dense(256, name="dense_256",)(x)
    x = BatchNormalization(name="batchnorm_2")(x)
    x = Activation("relu", name="activation_2")(x)
    x = Dropout(0.4, name="dropout_2")(x)

    # 第四層: 128 neurons
    x = Dense(128, name="dense_128")(x)
    x = BatchNormalization(name="batchnorm_3")(x)
    x = Activation("relu", name="activation_3")(x)
    x = Dropout(0.4, name="dropout_3")(x)

    # 第五層: 64 neurons
    x = Dense(64, name="dense_64")(x)
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
    print("神經網路模型訓練 (Keras Grid Search)")
    print("=" * 70)

    # 建立輸出資料夾
    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/result_images", exist_ok=True)

    # ==================== 1. 讀取傳統模型訓練結果 ====================
    print("\n步驟 1: 讀取資料")
    print("-" * 70)

    # 讀取傳統模型的訓練結果（包含 SMOTE 後的資料）
    traditional_file = "output/models/traditional_models.pkl"
    if not os.path.exists(traditional_file):
        print(f"錯誤: 找不到 {traditional_file}")
        print("請先執行 07_train_traditional.py")
        return

    with open(traditional_file, "rb") as f:
        trad_data = pickle.load(f)

    X_train_smote = trad_data["X_train_smote"]
    y_train_smote = trad_data["y_train_smote"]
    X_test = trad_data["X_test"]
    y_test = trad_data["y_test"]
    le = trad_data["label_encoder"]

    print(f"訓練集: {X_train_smote.shape}")
    print(f"測試集: {X_test.shape}")
    print(f"類別數: {len(le.classes_)}")

    # 轉換為 one-hot
    y_train_keras = to_categorical(y_train_smote, num_classes=len(le.classes_))
    y_test_keras = to_categorical(y_test, num_classes=len(le.classes_))

    # ==================== 2. Keras Grid Search ====================
    print("\n" + "=" * 70)
    print("步驟 2: Keras Grid Search (learning_rate × batch_size)")
    print("=" * 70)

    learning_rates = [0.00026, 0.00028, 0.0003]
    batch_sizes = [16, 24, 28]

    keras_histories = {}
    keras_accuracies = {}
    keras_train_times = {}

    best_keras_model = None
    best_keras_lr = None
    best_keras_bs = None
    best_keras_acc = 0
    best_keras_y_pred = None

    num_epochs = 300

    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n--- lr={lr}, batch_size={bs} ---")

            keras_model = build_keras_model(
                input_dim=X_train_smote.shape[1],
                num_classes=len(le.classes_),
                learning_rate=lr,
            )

            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",  # 監控驗證損失
                factor=0.5,  # 當指標沒進步時，學習率變為原來的 0.5 倍
                patience=5,  # 忍受 5 個 epoch 沒進步就降速
                min_lr=1e-6,  # 學習率下限
                verbose=0,  # 設為 1 可以看到何時觸發，設為 0 保持輸出乾淨
            )

            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=30,
                restore_best_weights=True,
                min_delta=0.0005,
            )

            start_time = time.time()

            # 切分訓練/驗證集
            X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
                X_train_smote,
                y_train_keras,
                test_size=0.1,
                random_state=42,
                stratify=y_train_smote,
            )

            history = keras_model.fit(
                X_train_sub,
                y_train_sub,
                batch_size=bs,
                epochs=num_epochs,
                validation_data=(X_valid, y_valid),
                callbacks=[early_stopping, reduce_lr],
                verbose=0,  # 設為 0 減少輸出
            )

            train_time = time.time() - start_time

            # 評估
            y_pred_keras_proba = keras_model.predict(X_test, verbose=0)
            y_pred_keras = np.argmax(y_pred_keras_proba, axis=1)
            keras_accuracy = accuracy_score(y_test, y_pred_keras)

            key = (lr, bs)
            keras_histories[key] = history
            keras_accuracies[key] = keras_accuracy
            keras_train_times[key] = train_time

            print(f"準確率: {keras_accuracy:.2%}, 訓練時間: {train_time:.1f} 秒")

            if keras_accuracy > best_keras_acc:
                best_keras_acc = keras_accuracy
                best_keras_lr = lr
                best_keras_bs = bs
                best_keras_model = keras_model
                best_keras_y_pred = y_pred_keras

    # ==================== 3. 輸出結果 ====================
    print("\n" + "=" * 70)
    print("Keras Grid Search 結果彙總")
    print("=" * 70)

    for (lr, bs), acc in sorted(keras_accuracies.items()):
        print(f"  lr={lr:8g}, batch_size={bs:3d}: {acc:.2%}")

    print(
        f"\n最佳組合: lr={best_keras_lr}, bs={best_keras_bs}, 準確率={best_keras_acc:.2%}"
    )

    # ==================== 4. 儲存結果 ====================
    print("\n步驟 4: 儲存神經網路訓練結果")
    print("-" * 70)

    keras_results = {
        "best_model": best_keras_model,
        "best_lr": best_keras_lr,
        "best_bs": best_keras_bs,
        "best_accuracy": best_keras_acc,
        "best_y_pred": best_keras_y_pred,
        "histories": keras_histories,
        "accuracies": keras_accuracies,
        "train_times": keras_train_times,
        "X_test": X_test,
        "y_test": y_test,
        "label_encoder": le,
    }

    keras_pkl_file = "output/models/keras_results.pkl"
    with open(keras_pkl_file, "wb") as f:
        pickle.dump(keras_results, f)
    print(f"Keras 結果已儲存到 {keras_pkl_file}")

    keras_model_file = "output/models/best_keras_model.keras"
    best_keras_model.save(keras_model_file)
    print(f"最佳 Keras 模型已儲存到 {keras_model_file}")

    # ==================== 5. 生成視覺化 ====================
    print("\n步驟 5: 生成視覺化圖表")
    print("-" * 70)

    # 5.1 訓練歷史曲線
    history = keras_histories[(best_keras_lr, best_keras_bs)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="訓練集損失")
    axes[0].plot(history.history["val_loss"], label="驗證集損失")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title(
        f"Keras 損失曲線 (lr={best_keras_lr}, bs={best_keras_bs})", fontsize=14
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="訓練集準確率")
    axes[1].plot(history.history["val_accuracy"], label="驗證集準確率")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title(
        f"Keras 準確率曲線 (lr={best_keras_lr}, bs={best_keras_bs})", fontsize=14
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "output/result_images/keras_training_history.png", dpi=300, bbox_inches="tight"
    )
    print("訓練歷史圖已儲存到 output/result_images/keras_training_history.png")
    plt.close()

    # 5.2 batch_size 比較圖
    plt.figure(figsize=(8, 5))
    bs_list = sorted(batch_sizes)
    acc_list = [keras_accuracies[(best_keras_lr, bs)] for bs in bs_list]
    plt.plot(bs_list, acc_list, marker="o")
    plt.xticks(bs_list)
    plt.xlabel("batch_size", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(
        f"Keras 不同 batch_size 準確率比較\n(learning_rate={best_keras_lr})",
        fontsize=14,
    )
    for x, yv in zip(bs_list, acc_list):
        plt.text(x, yv + 0.005, f"{yv:.2%}", ha="center", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "output/result_images/keras_batchsize_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        "batch_size 比較圖已儲存到 output/result_images/keras_batchsize_comparison.png"
    )
    plt.close()

    # 5.3 混淆矩陣
    cm = confusion_matrix(y_test, best_keras_y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title(
        f"混淆矩陣 - Neural Network (Keras)\n準確率: {best_keras_acc:.2%}",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("預測分類", fontsize=12)
    plt.ylabel("實際分類", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        "output/result_images/keras_confusion_matrix.png", dpi=300, bbox_inches="tight"
    )
    print("混淆矩陣已儲存到 output/result_images/keras_confusion_matrix.png")
    plt.close()

    print("\n" + "=" * 70)
    print("神經網路訓練完成")
    print("=" * 70)
    print("\n生成的檔案:")
    print(f"  - {keras_pkl_file} (Keras 結果)")
    print(f"  - {keras_model_file} (最佳 Keras 模型)")
    print("  - output/result_images/keras_training_history.png (訓練歷史)")
    print("  - output/result_images/keras_batchsize_comparison.png (batch_size 比較)")
    print("  - output/result_images/keras_confusion_matrix.png (混淆矩陣)")


if __name__ == "__main__":
    main()
