import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from scipy import sparse
from matplotlib import font_manager

import time

# 設定中文字型
import matplotlib.font_manager as fm

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT_PATH)

prop = fm.FontProperties(fname=FONT_PATH)
font_name = prop.get_name()

print("使用字型：", font_name)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False


def build_keras_model(input_dim, num_classes, learning_rate=0.0003):
    """
    建立 Keras 神經網路模型。
    架構固定為：
      輸入層 -> 1024 ReLU + Dropout(0.45)
             -> 512 ReLU + Dropout(0.4)
             -> 256 ReLU + Dropout(0.35)
             -> 128 ReLU + Dropout(0.3)
             -> 輸出層 (Softmax)
    learning_rate 作為超參數，供 grid search 使用。
    """
    inputs = Input(shape=(input_dim,), sparse=True, name="input_features")

    # 第一層: 1024 neurons
    x = Dense(1024, name="dense_1024")(inputs)
    x = BatchNormalization(name="batchnorm_0")(x)
    x = Activation("relu", name="activation_0")(x)
    x = Dropout(0.45, name="dropout_0")(x)

    # 第二層: 512 neurons
    x = Dense(512, name="dense_512")(x)
    x = BatchNormalization(name="batchnorm_1")(x)
    x = Activation("relu", name="activation_1")(x)
    x = Dropout(0.4, name="dropout_1")(x)

    # 第三層: 256 neurons
    x = Dense(256, name="dense_256")(x)
    x = BatchNormalization(name="batchnorm_2")(x)
    x = Activation("relu", name="activation_2")(x)
    x = Dropout(0.35, name="dropout_2")(x)

    # 第四層: 128 neurons
    x = Dense(128, name="dense_128")(x)
    x = BatchNormalization(name="batchnorm_3")(x)
    x = Activation("relu", name="activation_3")(x)
    x = Dropout(0.3, name="dropout_3")(x)

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
    print("基於多模態特徵的商品自動分類系統 - 模型訓練")
    print("=" * 70)

    # ==================== 1. 讀取處理好的特徵 ====================
    print("\n步驟 1: 讀取處理好的特徵")
    print("-" * 70)

    input_file = "output/processed_features.pkl"
    if not os.path.exists(input_file):
        print(f"錯誤: 找不到檔案 {input_file}")
        print("請先執行 05_prepare_features.py")
        return

    with open(input_file, "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]
    le = data["label_encoder"]

    print(f"特徵維度: {X.shape}")
    print(f"樣本數: {X.shape[0]}")
    print(f"類別數: {len(le.classes_)}")

    # ==================== 2. 分割訓練/測試集 ====================
    print("\n" + "=" * 70)
    print("步驟 2: 分割訓練/測試集")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"訓練集: {len(y_train)} 筆 ({len(y_train)/len(y)*100:.1f}%)")
    print(f"測試集: {len(y_test)} 筆 ({len(y_test)/len(y)*100:.1f}%)")

    # ==================== 2.5. SMOTE 過採樣 ====================
    print("\n步驟 2.5: SMOTE 過採樣")
    print("-" * 70)

    print("原始訓練集類別分佈：")
    print(Counter(y_train))

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 排序稀疏矩陣索引（避免 TensorFlow 和 SciPy 的稀疏相容性警告）
    if sparse.issparse(X_train_smote):
        X_train_smote.sort_indices()
    if sparse.issparse(X_test):
        X_test.sort_indices()

    print(f"SMOTE 後訓練集：{X_train_smote.shape}")
    print("SMOTE 後類別分佈：")
    print(Counter(y_train_smote))

    # Keras one-hot（用 SMOTE 後資料）
    y_train_keras = to_categorical(y_train_smote, num_classes=len(le.classes_))
    y_test_keras = to_categorical(y_test, num_classes=len(le.classes_))

    print(f"\nKeras 訓練標籤形狀: {y_train_keras.shape}")

    # ==================== 3. 訓練多個傳統模型 ====================
    print("\n" + "=" * 70)
    print("步驟 3: 訓練傳統模型")
    print("=" * 70)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    training_times = {}
    best_model_info = None
    best_accuracy = 0
    best_y_pred = None

    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"訓練 {name}...")
        print(f"{'='*70}")
        start_time = time.time()

        model.fit(X_train_smote, y_train_smote)
        training_times[name] = time.time() - start_time

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        print(f"準確率: {accuracy:.2%}")
        print("\n分類報告:")
        print(
            classification_report(
                y_test, y_pred, target_names=le.classes_, zero_division=0
            )
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_info = (name, model)
            best_y_pred = y_pred

    # ==================== 4. Keras Grid Search: learning_rate × batch_size ====================
    print(f"\n{'='*70}")
    print("訓練 Neural Network (Keras) - Grid Search (learning_rate × batch_size)...")
    print(f"{'='*70}")

    learning_rates = [0.0005, 0.0003, 0.0002]
    batch_sizes = [48, 64]

    keras_histories = {}      # key: (lr, bs) -> History
    keras_accuracies = {}     # key: (lr, bs) -> accuracy
    keras_train_times = {}    # key: (lr, bs) -> train time

    best_keras_model = None
    best_keras_lr = None
    best_keras_bs = None
    best_keras_acc = 0
    best_keras_y_pred = None

    num_epochs = 300 

    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n--- 使用 learning_rate = {lr}, batch_size = {bs} 訓練 Keras 模型 ---")

            keras_model = build_keras_model(
                input_dim=X_train_smote.shape[1],
                num_classes=len(le.classes_),
                learning_rate=lr,
            )

            print("\n模型結構:")
            keras_model.summary()

            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
                min_delta=1e-4,
            )

            start_time = time.time()

            # 使用 SMOTE 後資料 + one-hot，自己切 train/valid
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
                callbacks=[early_stopping],
                verbose=2,
            )

            train_time = time.time() - start_time

            # 評估
            y_pred_keras_proba = keras_model.predict(X_test)
            y_pred_keras = np.argmax(y_pred_keras_proba, axis=1)
            keras_accuracy = accuracy_score(y_test, y_pred_keras)

            key = (lr, bs)
            keras_histories[key] = history
            keras_accuracies[key] = keras_accuracy
            keras_train_times[key] = train_time

            print(
                f"\n[結果] lr={lr}, batch_size={bs} -> 準確率: {keras_accuracy:.2%}, 訓練時間: {train_time:.1f} 秒"
            )

            if keras_accuracy > best_keras_acc:
                best_keras_acc = keras_accuracy
                best_keras_lr = lr
                best_keras_bs = bs
                best_keras_model = keras_model
                best_keras_y_pred = y_pred_keras

    # 將最佳 Keras 結果納入總結果
    keras_name = (
        f"Neural Network (Keras, lr={best_keras_lr}, bs={best_keras_bs})"
    )
    results[keras_name] = best_keras_acc
    training_times[keras_name] = keras_train_times[(best_keras_lr, best_keras_bs)]

    print(f"\n{'='*70}")
    print("Keras Grid Search 結果彙總：")
    for (lr, bs), acc in keras_accuracies.items():
        print(f"  lr={lr:8g}, batch_size={bs:3d}: {acc:.2%}")
    print(
        f"\n最佳組合: learning_rate={best_keras_lr}, batch_size={best_keras_bs}, 準確率 = {best_keras_acc:.2%}"
    )
    print(f"{'='*70}")

    # 更新全域最佳模型
    if best_keras_acc > best_accuracy:
        best_accuracy = best_keras_acc
        best_model_info = (keras_name, best_keras_model)
        best_y_pred = best_keras_y_pred

    # ==================== 5. 繪製「最佳 Keras」的訓練歷史 ====================
    print("\n生成 Keras 訓練歷史圖（使用最佳組合）...")
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
    plt.savefig("output/keras_training_history.png", dpi=300, bbox_inches="tight")
    print("Keras 訓練歷史圖已儲存到 output/keras_training_history.png")
    plt.close()

    # 額外：畫一張「batch_size vs 準確率」圖（固定最佳 learning_rate）
    print("生成 batch_size vs 準確率 圖 (固定最佳 learning_rate)...")
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
    plt.savefig("output/keras_batchsize_comparison.png", dpi=300, bbox_inches="tight")
    print("batch_size 比較圖已儲存到 output/keras_batchsize_comparison.png")
    plt.close()

    # ==================== 6. 儲存最佳模型 ====================
    print("\n" + "=" * 70)
    print(f"最佳模型: {best_model_info[0]}")
    print(f"準確率: {best_accuracy:.2%}")
    print("=" * 70)

    data["model"] = best_model_info[1]
    data["model_name"] = best_model_info[0]
    data["accuracy"] = best_accuracy

    with open("output/best_model.pkl", "wb") as f:
        pickle.dump(data, f)
    print("模型已儲存到 output/best_model.pkl")

    if "Keras" in best_model_info[0]:
        best_model_info[1].save("output/best_keras_model.keras")
        print("Keras 模型已儲存到 output/best_keras_model.keras")

    # ==================== 7. 混淆矩陣視覺化 ====================
    print("\n生成混淆矩陣...")
    cm = confusion_matrix(y_test, best_y_pred)
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
        f"混淆矩陣 - {best_model_info[0]}\n準確率: {best_accuracy:.2%}",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("預測分類", fontsize=12)
    plt.ylabel("實際分類", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("output/confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("混淆矩陣已儲存到 output/confusion_matrix.png")
    plt.close()

    # ==================== 8. 模型比較圖 ====================
    print("生成模型比較圖...")
    plt.figure(figsize=(14, 6))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]
    bars = plt.bar(results.keys(), results.values(), color=colors[: len(results)])
    plt.ylabel("準確率", fontsize=12)
    plt.title("不同模型準確率比較", fontsize=16)
    plt.ylim(0, 1)
    for i, (name, acc) in enumerate(results.items()):
        plt.text(
            i, acc + 0.01, f"{acc:.2%}", ha="center", fontsize=11, fontweight="bold"
        )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("output/model_comparison.png", dpi=300, bbox_inches="tight")
    print("模型比較圖已儲存到 output/model_comparison.png")
    plt.close()

    # ==================== 9. 訓練時間比較圖 ====================
    print("生成訓練時間比較圖...")

    fig, ax = plt.subplots(figsize=(14, 6))

    model_names = list(training_times.keys())
    times = list(training_times.values())

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]
    bars = ax.bar(
        range(len(model_names)), times, color=colors[: len(model_names)], alpha=0.8
    )

    ax.set_ylabel("訓練時間 (秒)", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_title("模型訓練時間比較", fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, axis="y")

    for i, t in enumerate(times):
        if t < 60:
            time_text = f"{t:.1f}s"
        elif t < 3600:
            time_text = f"{t/60:.1f}m"
        else:
            time_text = f"{t/3600:.2f}h"

        ax.text(
            i,
            t + max(times) * 0.02,
            time_text,
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("output/training_time_comparison.png", dpi=300, bbox_inches="tight")
    print("訓練時間比較圖已儲存到 output/training_time_comparison.png")
    plt.close()

    # 輸出訓練時間統計
    print("\n" + "=" * 70)
    print("訓練時間統計")
    print("=" * 70)
    for name, t in training_times.items():
        if t < 60:
            print(f"  {name:40s} {t:8.2f} 秒")
        elif t < 3600:
            print(f"  {name:40s} {t/60:8.2f} 分鐘")
        else:
            print(f"  {name:40s} {t/3600:8.2f} 小時")

    print("\n" + "=" * 70)
    print("訓練完成")
    print("=" * 70)
    print("\n生成的檔案:")
    print("  - output/best_model.pkl (訓練好的模型)")
    print("  - output/best_keras_model.keras (Keras 模型，如果是最佳)")
    print("  - output/confusion_matrix.png (混淆矩陣)")
    print("  - output/model_comparison.png (模型比較)")
    print("  - output/keras_training_history.png (Keras 訓練歷史，最佳組合)")
    print("  - output/keras_batchsize_comparison.png (固定最佳 lr 的 batch_size 比較)")
    print("  - output/training_time_comparison.png (模型訓練時間比較)")


if __name__ == "__main__":
    main()
