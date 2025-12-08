# 06_train_model.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 設定中文字型
import matplotlib.font_manager as fm

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
prop = fm.FontProperties(fname=FONT_PATH)
font_name = prop.get_name()

print("使用字型：", font_name)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False


def build_keras_model(input_dim, num_classes):
    """
    使用 Functional API 建立 Keras 神經網路模型
    結構：input -> 1024 -> 512 -> 256 -> 128 -> output(softmax)
    """
    inputs = Input(shape=(input_dim,), name="input_features")

    # 第一層：1024 neurons
    x = Dense(1024, activation="relu", name="dense_1024")(inputs)
    x = Dropout(0.4, name="dropout_0")(x)

    # 第二層：512 neurons
    x = Dense(512, activation="relu", name="dense_512")(x)
    x = Dropout(0.3, name="dropout_1")(x)

    # 第三層：256 neurons
    x = Dense(256, activation="relu", name="dense_256")(x)
    x = Dropout(0.3, name="dropout_2")(x)

    # 第四層：128 neurons
    x = Dense(128, activation="relu", name="dense_128")(x)
    x = Dropout(0.2, name="dropout_3")(x)

    # 輸出層：num_classes, softmax
    outputs = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="product_classifier_keras")

    optimizer = Adam(learning_rate=0.0005)

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

    # ==================== 為 Keras 準備標籤（One-Hot Encoding）====================
    y_train_keras = to_categorical(y_train, num_classes=len(le.classes_))
    y_test_keras = to_categorical(y_test, num_classes=len(le.classes_))
    print(f"\nKeras 標籤形狀: {y_train_keras.shape}")

    # ==================== 3. 訓練多個傳統模型 ====================
    print("\n" + "=" * 70)
    print("步驟 3: 訓練模型")
    print("=" * 70)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
        "Neural Network (sklearn)": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False,
        ),
    }

    results = {}
    best_model_info = None
    best_accuracy = 0
    best_y_pred = None

    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"訓練 {name}...")
        print(f"{'='*70}")

        model.fit(X_train, y_train)
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

    # ==================== 4. 訓練多個 batch_size 的 Keras 模型 ====================
    print(f"\n{'='*70}")
    print(f"訓練 Neural Network (Keras) with different batch sizes...")
    print(f"{'='*70}")

    batch_sizes = [32, 64, 128]
    keras_histories = {}
    keras_accuracies = {}
    best_keras_model = None
    best_keras_bs = None
    best_keras_acc = 0
    best_keras_y_pred = None

    for bs in batch_sizes:
        print(f"\n--- 使用 batch_size = {bs} 訓練 Keras 模型 ---")

        # 每個 batch size 重新建一個新模型，避免權重延續[web:66]
        keras_model = build_keras_model(
            input_dim=X_train.shape[1], num_classes=len(le.classes_)
        )

        print("\n模型結構:")
        keras_model.summary()

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
        )

        history = keras_model.fit(
            X_train,
            y_train_keras,
            epochs=120,
            batch_size=bs,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=1,
        )

        keras_histories[bs] = history

        # 評估
        y_pred_keras_proba = keras_model.predict(X_test)
        y_pred_keras = np.argmax(y_pred_keras_proba, axis=1)
        keras_accuracy = accuracy_score(y_test, y_pred_keras)
        keras_accuracies[bs] = keras_accuracy

        print(f"\nbatch_size={bs} 準確率: {keras_accuracy:.2%}")

        if keras_accuracy > best_keras_acc:
            best_keras_acc = keras_accuracy
            best_keras_bs = bs
            best_keras_model = keras_model
            best_keras_y_pred = y_pred_keras

    # 將最佳 Keras 結果納入總結果
    results[f"Neural Network (Keras, bs={best_keras_bs})"] = best_keras_acc

    print(f"\n{'='*70}")
    print(f"Keras 不同 batch_size 結果：")
    for bs, acc in keras_accuracies.items():
        print(f"  batch_size={bs}: {acc:.2%}")
    print(f"最佳 batch_size = {best_keras_bs}, 準確率 = {best_keras_acc:.2%}")
    print(f"{'='*70}")

    # 更新全域最佳模型
    if best_keras_acc > best_accuracy:
        best_accuracy = best_keras_acc
        best_model_info = (
            f"Neural Network (Keras, bs={best_keras_bs})",
            best_keras_model,
        )
        best_y_pred = best_keras_y_pred

    # ==================== 5. 繪製「最佳 Keras」的訓練歷史 ====================
    print("\n生成 Keras 訓練歷史圖（使用最佳 batch_size）...")
    history = keras_histories[best_keras_bs]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="訓練集損失")
    axes[0].plot(history.history["val_loss"], label="驗證集損失")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title(f"Keras 損失曲線 (bs={best_keras_bs})", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="訓練集準確率")
    axes[1].plot(history.history["val_accuracy"], label="驗證集準確率")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title(f"Keras 準確率曲線 (bs={best_keras_bs})", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/keras_training_history.png", dpi=300, bbox_inches="tight")
    print("Keras 訓練歷史圖已儲存到 output/keras_training_history.png")
    plt.close()

    # 額外：畫一張「batch_size vs 準確率」圖
    print("生成 batch_size vs 準確率 圖...")
    plt.figure(figsize=(8, 5))
    bs_list = sorted(keras_accuracies.keys())
    acc_list = [keras_accuracies[bs] for bs in bs_list]
    plt.plot(bs_list, acc_list, marker="o")
    plt.xticks(bs_list)
    plt.xlabel("batch_size", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Keras 不同 batch_size 準確率比較", fontsize=14)
    for x, yv in zip(bs_list, acc_list):
        plt.text(x, yv + 0.005, f"{yv:.2%}", ha="center", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/keras_batchsize_comparison.png", dpi=300, bbox_inches="tight")
    print("batch_size 比較圖已儲存到 output/keras_batchsize_comparison.png")
    plt.close()

    # ==================== 6. 儲存最佳模型 ====================
    print(f"\n" + "=" * 70)
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

    print("\n" + "=" * 70)
    print("訓練完成！")
    print("=" * 70)
    print(f"\n生成的檔案:")
    print(f"  - output/best_model.pkl (訓練好的模型)")
    print(f"  - output/best_keras_model.keras (Keras 模型，如果是最佳)")
    print(f"  - output/confusion_matrix.png (混淆矩陣)")
    print(f"  - output/model_comparison.png (模型比較)")
    print(f"  - output/keras_training_history.png (Keras 訓練歷史，最佳 batch_size)")
    print(f"  - output/keras_batchsize_comparison.png (Keras batch_size 準確率比較)")


if __name__ == "__main__":
    main()
