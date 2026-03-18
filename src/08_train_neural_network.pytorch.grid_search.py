import os
import pickle
import time
import csv
import platform
from datetime import datetime
import itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# 指定 PyTorch 為後端
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_IMAGE_DATA_FORMAT"] = "channels_last"

import keras
from keras import Model, Input
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from utils.cross_platform_config import set_matplotlib_font

# 字型設定
font_name = set_matplotlib_font()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False


def build_keras_model(input_dim, num_classes, learning_rate, dropout_list):
    """
    根據 84.75% 最佳模型調整架構
    架構：512 -> 256 -> 128 -> 64 -> Softmax
    dropout_list: [d0, d1, d2, d3] (共 4 個參數)
    """
    if len(dropout_list) != 4:
        raise ValueError(f"Expected 4 dropout values, got {len(dropout_list)}")

    d0, d1, d2, d3 = dropout_list

    inputs = Input(shape=(input_dim,), name="input_features")

    # 第一層: 512
    x = Dense(512, name="dense_512")(inputs)
    x = BatchNormalization(name="batchnorm_0")(x)
    x = Activation("relu", name="activation_0")(x)
    x = Dropout(d0, name="dropout_0")(x)

    # 第二層: 256
    x = Dense(256, name="dense_256")(x)
    x = BatchNormalization(name="batchnorm_1")(x)
    x = Activation("relu", name="activation_1")(x)
    x = Dropout(d1, name="dropout_1")(x)

    # 第三層: 128
    x = Dense(128, name="dense_128")(x)
    x = BatchNormalization(name="batchnorm_2")(x)
    x = Activation("relu", name="activation_2")(x)
    x = Dropout(d2, name="dropout_2")(x)

    # 第四層: 64
    x = Dense(64, name="dense_64")(x)
    x = BatchNormalization(name="batchnorm_3")(x)
    x = Activation("relu", name="activation_3")(x)
    x = Dropout(d3, name="dropout_3")(x)

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
    system_name = platform.system()
    platform_info = f"{system_name}-{platform.machine()}"

    print("=" * 70)
    print("Keras Neural Network - Grid Search (Based on Best 84.75% Model)")
    print(f"執行平台: {platform_info}")
    print("模型結構: 512 -> 256 -> 128 -> 64")
    print("=" * 70)

    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/result_images", exist_ok=True)

    # 讀取前一階段的資料
    traditional_file = "output/models/traditional_models.pkl"
    if not os.path.exists(traditional_file):
        print(f"錯誤: 找不到 {traditional_file}")
        return

    with open(traditional_file, "rb") as f:
        trad_data = pickle.load(f)

    # 載入原始訓練集（未 SMOTE）
    X_train = trad_data["X_train"]
    y_train = trad_data["y_train"]
    X_test = trad_data["X_test"]
    y_test = trad_data["y_test"]
    le = trad_data["label_encoder"]

    # 轉換格式
    if sparse.issparse(X_train):
        X_train = X_train.toarray().astype("float32")
    if sparse.issparse(X_test):
        X_test = X_test.toarray().astype("float32")

    num_classes = len(le.classes_)

    results_file = "output/models/keras_results_grid.pkl"
    csv_all_file = "output/models/keras_grid_search_optimized.csv"

    # [設定全域種子] 用於 train_test_split 的一致性
    BASE_SEED = 821407
    print(f"設定 Base Random Seed: {BASE_SEED}")
    keras.utils.set_random_seed(BASE_SEED)

    # ========== 先分割驗證集（在 SMOTE 之前） ==========
    print("\n分割訓練/驗證集（在 SMOTE 之前）...")
    X_train_orig, X_valid, y_train_orig, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=BASE_SEED,
        stratify=y_train,
    )

    print(f"訓練子集（未 SMOTE）: {X_train_orig.shape}")
    print(f"驗證集: {X_valid.shape}")

    print("\n執行 SMOTE")
    smote_start = time.time()
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_orig, y_train_orig)

    y_train_keras = to_categorical(y_train_smote, num_classes=num_classes)
    y_valid_keras = to_categorical(y_valid, num_classes=num_classes)

    # 1. Learning Rate
    lr_list = [
        0.00005,
        0.00008,
        0.0001,
        0.00012,
        0.00015,
        0.00018,
        0.0002,
        0.00022,
        0.00025,
        0.00028,
        0.0003,
        0.0004,
    ]

    # 2. Batch Size
    bs_list = [16, 18, 20, 22, 24, 26, 28, 32, 48]

    # 3. Dropout Configs
    dropout_grid = [
        [0.3, 0.25, 0.2, 0.15],
        [0.35, 0.3, 0.25, 0.2],
        [0.4, 0.35, 0.3, 0.25],  # Standard (Baseline)
        [0.45, 0.4, 0.35, 0.3],
        [0.5, 0.45, 0.4, 0.35],
        [0.55, 0.5, 0.45, 0.4],
        [0.4, 0.4, 0.4, 0.4],
        [0.3, 0.3, 0.3, 0.3],
        [0.5, 0.3, 0.3, 0.1],
        [0.2, 0.3, 0.4, 0.5],
    ]

    # 產生「所有」可能的組合
    all_combinations = list(
        itertools.product(lr_list, bs_list, range(len(dropout_grid)))
    )
    total_original_combos = len(all_combinations)

    # ================= [斷點續練邏輯 START] =================
    existing_combinations = set()
    global_best_acc = 0.0

    if os.path.isfile(csv_all_file):
        print(f"\n檢測到歷史紀錄檔: {csv_all_file}")
        try:
            with open(csv_all_file, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        r_lr = float(row["Learning_Rate"])
                        r_bs = int(row["Batch_Size"])
                        r_did = int(row["Dropout_Config_ID"])
                        r_acc = float(row["Accuracy"])
                        existing_combinations.add((r_lr, r_bs, r_did))
                        if r_acc > global_best_acc:
                            global_best_acc = r_acc
                    except (ValueError, KeyError):
                        continue
        except Exception as e:
            print(f"讀取 CSV 時發生警告 (將重新開始): {e}")

    # 過濾掉已經跑過的組合
    param_combinations = []
    for combo in all_combinations:
        lr, bs, d_id = combo
        is_done = False
        if (lr, bs, d_id) in existing_combinations:
            is_done = True
        else:
            for e_lr, e_bs, e_did in existing_combinations:
                if e_bs == bs and e_did == d_id and abs(e_lr - lr) < 1e-9:
                    is_done = True
                    break
        if not is_done:
            param_combinations.append(combo)

    skipped_count = total_original_combos - len(param_combinations)
    total_combos_to_run = len(param_combinations)

    print("\nGrid Search 進度確認：")
    print(f"  原始總組合數 : {total_original_combos}")
    print(f"  已執行過組合 : {skipped_count}")
    print(f"  目前歷史最佳 : {global_best_acc:.6f}")
    print(f"  剩餘待執行   : {total_combos_to_run}")
    print("=" * 70)

    if total_combos_to_run == 0:
        print("所有組合皆已執行完畢！程式結束。")
        return
    # ================= [斷點續練邏輯 END] =================

    if not os.path.isfile(csv_all_file):
        with open(csv_all_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "時間戳記",
                    "Index",
                    "Learning_Rate",
                    "Batch_Size",
                    "Dropout_Config_ID",
                    "Dropout_Values",
                    "Accuracy",
                    "訓練耗時(秒)",
                    "平台",
                ]
            )

    best_params = None
    best_preds = None
    best_history = None
    start_session_time = time.time()
    start_idx = skipped_count + 1

    for idx, (lr, bs, d_id) in enumerate(param_combinations, start=start_idx):
        dropouts = dropout_grid[d_id]

        keras.utils.set_random_seed(BASE_SEED)

        print(
            f"\n[{idx:03d}/{total_original_combos}] 開始訓練 - "
            f"lr={lr}, bs={bs}, dropout={dropouts}"
        )


        model = build_keras_model(
            input_dim=X_train_smote.shape[1],
            num_classes=num_classes,
            learning_rate=lr,
            dropout_list=dropouts,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=0.001,
        )

        start_time = time.time()

        history = model.fit(
            X_train_smote,
            y_train_keras,
            batch_size=bs,
            epochs=100,
            validation_data=(X_valid, y_valid_keras),
            callbacks=[early_stopping, reduce_lr],
            verbose=0,
        )
        elapsed_time = time.time() - start_time

        probs = model.predict(X_test, verbose=0)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_test, preds)

        print(f" -> 測試集準確率: {acc:.4f}")
        print(f" -> 訓練耗時    : {elapsed_time:.2f} 秒")
        print(f" -> Stop Epoch  : {len(history.history['loss'])}")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_all_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    idx,
                    lr,
                    bs,
                    d_id,
                    str(dropouts),
                    f"{acc:.6f}",
                    f"{elapsed_time:.2f}",
                    platform_info,
                ]
            )

        if acc > global_best_acc:
            global_best_acc = acc
            best_params = {
                "learning_rate": lr,
                "batch_size": bs,
                "dropout_config_id": d_id,
                "dropouts": dropouts,
            }
            best_preds = preds
            best_history = history

            print(f" *** NEW BEST *** acc: {global_best_acc:.4f}")

            # 儲存最佳結果
            model.save("output/models/best_keras_model.keras")

            keras_results = {
                "best_model": None,
                "best_lr": lr,
                "best_bs": bs,
                "best_dropout_id": d_id,
                "best_dropouts": dropouts,
                "best_accuracy": acc,
                "best_y_pred": best_preds,
                "best_history": best_history.history,
                "X_test": X_test,
                "y_test": y_test,
                "label_encoder": le,
            }
            with open(results_file, "wb") as f:
                pickle.dump(keras_results, f)

            # 繪圖
            plt.figure(figsize=(12, 10))
            cm = confusion_matrix(y_test, best_preds)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Reds",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
            )
            plt.title(
                f"Keras Grid Search Best\nAcc={global_best_acc:.4f}, lr={lr}, bs={bs}"
            )
            plt.tight_layout()
            plt.savefig("output/result_images/keras_best_confusion_matrix.png")
            plt.close()

    # 最終繪圖
    if best_history is not None:
        print("\n繪製本次最佳組合的訓練曲線...")
        h_dict = best_history.history
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(h_dict["loss"], label="Train Loss")
        plt.plot(h_dict["val_loss"], label="Val Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(h_dict["accuracy"], label="Train Acc")
        plt.plot(h_dict["val_accuracy"], label="Val Acc")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("output/result_images/keras_best_training_history.png", dpi=300)
        plt.close()

        session_time = time.time() - start_session_time
        print("\nGrid Search 完成！")
        print(f"  總耗時    : {session_time/3600:.2f} 小時")
        print(
            f"  最佳組合: lr={best_params['learning_rate']}, bs={best_params['batch_size']}, acc={global_best_acc:.4f}"
        )
    else:
        print("\n無新結果。")


if __name__ == "__main__":
    main()
