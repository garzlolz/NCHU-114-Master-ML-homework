import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
import numpy as np  # 需要 numpy

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)  # 新增 RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from scipy import sparse

# 設定中文字體
from utils.cross_platform_config import set_matplotlib_font

font_name = set_matplotlib_font()
print("使用字型：", font_name)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False


def main():
    print("=" * 70)
    print("傳統機器學習模型訓練 (含 Random Forest 優化)")
    print("=" * 70)

    # 建立輸出資料夾
    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/result_images", exist_ok=True)

    # ==================== 1. 讀取處理好的特徵 ====================
    print("\n步驟 1: 讀取處理好的特徵")
    input_file = "output/processed_features.pkl"
    if not os.path.exists(input_file):
        print(f"錯誤: 找不到檔案 {input_file}")
        return

    with open(input_file, "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]
    le = data["label_encoder"]

    # ==================== 2. 分割訓練/測試集 ====================
    print("\n步驟 2: 分割訓練/測試集")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ==================== 3. SMOTE 過採樣 ====================
    print("\n步驟 3: SMOTE 過採樣")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    if sparse.issparse(X_train_smote):
        X_train_smote.sort_indices()
    if sparse.issparse(X_test):
        X_test.sort_indices()

    print(f"SMOTE 後訓練集樣本數: {X_train_smote.shape[0]}")

    # ==================== 4. 定義模型與優化 ====================
    print("\n" + "=" * 70)
    print("步驟 4: 尋找 Random Forest 最佳參數 (Optimization)")
    print("=" * 70)

    # --- 4.1 設定 Random Forest 的參數網格 ---
    # 這些是我們希望程式幫忙嘗試的範圍
    rf_param_dist = {
        "n_estimators": [100, 200, 300],  # 樹的數量：越多通常越穩，但越慢
        "max_depth": [
            None,
            20,
            30,
            50,
        ],  # 樹的深度：None表示無限深(易過擬合)，數字限制深度(抗過擬合)
        "min_samples_split": [
            2,
            5,
            10,
        ],  # 分裂節點所需的最小樣本數：越大越保守(抗過擬合)
        "min_samples_leaf": [1, 2, 4],  # 葉子節點最小樣本數：越大越平滑(抗過擬合)
        "max_features": ["sqrt", "log2"],  # 每次分裂考慮的特徵數量
        "bootstrap": [True, False],  # 是否取後放回
    }

    print("正在進行 Randomized Search (這可能需要幾分鐘)...")
    start_search = time.time()

    # 建立基礎模型
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    # 設定搜尋器
    # n_iter=20 代表隨機抽 20 種組合來跑，cv=3 代表做 3 折交叉驗證
    rf_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=rf_param_dist,
        n_iter=20,  # 嘗試的組合次數，想跑久一點求精準可改 50
        cv=3,  # Cross Validation 折數
        verbose=1,
        random_state=42,
        n_jobs=-1,  # 使用所有 CPU 核心
        scoring="accuracy",
    )

    # 開始搜尋
    rf_search.fit(X_train_smote, y_train_smote)

    print(f"搜尋結束，耗時: {time.time() - start_search:.2f} 秒")
    print(f"Random Forest 最佳參數: {rf_search.best_params_}")
    print(f"最佳驗證分數: {rf_search.best_score_:.2%}")

    # --- 4.2 設定最終要訓練的模型清單 ---
    models = {
        # 使用剛剛找到的最佳參數
        "Random Forest": rf_search.best_estimator_,
        # Logistic Regression 保持不變 (或是也可以手動微調 C 值)
        "Logistic Regression": LogisticRegression(
            solver="lbfgs", max_iter=1000, C=1, random_state=42, n_jobs=-1
        ),
    }

    # ==================== 5. 執行訓練與評估 ====================
    print("\n步驟 5: 使用最佳參數進行最終評估")

    results = {}
    training_times = {}
    predictions = {}
    best_estimators = {}

    for name, model in models.items():
        print(f"\n正在評估: {name} ...")
        start_time = time.time()

        # 注意：如果 model 來自 best_estimator_，其實已經 fit 過了
        # 但為了計算純粹的訓練時間與確保流程一致，我們這裡重新 fit 一次
        model.fit(X_train_smote, y_train_smote)

        elapsed_time = time.time() - start_time
        training_times[name] = elapsed_time
        best_estimators[name] = model

        # 預測
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results[name] = acc
        predictions[name] = y_pred

        print(f" -> 測試集準確率: {acc:.2%}")
        print(f" -> 耗時: {elapsed_time:.2f} 秒")

    # ==================== 6. 儲存結果 ====================
    print("\n" + "=" * 70)
    print("步驟 6: 儲存訓練結果")

    traditional_results = {
        "models": best_estimators,
        "results": results,
        "training_times": training_times,
        "predictions": predictions,
        "X_train": X_train,
        "y_train": y_train,
        "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote,
        "X_test": X_test,
        "y_test": y_test,
        "label_encoder": le,
        "rf_best_params": rf_search.best_params_,  # 額外儲存最佳參數
    }

    model_file = "output/models/traditional_models.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(traditional_results, f)
    print(f"最佳模型結果已儲存到 {model_file}")

    # ==================== 7. 生成混淆矩陣 ====================
    print("\n步驟 7: 更新混淆矩陣圖表")

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    for idx, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if idx == 0 else "Greens",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=axes[idx],
        )
        axes[idx].set_title(
            f"{name} (Optimized)\nAcc: {results[name]:.2%}", fontsize=14
        )
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("output/result_images/traditional_confusion_matrices.png", dpi=300)
    print("圖表已更新。")


if __name__ == "__main__":
    main()
