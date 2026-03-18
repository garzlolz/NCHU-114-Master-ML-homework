import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
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
    print("傳統機器學習模型訓練")
    print("=" * 70)

    # 建立輸出資料夾
    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/result_images", exist_ok=True)

    # ==================== 1. 讀取處理好的特徵 ====================
    print("\n步驟 1: 讀取處理好的特徵")
    print("-" * 70)

    input_file = "output/processed_features.pkl"
    if not os.path.exists(input_file):
        print(f"錯誤: 找不到檔案 {input_file}")
        print("請先執行 06_prepare_features.py")
        return

    with open(input_file, "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]
    le = data["label_encoder"]

    print(f"特徵維度: {X.shape}")
    print(f"樣本數: {X.shape[0]}")

    # ==================== 2. 分割訓練/測試集 ====================
    print("\n步驟 2: 分割訓練/測試集")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ==================== 3. SMOTE 過採樣 ====================
    print("\n步驟 3: SMOTE 過採樣")

    # 注意: 這裡只對訓練集做 SMOTE，測試集必須保持純淨
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    if sparse.issparse(X_train_smote):
        X_train_smote.sort_indices()
    if sparse.issparse(X_test):
        X_test.sort_indices()

    print(f"SMOTE 後訓練集樣本數: {X_train_smote.shape[0]}")

    # ==================== 4. 定義最佳模型 ====================
    print("\n" + "=" * 70)
    print("步驟 4: 使用最佳參數建立模型")
    print("=" * 70)

    # 根據您提供的 Log 直接填入最佳參數
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features="sqrt",
            max_depth=50,
            random_state=42,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            solver="lbfgs", max_iter=1000, C=1, random_state=42, n_jobs=-1
        ),
    }

    # ==================== 5. 執行訓練 ====================
    print("步驟 5: 開始訓練模型")

    results = {}
    training_times = {}
    predictions = {}
    best_estimators = {}  # 為了保持格式一致，這裡直接存模型

    for name, model in models.items():
        print(f"\n正在訓練: {name} (Best Params) ...")
        start_time = time.time()

        # 直接訓練
        model.fit(X_train_smote, y_train_smote)

        # 記錄時間
        elapsed_time = time.time() - start_time
        training_times[name] = elapsed_time

        # 儲存模型
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
        axes[idx].set_title(f"{name} (Final)\nAcc: {results[name]:.2%}", fontsize=14)
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("output/result_images/traditional_confusion_matrices.png", dpi=300)
    print("圖表已更新。")


if __name__ == "__main__":
    main()
