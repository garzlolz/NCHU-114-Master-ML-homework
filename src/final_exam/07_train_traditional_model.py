import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time

from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from scipy import sparse
import matplotlib.font_manager as fm

# 設定中文字體
from utils.cross_platform_config import set_matplotlib_font

font_name = set_matplotlib_font()

print("使用字型：", font_name)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False


def main():
    print("=" * 70)
    print("傳統機器學習模型訓練 (Random Forest & Logistic Regression)")
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

    # ==================== 3. SMOTE 過採樣 ====================
    print("\n步驟 3: SMOTE 過採樣")
    print("-" * 70)

    print("原始訓練集類別分佈：")
    print(Counter(y_train))

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 排序稀疏矩陣索引
    if sparse.issparse(X_train_smote):
        X_train_smote.sort_indices()
    if sparse.issparse(X_test):
        X_test.sort_indices()

    print(f"SMOTE 後訓練集：{X_train_smote.shape}")
    print("SMOTE 後類別分佈：")
    print(Counter(y_train_smote))

    # ==================== 4. 訓練傳統模型 ====================
    print("\n" + "=" * 70)
    print("步驟 4: 訓練傳統模型")
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
    predictions = {}  # 儲存預測結果

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
        predictions[name] = y_pred

        print(f"準確率: {accuracy:.2%}")
        print(f"訓練時間: {training_times[name]:.2f} 秒")
        print("\n分類報告:")
        print(
            classification_report(
                y_test, y_pred, target_names=le.classes_, zero_division=0
            )
        )

    # ==================== 5. 儲存結果 ====================
    print("\n" + "=" * 70)
    print("步驟 5: 儲存訓練結果")
    print("=" * 70)

    # 儲存模型與結果
    traditional_results = {
        "models": models,
        "results": results,
        "training_times": training_times,
        "predictions": predictions,
        "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote,
        "X_test": X_test,
        "y_test": y_test,
        "label_encoder": le,
    }

    model_file = "output/models/traditional_models.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(traditional_results, f)
    print(f"傳統模型結果已儲存到 {model_file}")

    # ==================== 6. 生成混淆矩陣 ====================
    print("\n生成傳統模型混淆矩陣...")

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
            f"混淆矩陣 - {name}\n準確率: {results[name]:.2%}",
            fontsize=14,
            pad=15,
        )
        axes[idx].set_xlabel("預測分類", fontsize=11)
        axes[idx].set_ylabel("實際分類", fontsize=11)
        axes[idx].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        "output/result_images/traditional_confusion_matrices.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("混淆矩陣已儲存到 output/result_images/traditional_confusion_matrices.png")
    plt.close()

    # ==================== 7. 輸出摘要 ====================
    print("\n" + "=" * 70)
    print("訓練完成 - 結果摘要")
    print("=" * 70)

    for name in models.keys():
        print(f"\n{name}:")
        print(f"  準確率: {results[name]:.2%}")
        print(f"  訓練時間: {training_times[name]:.2f} 秒")

    print("\n生成的檔案:")
    print(f"  - {model_file} (模型與結果)")
    print("  - output/result_images/traditional_confusion_matrices.png (混淆矩陣)")


if __name__ == "__main__":
    main()
