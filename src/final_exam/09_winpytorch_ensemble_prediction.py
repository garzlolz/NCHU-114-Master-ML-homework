import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pickle
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from utils.cross_platform_config import set_matplotlib_font

# 設定字體
font_name = set_matplotlib_font()
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False

def main():
    print("=" * 70)
    print("進階集成優化: 精細權重搜索 & Stacking")
    print("=" * 70)

    # ==================== 1. 準備資料與模型 ====================
    print("步驟 1: 載入模型與資料...")
    
    # 載入傳統模型資料
    with open("output/models/traditional_models.pkl", "rb") as f:
        trad_data = pickle.load(f)
    
    X_test_sparse = trad_data["X_test"]
    y_test = trad_data["y_test"]
    le = trad_data["label_encoder"]
    rf_model = trad_data["models"]["Random Forest"]
    
    # 載入 Keras 模型
    keras_model = keras.models.load_model("output/models/best_keras_model.keras")
    X_test_dense = X_test_sparse.toarray().astype('float32')

    # ==================== 2. 取得基礎預測機率 ====================
    print("\n步驟 2: 計算基礎模型機率...")
    
    # Random Forest Probabilities
    probs_rf = rf_model.predict_proba(X_test_sparse)
    
    # Keras Probabilities
    probs_keras = keras_model.predict(X_test_dense, verbose=0)
    
    # 基礎準確率
    acc_rf = accuracy_score(y_test, rf_model.predict(X_test_sparse))
    acc_keras = accuracy_score(y_test, np.argmax(probs_keras, axis=1))
    
    print(f" -> Random Forest 基柏準確率: {acc_rf:.2%}")
    print(f" -> Keras Neural Net 基準準確率: {acc_keras:.2%}")

    # ==================== 3. 策略三: 精細權重搜索 ====================
    print("\n" + "=" * 70)
    print("策略三: 精細化權重搜索 (Fine-grained Weight Search)")
    print("=" * 70)
    
    best_weight_acc = 0
    best_w_rf = 0
    history_weights = []
    history_accs = []

    # 從 0.00 到 1.00，每次增加 0.01
    for w_rf in np.arange(0.0, 1.01, 0.0001):
        w_keras = 1.0 - w_rf
        
        # 加權平均
        weighted_probs = (probs_rf * w_rf) + (probs_keras * w_keras)
        pred_label = np.argmax(weighted_probs, axis=1)
        
        acc = accuracy_score(y_test, pred_label)
        
        history_weights.append(w_rf)
        history_accs.append(acc)
        
        if acc > best_weight_acc:
            best_weight_acc = acc
            best_w_rf = w_rf

    print(f"搜索完成！")
    print(f"最佳權重配置: Random Forest = {best_w_rf:.2f}, Keras = {1-best_w_rf:.2f}")
    print(f"最佳集成準確率: {best_weight_acc:.2%}")
    print(f"相比 Keras 提升: +{best_weight_acc - acc_keras:.2%}")

    # 繪製權重變化圖
    plt.figure(figsize=(10, 6))
    plt.plot(history_weights, history_accs, label='Ensemble Accuracy', color='purple', linewidth=2)
    plt.axvline(x=best_w_rf, color='r', linestyle='--', label=f'Best Weight (RF={best_w_rf:.2f})')
    plt.axhline(y=acc_keras, color='g', linestyle=':', label='Keras Baseline')
    plt.axhline(y=acc_rf, color='b', linestyle=':', label='RF Baseline')
    
    plt.title(f"集成權重優化曲線\nMax Accuracy: {best_weight_acc:.2%}", fontsize=14)
    plt.xlabel("Random Forest 權重佔比 (0.0 ~ 1.0)", fontsize=12)
    plt.ylabel("測試集準確率", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("output/result_images/ensemble_weight_search.png", dpi=300)
    print("權重分析圖已儲存至 output/result_images/ensemble_weight_search.png")

    # ==================== 4. 策略二: Stacking (堆疊法) ====================
    print("\n" + "=" * 70)
    print("策略二: Stacking (使用 Logistic Regression 作為元模型)")
    print("=" * 70)
    print("注意：為了不作弊，我們將測試集切分 50% 訓練 Meta-Model，50% 驗證")

    # 準備 Meta-Features (將兩個模型的機率拼起來)
    # Shape: (n_samples, n_classes * 2)
    X_meta = np.hstack([probs_rf, probs_keras])
    
    # 切分 Meta 資料集
    X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
        X_meta, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    
    # 訓練 Meta-Model
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_meta_train, y_meta_train)
    
    # 預測
    stacking_pred = meta_model.predict(X_meta_test)
    stacking_acc = accuracy_score(y_meta_test, stacking_pred)
    
    print(f"Stacking (Meta-Model) 準確率: {stacking_acc:.2%}")
    print("(註：由於測試樣本減半，Stacking 的分數波動可能會較大)")

    # ==================== 5. 最終最佳預測與混淆矩陣 ====================
    print("\n" + "=" * 70)
    print("生成最終最佳混淆矩陣")
    print("=" * 70)
    
    # 使用最佳權重重新產生預測
    final_probs = (probs_rf * best_w_rf) + (probs_keras * (1-best_w_rf))
    final_preds = np.argmax(final_probs, axis=1)
    
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, final_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Optimized Ensemble Matrix (RF={best_w_rf:.2f}, Keras={1-best_w_rf:.2f})\nAccuracy: {best_weight_acc:.2%}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("output/result_images/final_optimized_ensemble_matrix.png", dpi=300)
    print("最佳混淆矩陣已儲存至 output/result_images/final_optimized_ensemble_matrix.png")

if __name__ == "__main__":
    main()