import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

warnings.filterwarnings("ignore")

# 設定中文字體
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 70)
print("基於多模態特徵的商品自動分類系統")
print("=" * 70)

# ==================== 1. 讀取資料 ====================
print("\n步驟 1: 讀取資料")
print("-" * 70)

df = pd.read_csv(
    "output/savesafe_clean_sold_out_product_20251114_103512.csv", encoding="utf-8-sig"
)
image_features = np.load("output/image_features_500.npy")

print(f"商品數: {len(df)}")
print(f"圖片特徵維度: {image_features.shape}")

# ==================== 2. 文字特徵處理 ====================
print("\n" + "=" * 70)
print("步驟 2: 文字特徵提取 (TF-IDF)")
print("=" * 70)

# 處理空值
df["text_name"] = df["name"].fillna("")
df["text_desc"] = (
    df["description"].fillna("") + " " + df["description_detail"].fillna("")
).str.strip()

# TF-IDF 向量化 - name
print("\n處理商品名稱...")
tfidf_name = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_text_name = tfidf_name.fit_transform(df["text_name"])
print(f"name 特徵維度: {X_text_name.shape}")

# TF-IDF 向量化 - description
print("處理商品描述...")
tfidf_desc = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_text_desc = tfidf_desc.fit_transform(df["text_desc"])
print(f"description 特徵維度: {X_text_desc.shape}")

# ==================== 3. 數值特徵 ====================
print("\n" + "=" * 70)
print("步驟 3: 價格特徵標準化")
print("=" * 70)

scaler_price = StandardScaler()
X_price = scaler_price.fit_transform(df[["price"]].values)
print(f"價格特徵維度: {X_price.shape}")
print(f"價格範圍: {df['price'].min():.0f} - {df['price'].max():.0f} 元")
print(f"平均價格: {df['price'].mean():.0f} 元")

# ==================== 4. 標籤編碼 ====================
print("\n" + "=" * 70)
print("步驟 4: 目標變數編碼")
print("=" * 70)

le = LabelEncoder()
y = le.fit_transform(df["category"])

print(f"\n類別數: {len(le.classes_)}")
print("各分類商品數量:")
for i, cat in enumerate(le.classes_):
    count = (y == i).sum()
    percentage = count / len(df) * 100
    print(f"  {i}. {cat:20s} {count:4d} 筆 ({percentage:5.1f}%)")

# ==================== 5. 分割訓練/測試集 ====================
print("\n" + "=" * 70)
print("步驟 5: 分割訓練/測試集")
print("=" * 70)

train_idx, test_idx = train_test_split(
    range(len(df)), test_size=0.2, stratify=y, random_state=42
)

# 文字特徵
X_name_train = X_text_name[train_idx]
X_name_test = X_text_name[test_idx]
X_desc_train = X_text_desc[train_idx]
X_desc_test = X_text_desc[test_idx]

# 圖片特徵標準化
print("標準化圖片特徵...")
scaler_img = StandardScaler()
image_features_scaled = scaler_img.fit_transform(image_features)
X_img_train = image_features_scaled[train_idx]
X_img_test = image_features_scaled[test_idx]

# 價格特徵
X_price_train = X_price[train_idx]
X_price_test = X_price[test_idx]

# 目標變數
y_train = y[train_idx]
y_test = y[test_idx]

print(f"訓練集: {len(y_train)} 筆 ({len(y_train)/len(df)*100:.1f}%)")
print(f"測試集: {len(y_test)} 筆 ({len(y_test)/len(df)*100:.1f}%)")

# ==================== 6. 合併所有特徵 ====================
print("\n" + "=" * 70)
print("步驟 6: 合併所有特徵")
print("=" * 70)

X_train = hstack(
    [
        X_name_train,  # 500 維
        X_desc_train,  # 500 維
        csr_matrix(X_img_train),  # 3012 維
        csr_matrix(X_price_train),  # 1 維
    ]
)

X_test = hstack(
    [X_name_test, X_desc_test, csr_matrix(X_img_test), csr_matrix(X_price_test)]
)

print(f"最終特徵維度: {X_train.shape}")
print(f"  - 文字 (name):        500 維")
print(f"  - 文字 (description): 500 維")
print(f"  - 圖片 (500x500):    3012 維")
print(f"  - 價格:                 1 維")
print(f"  - 總計:              4013 維")

# ==================== 7. 訓練多個模型 ====================
print("\n" + "=" * 70)
print("步驟 7: 訓練模型")
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
best_model = None
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
        classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = (name, model)
        best_y_pred = y_pred

# ==================== 8. 儲存最佳模型 ====================
print(f"\n" + "=" * 70)
print(f"最佳模型: {best_model[0]}")
print(f"準確率: {best_accuracy:.2%}")
print("=" * 70)

with open("output/best_model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": best_model[1],
            "model_name": best_model[0],
            "accuracy": best_accuracy,
            "tfidf_name": tfidf_name,
            "tfidf_desc": tfidf_desc,
            "scaler_img": scaler_img,
            "scaler_price": scaler_price,
            "label_encoder": le,
        },
        f,
    )
print("模型已儲存到 output/best_model.pkl")

# ==================== 9. 混淆矩陣視覺化 ====================
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
    f"混淆矩陣 - {best_model[0]}\n準確率: {best_accuracy:.2%}", fontsize=16, pad=20
)
plt.xlabel("預測分類", fontsize=12)
plt.ylabel("實際分類", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("output/confusion_matrix.png", dpi=300, bbox_inches="tight")
print("混淆矩陣已儲存到 output/confusion_matrix.png")

# ==================== 10. 模型比較圖 ====================
print("生成模型比較圖...")
plt.figure(figsize=(10, 6))
colors = ["#3498db", "#2ecc71", "#e74c3c"]
bars = plt.bar(results.keys(), results.values(), color=colors[: len(results)])
plt.ylabel("準確率", fontsize=12)
plt.title("不同模型準確率比較", fontsize=16)
plt.ylim(0, 1)
for i, (name, acc) in enumerate(results.items()):
    plt.text(i, acc + 0.01, f"{acc:.2%}", ha="center", fontsize=11, fontweight="bold")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("output/model_comparison.png", dpi=300, bbox_inches="tight")
print("模型比較圖已儲存到 output/model_comparison.png")

print("\n" + "=" * 70)
print("訓練完成！")
print("=" * 70)
print(f"\n生成的檔案:")
print(f"  - output/best_model.pkl (訓練好的模型)")
print(f"  - output/confusion_matrix.png (混淆矩陣)")
print(f"  - output/model_comparison.png (模型比較)")
