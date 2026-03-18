import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import time
import os

print("=" * 70)
print("TF-IDF 參數自動調優")
print("=" * 70)

# ==================== 1. 讀取資料 ====================
print("\n[步驟 1] 讀取資料...")
csv_path = "output/savesafe_cleaned_products_20251207_143149.csv"

if not os.path.exists(csv_path):
    print(f"錯誤: 找不到檔案 {csv_path}")
    exit(1)

df = pd.read_csv(csv_path, encoding="utf-8-sig")

# 準備文字欄位
df["text_brand_name"] = (
    df["brand"].fillna("") + " " + df["name"].fillna("")
).str.strip()
df["text_desc"] = df["description_detail"].fillna("")

# 準備標籤
le = LabelEncoder()
y = le.fit_transform(df["category"])

print(f"總數據筆數: {len(df):,}")
print(f"類別數量: {len(le.classes_)}")

# ==================== 2. 分析詞彙庫規模 ====================
print("\n" + "=" * 70)
print("[步驟 2] 分析詞彙庫規模...")
print("=" * 70)

# 不限制 max_features，查看總詞彙數
print("\n正在掃描品牌+名稱詞彙...")
tfidf_full_name = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.8)
tfidf_full_name.fit(df["text_brand_name"])
vocab_size_name = len(tfidf_full_name.vocabulary_)

print("正在掃描商品描述詞彙...")
tfidf_full_desc = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.8)
tfidf_full_desc.fit(df["text_desc"])
vocab_size_desc = len(tfidf_full_desc.vocabulary_)

print(f"\n詞彙庫統計:")
print(f"  品牌+名稱: {vocab_size_name:,} 個不重複 n-gram")
print(f"  商品描述:  {vocab_size_desc:,} 個不重複 n-gram")

# 顯示一些詞彙範例
vocab_list_name = list(tfidf_full_name.vocabulary_.keys())
print(f"\n品牌+名稱詞彙範例 (前 20 個):")
print(f"  {vocab_list_name[:20]}")

# ==================== 3. 測試品牌+名稱的最佳 max_features ====================
print("\n" + "=" * 70)
print("[步驟 3] 測試品牌+名稱的最佳 max_features...")
print("=" * 70)

# 根據詞彙庫大小動態設定測試範圍
name_options = [
    int(vocab_size_name * 0.25),  # 820
    int(vocab_size_name * 0.30),  # 984
    int(vocab_size_name * 0.35),  # 1,148
    int(vocab_size_name * 0.40),  # 1,312
    int(vocab_size_name * 0.45),  # 1,476
    int(vocab_size_name * 0.50),  # 1,640
    int(vocab_size_name * 0.55),  # 1,804
    int(vocab_size_name * 0.60),  # 1,968
    int(vocab_size_name * 0.65),  # 2,132
    int(vocab_size_name * 0.70),  # 2,296
]

print(f"\n測試範圍: {name_options}")
print("(使用 3-fold 交叉驗證評估)\n")

results_name = {}
best_score_name = 0
best_mf_name = None

for mf in name_options:
    print(f"測試 max_features={mf:,}...")
    start_time = time.time()

    # 生成特徵
    tfidf_name = TfidfVectorizer(
        max_features=mf, ngram_range=(1, 3), min_df=3, max_df=0.8, sublinear_tf=True
    )
    X_name = tfidf_name.fit_transform(df["text_brand_name"])

    # 使用 Logistic Regression 快速評估
    lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(lr, X_name, y, cv=cv, scoring="accuracy", n_jobs=-1)

    mean_score = scores.mean()
    std_score = scores.std()
    elapsed = time.time() - start_time

    results_name[mf] = mean_score

    print(f"  準確率: {mean_score:.4f} (±{std_score:.4f})")
    print(f"  耗時:   {elapsed:.1f} 秒\n")

    if mean_score > best_score_name:
        best_score_name = mean_score
        best_mf_name = mf

print(f">> 品牌+名稱最佳配置: max_features={best_mf_name:,}")
print(f"   準確率: {best_score_name:.4f}")

# ==================== 4. 測試商品描述的最佳 max_features ====================
print("\n" + "=" * 70)
print("[步驟 4] 測試商品描述的最佳 max_features...")
print("=" * 70)

desc_options = [
    int(vocab_size_desc * 0.10),  # 923
    int(vocab_size_desc * 0.12),  # 1,107
    int(vocab_size_desc * 0.15),  # 1,384
    int(vocab_size_desc * 0.18),  # 1,661
    int(vocab_size_desc * 0.20),  # 1,846
    int(vocab_size_desc * 0.25),  # 2,308
    int(vocab_size_desc * 0.30),  # 2,769
]

print(f"\n測試範圍: {desc_options}")
print("(使用 3-fold 交叉驗證評估)\n")

results_desc = {}
best_score_desc = 0
best_mf_desc = None

for mf in desc_options:
    print(f"測試 max_features={mf:,}...")
    start_time = time.time()

    # 生成特徵
    tfidf_desc = TfidfVectorizer(
        max_features=mf, ngram_range=(1, 3), min_df=3, max_df=0.8, sublinear_tf=True
    )
    X_desc = tfidf_desc.fit_transform(df["text_desc"])

    # 交叉驗證
    lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(lr, X_desc, y, cv=cv, scoring="accuracy", n_jobs=-1)

    mean_score = scores.mean()
    std_score = scores.std()
    elapsed = time.time() - start_time

    results_desc[mf] = mean_score

    print(f"  準確率: {mean_score:.4f} (±{std_score:.4f})")
    print(f"  耗時:   {elapsed:.1f} 秒\n")

    if mean_score > best_score_desc:
        best_score_desc = mean_score
        best_mf_desc = mf

print(f">> 商品描述最佳配置: max_features={best_mf_desc:,}")
print(f"   準確率: {best_score_desc:.4f}")

# ==================== 5. 組合測試 ====================
print("\n" + "=" * 70)
print("[步驟 5] 測試組合效果...")
print("=" * 70)

print(f"\n使用最佳組合: name={best_mf_name:,}, desc={best_mf_desc:,}")
start_time = time.time()

# 生成組合特徵
tfidf_name = TfidfVectorizer(
    max_features=best_mf_name,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.8,
    sublinear_tf=True,
)
tfidf_desc = TfidfVectorizer(
    max_features=best_mf_desc,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.8,
    sublinear_tf=True,
)

X_name = tfidf_name.fit_transform(df["text_brand_name"])
X_desc = tfidf_desc.fit_transform(df["text_desc"])
X_combined = hstack([X_name, X_desc])

# 交叉驗證
lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(lr, X_combined, y, cv=cv, scoring="accuracy", n_jobs=-1)

mean_score = scores.mean()
std_score = scores.std()
elapsed = time.time() - start_time

print(f"\n組合特徵準確率: {mean_score:.4f} (±{std_score:.4f})")
print(f"總特徵維度: {X_combined.shape[1]:,}")
print(f"耗時: {elapsed:.1f} 秒")

# ==================== 6. 保存結果 ====================
print("\n" + "=" * 70)
print("[步驟 6] 保存結果...")
print("=" * 70)

results_df = pd.DataFrame(
    {
        "配置項": ["品牌+名稱", "商品描述", "組合"],
        "max_features": [best_mf_name, best_mf_desc, best_mf_name + best_mf_desc],
        "準確率": [best_score_name, best_score_desc, mean_score],
    }
)

output_csv = "output/tfidf_tuning_results.csv"
results_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"結果已保存至: {output_csv}")

# ==================== 7. 輸出建議代碼 ====================
print("\n" + "=" * 70)
print("調優完成！建議配置")
print("=" * 70)

print(
    f"""
 brand + name 的 max_features={best_mf_name}
 description 的 max_features={best_mf_desc}
"""
)

print("=" * 70)
print(f"預期文字特徵維度: {best_mf_name + best_mf_desc:,} 維")
print(f"預期準確率提升: {mean_score:.4f} (僅文字特徵)")
print("=" * 70)
