import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import pickle
import os

import matplotlib.pyplot as plt
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
    print("步驟 1: 讀取資料")
    print("-" * 70)

    # 讀取 CSV
    csv_path = "output/savesafe_cleaned_products_20251207_143149.csv"
    if not os.path.exists(csv_path):
        print(f"錯誤: 找不到檔案 {csv_path}")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    print(f"\n欄位檢查: {df.columns.tolist()[:8]}")

    # 讀取圖片特徵
    img_features_path = "output/image_features_500.npy"
    if not os.path.exists(img_features_path):
        print(f"錯誤: 找不到檔案 {img_features_path}")
        return

    image_features = np.load(img_features_path)

    print(f"\n商品數: {len(df)}")
    print(f"圖片特徵維度: {image_features.shape}")

    # ==================== 2. 文字特徵處理 ====================
    print("\n" + "=" * 70)
    print("步驟 2: 文字特徵提取 (TF-IDF)")
    print("=" * 70)

    # 處理空值 (brand=品牌, name=商品名稱, description=詳細描述)
    df["text_brand_name"] = (
        df["brand"].fillna("") + " " + df["name"].fillna("")
    ).str.strip()
    df["text_desc"] = df["description_detail"].fillna("")

    # TF-IDF 向量化 - brand + name
    print("\n處理品牌+商品名稱...")
    tfidf_name = TfidfVectorizer(
        max_features=2296, ngram_range=(1, 3), max_df=0.8, sublinear_tf=True
    )
    X_text_name = tfidf_name.fit_transform(df["text_brand_name"])
    print(f"品牌+名稱特徵維度: {X_text_name.shape}")

    # TF-IDF 向量化 - description
    print("處理商品描述...")
    tfidf_desc = TfidfVectorizer(
        max_features=2769, ngram_range=(1, 3), max_df=0.8, sublinear_tf=True
    )
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

    # ==================== 4. 圖片特徵標準化 ====================
    print("\n" + "=" * 70)
    print("步驟 4: 圖片特徵標準化")
    print("=" * 70)

    scaler_img = StandardScaler()
    image_features_scaled = scaler_img.fit_transform(image_features)
    print(f"圖片特徵已標準化")

    # ==================== 5. 標籤編碼 ====================
    print("\n" + "=" * 70)
    print("步驟 5: 目標變數編碼")
    print("=" * 70)

    le = LabelEncoder()
    y = le.fit_transform(df["category"])

    print(f"\n類別數: {len(le.classes_)}")
    print("各分類商品數量:")
    for i, cat in enumerate(le.classes_):
        count = (y == i).sum()
        percentage = count / len(df) * 100
        print(f"  {i}. {cat:20s} {count:4d} 筆 ({percentage:5.1f}%)")

    # ==================== 6. 合併所有特徵 ====================
    print("\n" + "=" * 70)
    print("步驟 6: 合併所有特徵")
    print("=" * 70)

    X = hstack(
        [
            X_text_name,  # 500 維
            X_text_desc,  # 500 維
            csr_matrix(image_features_scaled),  # 3012 維
            csr_matrix(X_price),  # 1 維
        ]
    )

    print(f"最終特徵維度: {X.shape}")
    print(f"  - 文字 (brand+name):  2296 維")
    print(f"  - 文字 (description): 2769 維")
    print(f"  - 圖片 (500x500):    720 維")
    print(f"  - 價格:                 1 維")
    print(f"  - 總計:              5786 維")

    # ==================== 7. 儲存處理後的資料 ====================
    print("\n" + "=" * 70)
    print("步驟 7: 儲存處理後的資料")
    print("=" * 70)

    output_file = "output/processed_features.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(
            {
                "X": X,
                "y": y,
                "label_encoder": le,
                "tfidf_name": tfidf_name,
                "tfidf_desc": tfidf_desc,
                "scaler_img": scaler_img,
                "scaler_price": scaler_price,
                "feature_info": {
                    "name_dim": X_text_name.shape[1],
                    "desc_dim": X_text_desc.shape[1],
                    "img_dim": image_features.shape[1],
                    "price_dim": X_price.shape[1],
                },
            },
            f,
        )
    print(f"處理完成！資料已儲存至 {output_file}")


if __name__ == "__main__":
    main()
