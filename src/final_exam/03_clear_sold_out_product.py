import pandas as pd

# 讀取資料
df = pd.read_csv("output/savesafe_products_20251207_143149.csv", encoding="utf-8-sig")

# 過濾掉售完商品
df_clean = df[~df["image_url"].str.contains("Sold_Out", na=False)].copy()

print(f"原始商品數: {len(df)}")
print(f"過濾後商品數: {len(df_clean)}")
print(f"移除了 {len(df) - len(df_clean)} 個售完商品")

# 儲存清理後的資料
df_clean.to_csv(
    "output/savesafe_cleaned_products_20251207_143149.csv",
    index=False,
    encoding="utf-8-sig",
)
