import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
from tqdm import tqdm
import time

df = pd.read_csv("output/savesafe_products_20251207_143149.csv", encoding="utf-8-sig")

# 建立圖片資料夾
os.makedirs("output/images", exist_ok=True)

# 下載圖片
success_count = 0
failed_urls = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="下載圖片"):
    try:
        img_url = row["image_url"]
        sku = row["sku"]

        response = requests.get(img_url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(f"output/images/{sku}.jpg")
            success_count += 1
        else:
            failed_urls.append(img_url)
    except Exception as e:
        print(f"\nFailed: {img_url}, Error: {e}")
        failed_urls.append(img_url)

    # 每 50 張休息一下
    if (idx + 1) % 50 == 0:
        time.sleep(1)

print(f"\n下載完成: {success_count}/{len(df)}")
print(f"失敗數量: {len(failed_urls)}")

# 儲存失敗清單
if failed_urls:
    with open("failed_downloads.txt", "w") as f:
        f.write("\n".join(failed_urls))
