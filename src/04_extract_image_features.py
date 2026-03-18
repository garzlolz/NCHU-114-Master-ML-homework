import pandas as pd
import numpy as np
from PIL import Image
import cv2
from skimage.feature import hog
from tqdm import tqdm
import os


# ======================================================================
# 圖片特徵提取（RGB + HSV + HOG）
# CSV 欄位: brand(品牌), name(商品名稱), description(詳細描述)
# ======================================================================


# 讀取資料
df = pd.read_csv(
    "output/savesafe_cleaned_products_20251207_143149.csv", encoding="utf-8-sig"
)
print(f"總商品數: {len(df)}")


def extract_image_features_500(sku):
    """提取圖片特徵 - 500x500 原始解析度（RGB + HSV + HOG）"""
    img_path = f"output/images/{sku}.jpg"

    if not os.path.exists(img_path):
        return None, False

    try:
        # 使用原始 500x500 解析度（不調整大小）
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        # 確保是 500x500
        if img_array.shape[:2] != (500, 500):
            img = img.resize((500, 500))
            img_array = np.array(img)

        # ==================== 特徵 1: RGB 顏色直方圖（32 bins × 3 = 96 維）====================
        hist_r = cv2.calcHist([img_array], [0], None, [32], [0, 256]).flatten()
        hist_g = cv2.calcHist([img_array], [1], None, [32], [0, 256]).flatten()
        hist_b = cv2.calcHist([img_array], [2], None, [32], [0, 256]).flatten()
        rgb_features = np.concatenate([hist_r, hist_g, hist_b])  # 96 維

        # ==================== 特徵 2: HSV 顏色直方圖（16 bins × 3 = 48 維）====================
        # 轉換為 HSV 色彩空間（對光照變化更穩健）
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # H (色相): 0-180，使用 16 bins
        hist_h = cv2.calcHist([img_hsv], [0], None, [16], [0, 180]).flatten()

        # S (飽和度): 0-256，使用 16 bins
        hist_s = cv2.calcHist([img_hsv], [1], None, [16], [0, 256]).flatten()

        # V (明度): 0-256，使用 16 bins
        hist_v = cv2.calcHist([img_hsv], [2], None, [16], [0, 256]).flatten()

        hsv_features = np.concatenate([hist_h, hist_s, hist_v])  # 48 維

        # ==================== 特徵 3: HOG 特徵（方向梯度直方圖）====================
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hog_features = hog(
            gray,
            pixels_per_cell=(100, 100),  # 適應 500x500 (10×10 cells)
            cells_per_block=(2, 2),  # 每個 block 包含 4 個 cells
            visualize=False,
            feature_vector=True,
        )

        # ==================== 合併所有特徵 ====================
        features = np.concatenate([rgb_features, hsv_features, hog_features])
        # RGB(96) + HSV(48) + HOG(約3000) = 約3144 維

        return features, True

    except Exception as e:
        print(f"\nError processing {sku}.jpg: {e}")
        return None, False


# 先測試一張圖片，確定維度
print("\n測試第一張圖片...")
test_sku = df["sku"].iloc[0]
test_features, test_success = extract_image_features_500(test_sku)

if test_success:
    TARGET_DIM = test_features.shape[0]
    print(f"  測試成功")
    print(f"  RGB 顏色特徵: 96 維 (32 bins × 3 通道)")
    print(f"  HSV 顏色特徵: 48 維 (16 bins × 3 通道)")
    print(f"  HOG 特徵: {TARGET_DIM - 144} 維")
    print(f"  總維度: {TARGET_DIM} 維")
else:
    print("  測試失敗，請檢查圖片路徑")
    exit(1)

# 提取所有圖片特徵
print("\n" + "=" * 70)
print("開始提取圖片特徵 (500x500，RGB + HSV + HOG)")
print(f"目標特徵維度: {TARGET_DIM}")
print("=" * 70)

image_features = []
success_count = 0
failed_skus = []

for sku in tqdm(df["sku"], desc="提取圖片特徵"):
    features, success = extract_image_features_500(sku)

    if success:
        # 確保維度一致
        if features.shape[0] != TARGET_DIM:
            print(
                f"\nWarning: SKU {sku} dim={features.shape[0]}, adjusting to {TARGET_DIM}"
            )
            if features.shape[0] < TARGET_DIM:
                features = np.pad(features, (0, TARGET_DIM - features.shape[0]))
            else:
                features = features[:TARGET_DIM]
        image_features.append(features)
        success_count += 1
    else:
        # 失敗用零向量
        image_features.append(np.zeros(TARGET_DIM))
        failed_skus.append(sku)

# 轉換為 numpy 陣列
image_features_array = np.array(image_features)

# 結果統計
print("\n" + "=" * 70)
print("提取結果統計")
print("=" * 70)
print(f"圖片特徵維度: {image_features_array.shape}")
print(f"  - RGB 直方圖: 96 維")
print(f"  - HSV 直方圖: 48 維")
print(f"  - HOG 特徵: {TARGET_DIM - 144} 維")
print(f"成功提取: {success_count}/{len(df)}")
print(f"失敗數量: {len(failed_skus)}")
print(f"成功率: {success_count/len(df)*100:.1f}%")

if failed_skus:
    print(f"\n缺失圖片的 SKU: {failed_skus}")

# 儲存特徵
output_path = "output/image_features_500.npy"
np.save(output_path, image_features_array)
print(f"\n 圖片特徵已儲存到 {output_path}")

# 驗證
print("\n" + "=" * 70)
print("驗證儲存的特徵")
print("=" * 70)
loaded = np.load(output_path)
print(f"載入的特徵維度: {loaded.shape}")
print(f"資料型態: {loaded.dtype}")
print(f"記憶體大小: {loaded.nbytes / 1024 / 1024:.2f} MB")

print(f"\n特徵統計:")
print(f"  最小值: {loaded.min():.4f}")
print(f"  最大值: {loaded.max():.4f}")
print(f"  平均值: {loaded.mean():.4f}")
print(f"  零向量數量: {(loaded.sum(axis=1) == 0).sum()}")

print("\n 特徵提取完成")
