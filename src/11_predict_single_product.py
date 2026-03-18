"""
單一商品預測器
輸入圖片路徑、商品名稱、描述進行分類預測
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pickle
from PIL import Image
import cv2
from skimage.feature import hog
import keras
from scipy.sparse import hstack, csr_matrix

# 設定中文字體
from utils.cross_platform_config import set_matplotlib_font

font_name = set_matplotlib_font()
print("使用字型：", font_name)


class ProductClassifier:
    """商品分類預測器"""

    def __init__(self):
        """載入所有需要的模型與編碼器"""
        print("=" * 70)
        print("初始化商品分類預測器")
        print("=" * 70)

        # 1. 載入特徵處理器
        features_file = "output/processed_features.pkl"
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"找不到特徵處理檔案: {features_file}")

        with open(features_file, "rb") as f:
            feature_data = pickle.load(f)

        self.tfidf_name = feature_data["tfidf_name"]
        self.tfidf_desc = feature_data["tfidf_desc"]
        self.scaler_img = feature_data["scaler_img"]
        self.scaler_price = feature_data["scaler_price"]
        self.label_encoder = feature_data["label_encoder"]

        print("[OK] 特徵處理器載入完成")

        # 2. 載入最佳模型
        best_model_file = "output/models/final_best_model_info.pkl"

        if os.path.exists(best_model_file):
            with open(best_model_file, "rb") as f:
                model_info = pickle.load(f)

            self.model_name = model_info["model_name"]
            print(f"[OK] 使用最佳模型: {self.model_name}")

            # 判斷是傳統模型還是神經網路
            if "Neural" in self.model_name:
                # Keras 模型
                keras_model_path = "output/models/best_keras_model.keras"
                self.model = keras.models.load_model(keras_model_path)
                self.model_type = "keras"
                print("[OK] Keras 神經網路模型載入完成")
            else:
                # 傳統機器學習模型
                self.model = model_info["model"]
                self.model_type = "traditional"
                print("[OK] 傳統機器學習模型載入完成")
        else:
            # 備用方案：直接載入 Keras 模型
            print("[WARNING] 找不到最佳模型資訊，嘗試載入 Keras 模型...")
            keras_results_file = "output/models/keras_results.pkl"

            with open(keras_results_file, "rb") as f:
                keras_data = pickle.load(f)

            keras_model_path = "output/models/best_keras_model.keras"
            self.model = keras.models.load_model(keras_model_path)
            self.model_type = "keras"
            self.model_name = "Neural Network"
            self.label_encoder = keras_data["label_encoder"]
            print("[OK] Keras 模型載入完成")

        print("\n分類類別：")
        for i, category in enumerate(self.label_encoder.classes_):
            print(f"  {i}. {category}")

        print("\n預測器初始化完成！")
        print("=" * 70)

    def extract_image_features(self, image_path, target_size=(500, 500)):
        """
        提取圖片特徵（RGB直方圖 + HSV直方圖 + HOG）

        Args:
            image_path: 圖片檔案路徑
            target_size: 調整後的圖片尺寸

        Returns:
            np.ndarray: 圖片特徵向量
        """
        try:
            # 讀取並調整圖片大小
            img = Image.open(image_path).convert("RGB")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img)

            # 1. RGB 色彩直方圖 (每通道 32 bins)
            rgb_hist = []
            for i in range(3):
                hist, _ = np.histogram(img_array[:, :, i], bins=32, range=(0, 256))
                rgb_hist.extend(hist)

            # 2. HSV 色彩直方圖
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            hsv_hist = []
            for i in range(3):
                hist, _ = np.histogram(img_hsv[:, :, i], bins=16, range=(0, 256))
                hsv_hist.extend(hist)

            # 3. HOG 特徵
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            hog_features = hog(
                img_gray,
                orientations=9,
                pixels_per_cell=(100, 100),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                feature_vector=True,
            )

            # 合併所有特徵
            features = np.concatenate([rgb_hist, hsv_hist, hog_features])

            return features

        except Exception as e:
            print(f"[ERROR] 圖片特徵提取失敗: {e}")
            # 返回零向量
            return np.zeros(720)

    def prepare_features(
        self, brand="", name="", description="", image_path=None, price=0
    ):
        """
        準備預測所需的完整特徵向量

        Args:
            brand: 品牌名稱
            name: 商品名稱
            description: 商品描述
            image_path: 圖片路徑（可選）
            price: 價格（可選）

        Returns:
            csr_matrix: 稀疏矩陣格式的特徵向量
        """
        # 1. 文字特徵 - 品牌+名稱
        text_brand_name = f"{brand} {name}".strip()
        X_text_name = self.tfidf_name.transform([text_brand_name])

        # 2. 文字特徵 - 描述
        X_text_desc = self.tfidf_desc.transform([description])

        # 3. 圖片特徵
        if image_path and os.path.exists(image_path):
            img_features = self.extract_image_features(image_path)
        else:
            print("[WARNING] 未提供圖片或圖片不存在，使用零向量")
            img_features = np.zeros(720)

        # 標準化圖片特徵
        img_features_scaled = self.scaler_img.transform(img_features.reshape(1, -1))

        # 4. 價格特徵
        price_scaled = self.scaler_price.transform([[price]])

        # 5. 合併所有特徵
        X = hstack(
            [
                X_text_name,  # 品牌+名稱 TF-IDF
                X_text_desc,  # 描述 TF-IDF
                csr_matrix(img_features_scaled),  # 圖片特徵
                csr_matrix(price_scaled),  # 價格
            ]
        )

        return X

    def predict(
        self,
        brand="",
        name="",
        description="",
        image_path=None,
        price=0,
        show_probabilities=True,
    ):
        """
        預測商品分類

        Args:
            brand: 品牌名稱
            name: 商品名稱
            description: 商品描述
            image_path: 圖片路徑
            price: 價格
            show_probabilities: 是否顯示各類別機率

        Returns:
            tuple: (預測類別, 信心分數, 所有類別機率)
        """
        print("\n" + "=" * 70)
        print("開始預測")
        print("=" * 70)

        # 準備特徵
        X = self.prepare_features(brand, name, description, image_path, price)

        # 預測
        if self.model_type == "keras":
            # Keras 模型需要轉換為 dense array
            X_dense = X.toarray().astype("float32")
            probs = self.model.predict(X_dense, verbose=0)[0]
            pred_idx = np.argmax(probs)
        else:
            # 傳統模型
            pred_idx = self.model.predict(X)[0]

            # 獲取機率（如果模型支援）
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
            else:
                probs = np.zeros(len(self.label_encoder.classes_))
                probs[pred_idx] = 1.0

        # 解碼預測結果
        predicted_category = self.label_encoder.classes_[pred_idx]
        confidence = probs[pred_idx]

        # 顯示結果
        print(f"\n預測結果: {predicted_category}")
        print(f"信心分數: {confidence:.2%}")

        if show_probabilities:
            print("\n各類別機率：")
            # 按機率排序
            sorted_indices = np.argsort(probs)[::-1]
            for idx in sorted_indices:
                category = self.label_encoder.classes_[idx]
                prob = probs[idx]
                bar = "█" * int(prob * 50)
                print(f"  {category:20s} {prob:6.2%} {bar}")

        print("=" * 70)

        return predicted_category, confidence, probs


def main():
    """示範使用方式"""

    # 初始化預測器
    classifier = ProductClassifier()

    # 範例 1: 完整資訊預測
    print("\n\n範例 1: 完整資訊預測")
    print("-" * 70)

    result = classifier.predict(
        brand="義美",
        name="義美純濃豆奶 946ml",
        description="使用非基因改造黃豆，純淨自然，營養豐富",
        image_path="output/images/example.jpg",  # 請替換為實際圖片路徑
        price=45,
        show_probabilities=True,
    )

    # 範例 2: 僅文字預測（無圖片）
    print("\n\n範例 2: 僅文字預測")
    print("-" * 70)

    result = classifier.predict(
        brand="",
        name="五木拉麵",
        description="日式拉麵，口感Q彈，適合搭配各式湯頭",
        image_path=None,
        price=35,
        show_probabilities=False,
    )

    # 範例 3: 互動式輸入
    print("\n\n範例 3: 互動式輸入")
    print("-" * 70)

    user_input = input("\n是否要手動輸入商品資訊進行預測？(y/n): ")

    if user_input.lower() == "y":
        brand = input("品牌名稱: ")
        name = input("商品名稱: ")
        description = input("商品描述: ")
        image_path = input("圖片路徑 (可留空): ").strip()
        price_str = input("價格 (可留空): ").strip()

        price = float(price_str) if price_str else 0
        image_path = image_path if image_path else None

        classifier.predict(
            brand=brand,
            name=name,
            description=description,
            image_path=image_path,
            price=price,
            show_probabilities=True,
        )


def predict_from_csv(csv_path, output_path="output/predictions_result.csv"):
    """
    從 CSV 讀取商品資料並進行批次預測

    Args:
        csv_path: 輸入 CSV 檔案路徑
        output_path: 輸出結果 CSV 檔案路徑

    CSV 必要欄位:
        - brand (品牌)
        - name (商品名稱)
        - description 或 description_detail (商品描述)
        - price (價格，可選，預設0)
        - image_path (圖片路徑，可選)
    """
    import pandas as pd
    from tqdm import tqdm

    print("=" * 70)
    print("批次預測：從 CSV 讀取商品資料")
    print("=" * 70)

    # 讀取 CSV
    if not os.path.exists(csv_path):
        print(f"錯誤: 找不到檔案 {csv_path}")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"\n讀取商品數量: {len(df)}")
    print(f"欄位: {df.columns.tolist()}")

    # 處理空值（與 06_prepare_features.py 相同方式）
    df["brand"] = df["brand"].fillna("")
    df["name"] = df["name"].fillna("")

    # 描述欄位處理
    if "description_detail" in df.columns:
        df["description"] = df["description_detail"].fillna("")
    elif "description" in df.columns:
        df["description"] = df["description"].fillna("")
    else:
        df["description"] = ""

    # 價格處理
    df["price"] = df["price"].fillna(0)

    # 圖片路徑處理
    df["image_path"] = df["image_path"].fillna("")

    # 初始化預測器
    classifier = ProductClassifier()

    # 準備結果列表
    results = []

    # 批次預測
    print("\n開始批次預測...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="預測進度"):
        brand = row["brand"]
        name = row["name"]
        description = row["description"]
        price = float(row["price"]) if row["price"] else 0

        # 圖片路徑處理：處理相對路徑和絕對路徑
        image_path_raw = row["image_path"].strip() if row["image_path"] else ""

        if image_path_raw:
            # 檢查是否為絕對路徑
            if os.path.isabs(image_path_raw):
                image_path = image_path_raw
            else:
                # 相對路徑：相對於程式執行位置
                image_path = os.path.abspath(image_path_raw)

            # 驗證圖片是否存在
            if not os.path.exists(image_path):
                print(f"\n[警告] 圖片不存在: {image_path}")
                print(f"  原始路徑: {image_path_raw}")
                print(f"  當前工作目錄: {os.getcwd()}")
                image_path = None
        else:
            image_path = None

        # 進行預測（關閉詳細輸出）
        try:
            pred, conf, probs = classifier.predict(
                brand=brand,
                name=name,
                description=description,
                image_path=image_path,
                price=price,
                show_probabilities=False,
            )

            # 儲存結果
            results.append(
                {
                    "brand": brand,
                    "name": name,
                    "predicted_category": pred,
                    "confidence": conf,
                    "top2_category": classifier.label_encoder.classes_[
                        np.argsort(probs)[-2]
                    ],
                    "top2_confidence": np.sort(probs)[-2],
                }
            )

        except Exception as e:
            print(f"\n預測失敗 (商品: {brand} {name}): {e}")
            import traceback

            traceback.print_exc()

            results.append(
                {
                    "brand": brand,
                    "name": name,
                    "predicted_category": "ERROR",
                    "confidence": 0.0,
                    "top2_category": "",
                    "top2_confidence": 0.0,
                }
            )

    # 轉換為 DataFrame
    result_df = pd.DataFrame(results)

    # 儲存結果
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[完成] 預測完成！結果已儲存至: {output_path}")

    # 統計資訊
    print("\n" + "=" * 70)
    print("預測結果統計")
    print("=" * 70)
    print(f"總預測數量: {len(result_df)}")

    if len(result_df) > 0:
        print(f"平均信心分數: {result_df['confidence'].mean():.2%}")

    print(f"\n各類別預測數量:")
    print(result_df["predicted_category"].value_counts())

    print(f"\n信心分數分佈:")
    print(f"  高信心 (>90%): {(result_df['confidence'] > 0.90).sum()} 筆")
    print(
        f"  中信心 (60-90%): {((result_df['confidence'] >= 0.60) & (result_df['confidence'] <= 0.90)).sum()} 筆"
    )
    print(f"  低信心 (<60%): {(result_df['confidence'] < 0.60).sum()} 筆")

    return result_df


if __name__ == "__main__":

    predict_from_csv(
        csv_path="input/11_predict_single_product.csv",
        output_path="output/predictions_result.csv",
    )
