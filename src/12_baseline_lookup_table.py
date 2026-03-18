# 07_baseline_lookup_table.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
import os

# 設定中文字體
from utils.cross_platform_config import set_matplotlib_font
font_name = set_matplotlib_font()

print("使用字型：", font_name)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False


def main():
    print("=" * 70)
    print("Baseline 方法: Lookup Table (關鍵字查表法)")
    print("=" * 70)

    # ==================== 1. 讀取資料 ====================
    print("\n步驟 1: 讀取資料")
    print("-" * 70)

    csv_path = "output/savesafe_cleaned_products_20251207_143149.csv"
    if not os.path.exists(csv_path):
        print(f"錯誤: 找不到檔案 {csv_path}")
        return

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"商品數: {len(df)}")

    # ==================== 統一欄位名稱 ====================
    # 將 description_detail 統一命名為 description
    if "description_detail" in df.columns and "description" not in df.columns:
        df["description"] = df["description_detail"]
        print("已將 'description_detail' 欄位重新命名為 'description'")

    # 檢查欄位
    print(f"\n欄位名稱: {df.columns.tolist()}")

    # 顯示類別
    categories = df["category"].unique()
    print(f"\n類別數: {len(categories)}")
    print("類別清單:")
    for i, cat in enumerate(sorted(categories)):
        count = (df["category"] == cat).sum()
        print(f"  {i+1}. {cat:20s} ({count} 筆)")

    # ==================== 2. 建立關鍵字規則表 ====================
    print("\n" + "=" * 70)
    print("步驟 2: 建立關鍵字規則表")
    print("=" * 70)

    # 根據你的類別設計關鍵字（需要根據實際類別調整）
    LOOKUP_RULES = {
        "米油罐頭泡麵": [
            "米",
            "白米",
            "香米",
            "糙米",
            "胚芽米",
            "油",
            "食用油",
            "橄欖油",
            "葵花油",
            "沙拉油",
            "罐頭",
            "鮪魚",
            "玉米",
            "肉醬",
            "泡麵",
            "速食麵",
            "科學麵",
            "王子麵",
            "拉麵",
            "烏龍麵",
            "麵",
            "義大利麵",
            "關東煮",
            "冬粉",
        ],
        "餅乾零食飲料": [
            "餅乾",
            "夾心餅",
            "蘇打餅",
            "威化餅",
            "消化餅",
            "零食",
            "洋芋片",
            "蝦味先",
            "乖乖",
            "可樂果",
            "飲料",
            "可樂",
            "汽水",
            "果汁",
            "茶",
            "奶茶",
            "咖啡",
            "糖果",
            "巧克力",
            "軟糖",
            "口香糖",
            "飲品",
            "運動飲料",
            "機能飲料",
        ],
        "奶粉養生保健": [
            "奶粉",
            "配方奶",
            "牛奶",
            "羊奶",
            "保健",
            "維他命",
            "維生素",
            "鈣",
            "益生菌",
            "膠原蛋白",
            "葡萄糖胺",
            "魚油",
            "養生",
            "補品",
            "麥片",
            "燕麥",
            "穀物",
        ],
        "沐浴開架保養": [
            "沐浴",
            "洗髮",
            "洗面",
            "洗手",
            "肥皂",
            "香皂",
            "洗髮精",
            "潤髮",
            "護髮",
            "面膜",
            "保養",
            "乳液",
            "化妝水",
            "精華液",
            "牙膏",
            "牙刷",
            "漱口",
        ],
        "餐廚衛浴居家": [
            "鍋",
            "碗",
            "盤",
            "杯",
            "刀",
            "叉",
            "湯匙",
            "保鮮",
            "廚房",
            "衛浴",
            "毛巾",
            "浴巾",
            "收納",
            "置物",
            "掛勾",
            "衣架",
            "寢具",
            "枕頭",
            "棉被",
        ],
        "日用清潔用品": [
            "衛生紙",
            "面紙",
            "濕紙巾",
            "洗衣",
            "清潔劑",
            "柔軟精",
            "漂白水",
            "消毒",
            "拖把",
            "掃把",
            "垃圾袋",
            "衛生棉",
            "護墊",
            "紙尿褲",
            "尿布",
        ],
        "家電/3C配件": [
            "風扇",
            "電扇",
            "冷氣",
            "暖氣",
            "除濕",
            "空氣清淨",
            "吸塵器",
            "電鍋",
            "電子鍋",
            "微波爐",
            "烤箱",
            "果汁機",
            "電腦",
            "滑鼠",
            "鍵盤",
            "耳機",
            "充電",
            "電池",
        ],
        "文具休閒服飾": [
            "筆",
            "紙",
            "文具",
            "筆記本",
            "便條紙",
            "膠帶",
            "剪刀",
            "釘書機",
            "服飾",
            "衣服",
            "褲子",
            "襪子",
            "內衣",
            "帽子",
            "包包",
            "背包",
            "運動",
            "球",
            "玩具",
        ],
    }

    # 顯示規則
    print("\n關鍵字規則:")
    for category, keywords in LOOKUP_RULES.items():
        print(f"\n{category}:")
        print(f"  關鍵字數量: {len(keywords)}")
        print(f"  範例: {', '.join(keywords[:10])}...")

    # ==================== 3. 定義分類函數 ====================
    def classify_by_lookup(brand, name, description):
        """用關鍵字查表分類"""
        # 合併所有文字資訊 (brand=品牌, name=商品名稱, description=詳細描述)
        text = str(brand) + " " + str(name) + " " + str(description)
        text = text.lower()

        # 統計每個類別匹配到幾個關鍵字
        scores = {}
        matched_keywords = {}

        for category, keywords in LOOKUP_RULES.items():
            score = 0
            matched = []
            for kw in keywords:
                if kw.lower() in text:
                    score += 1
                    matched.append(kw)
            scores[category] = score
            matched_keywords[category] = matched

        # 選分數最高的類別
        max_score = max(scores.values())
        if max_score > 0:
            predicted_category = max(scores, key=scores.get)
            return predicted_category, max_score, matched_keywords[predicted_category]
        else:
            return "未分類", 0, []

    # ==================== 4. 對所有商品做分類 ====================
    print("\n" + "=" * 70)
    print("步驟 3: 對所有商品進行分類")
    print("=" * 70)

    predictions = []
    scores = []
    matched_kws = []

    for idx, row in df.iterrows():
        pred, score, kws = classify_by_lookup(
            row["brand"], row["name"], row["description"]
        )
        predictions.append(pred)
        scores.append(score)
        matched_kws.append(kws)

    df["predicted_category"] = predictions
    df["match_score"] = scores
    df["matched_keywords"] = matched_kws

    # ==================== 5. 計算準確率 ====================
    print("\n" + "=" * 70)
    print("步驟 4: 評估結果")
    print("=" * 70)

    y_true = df["category"]
    y_pred = df["predicted_category"]

    # 統計未分類數量
    unclassified_count = (y_pred == "未分類").sum()
    classified_mask = y_pred != "未分類"

    print(f"\n分類統計:")
    print(
        f"  成功分類: {classified_mask.sum()} / {len(df)} ({classified_mask.sum()/len(df)*100:.1f}%)"
    )
    print(f"  未分類: {unclassified_count} ({unclassified_count/len(df)*100:.1f}%)")

    # 對成功分類的商品計算準確率
    if classified_mask.sum() > 0:
        accuracy_classified = accuracy_score(
            y_true[classified_mask], y_pred[classified_mask]
        )
        print(f"\n已分類商品的準確率: {accuracy_classified:.2%}")
    else:
        accuracy_classified = 0.0
        print("\n錯誤: 沒有商品被成功分類")

    # 整體準確率（未分類視為錯誤）
    accuracy_overall = accuracy_score(y_true, y_pred)
    print(f"整體準確率（含未分類）: {accuracy_overall:.2%}")

    # 分類報告（只看成功分類的）
    if classified_mask.sum() > 0:
        print("\n分類報告（已分類商品）:")
        print(
            classification_report(
                y_true[classified_mask], y_pred[classified_mask], zero_division=0
            )
        )

    # ==================== 6. 顯示錯誤案例 ====================
    print("\n" + "=" * 70)
    print("錯誤案例分析（顯示前 10 個）")
    print("=" * 70)

    errors = df[classified_mask & (y_true != y_pred)].head(10)
    if len(errors) > 0:
        for idx, row in errors.iterrows():
            print(f"\nSKU: {row['sku']}")
            print(f"  品牌: {row['brand']}")
            print(f"  商品名稱: {row['name']}")
            print(f"  實際類別: {row['category']}")
            print(f"  預測類別: {row['predicted_category']}")
            print(f"  匹配關鍵字: {', '.join(row['matched_keywords'][:5])}")

    # 顯示未分類案例
    print("\n" + "=" * 70)
    print("未分類案例（顯示前 10 個）")
    print("=" * 70)

    unclassified = df[~classified_mask].head(10)
    if len(unclassified) > 0:
        for idx, row in unclassified.iterrows():
            print(f"\nSKU: {row['sku']}")
            print(f"  品牌: {row['brand']}")
            print(f"  商品名稱: {row['name']}")
            print(f"  實際類別: {row['category']}")
            print(f"  商品描述: {str(row['description'])[:100]}...")

    # ==================== 7. 混淆矩陣 ====================
    print("\n生成混淆矩陣...")

    # 只對成功分類的商品畫混淆矩陣
    if classified_mask.sum() > 0:
        valid_categories = sorted(df[classified_mask]["category"].unique())
        cm = confusion_matrix(
            y_true[classified_mask], y_pred[classified_mask], labels=valid_categories
        )

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=valid_categories,
            yticklabels=valid_categories,
        )
        plt.title(
            f"Lookup Table 混淆矩陣\n整體準確率: {accuracy_overall:.2%} | 已分類準確率: {accuracy_classified:.2%}",
            fontsize=16,
            pad=20,
        )
        plt.xlabel("預測分類", fontsize=12)
        plt.ylabel("實際分類", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("output/lookup_confusion_matrix.png", dpi=300, bbox_inches="tight")
        print("混淆矩陣已儲存到 output/lookup_confusion_matrix.png")
        plt.close()

    # ==================== 8. 儲存結果 ====================
    print("\n儲存詳細結果...")
    output_path = "output/lookup_results.csv"
    df[
        [
            "sku",
            "brand",
            "name",
            "category",
            "predicted_category",
            "match_score",
            "matched_keywords",
        ]
    ].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"結果已儲存到 {output_path}")

    # 儲存統計摘要
    summary = {
        "方法": "Lookup Table",
        "整體準確率": accuracy_overall,
        "已分類準確率": accuracy_classified,
        "分類覆蓋率": classified_mask.sum() / len(df),
        "未分類數量": unclassified_count,
        "總商品數": len(df),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("output/lookup_summary.csv", index=False, encoding="utf-8-sig")
    print("統計摘要已儲存到 output/lookup_summary.csv")

    print("\n" + "=" * 70)
    print("Lookup Table 方法評估完成")
    print("=" * 70)
    print(f"\n主要指標:")
    print(f"  整體準確率: {accuracy_overall:.2%}")
    print(f"  已分類準確率: {accuracy_classified:.2%}")
    print(f"  分類覆蓋率: {classified_mask.sum()/len(df)*100:.1f}%")


if __name__ == "__main__":
    main()
