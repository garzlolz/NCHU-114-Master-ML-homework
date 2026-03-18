# ==========================================
# 模型比較腳本：整合傳統機器學習與神經網路的結果
# 讀取 07 和 08 的訓練結果，生成比較圖表和摘要報告
# ==========================================
import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_IMAGE_DATA_FORMAT"] = "channels_last"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import keras

# 設定中文字體
from utils.cross_platform_config import set_matplotlib_font

font_name = set_matplotlib_font()
print("使用字型：", font_name)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [font_name]
plt.rcParams["axes.unicode_minus"] = False


def main():
    print("=" * 70)
    print("模型比較分析")
    print("=" * 70)

    # 建立輸出資料夾
    os.makedirs("output/result_images", exist_ok=True)

    # ==================== 1. 讀取傳統模型結果 ====================
    print("\n步驟 1: 讀取傳統模型結果")
    print("-" * 70)

    traditional_file = "output/models/traditional_models.pkl"
    if not os.path.exists(traditional_file):
        print(f"錯誤: 找不到 {traditional_file}")
        print("請先執行 07_train_traditional_model.py")
        return

    with open(traditional_file, "rb") as f:
        trad_data = pickle.load(f)

    print("傳統模型讀取成功")

    # ==================== 2. 讀取神經網路結果 ====================
    print("\n步驟 2: 讀取神經網路結果")
    print("-" * 70)

    keras_file = "output/models/keras_results.pkl"
    if not os.path.exists(keras_file):
        print(f"錯誤: 找不到 {keras_file}")
        print("請先執行 08_train_neural_network.pytorch.py")
        return

    with open(keras_file, "rb") as f:
        keras_data = pickle.load(f)

    print("神經網路結果讀取成功")

    # ==================== 3. 整合所有模型的結果 ====================
    print("\n步驟 3: 整合模型結果")
    print("-" * 70)

    # 整合準確率
    all_results = {}
    all_results.update(trad_data["results"])  # Random Forest, Logistic Regression

    # 加入 Keras 結果
    keras_name = f"Neural Net (Seed={keras_data['best_seed']})"
    all_results[keras_name] = keras_data["best_accuracy"]

    # 整合訓練時間
    all_times = {}
    all_times.update(trad_data["training_times"])
    all_times[keras_name] = keras_data["train_time"]

    # 整合預測結果（用於混淆矩陣）
    all_predictions = {}
    all_predictions.update(trad_data["predictions"])
    all_predictions[keras_name] = keras_data["best_y_pred"]

    # 取得測試集標籤
    y_test = trad_data["y_test"]
    le = trad_data["label_encoder"]

    # 顯示整合結果
    print("\n整合完成:")
    for name in all_results.keys():
        print(
            f"  - {name}: 準確率 {all_results[name]:.2%}, 耗時 {all_times[name]:.2f}秒"
        )

    # ==================== 4. 找出最佳模型 ====================
    best_model_name = max(all_results, key=all_results.get)
    best_accuracy = all_results[best_model_name]

    print(f"\n最佳模型: {best_model_name}")
    print(f"最佳準確率: {best_accuracy:.2%}")

    # ==================== 5. 生成比較圖表 ====================
    print("\n步驟 4: 生成比較圖表")
    print("-" * 70)

    # 5.1 準確率比較圖
    plt.figure(figsize=(14, 6))

    # 設定顏色 (神經網路用紅色，其他用藍色)
    colors = [
        "#e74c3c" if "Neural" in name else "#3498db" for name in all_results.keys()
    ]

    bars = plt.bar(all_results.keys(), all_results.values(), color=colors, alpha=0.8)
    plt.ylabel("準確率", fontsize=12)
    plt.title("不同模型準確率比較", fontsize=16, pad=20)
    plt.ylim(0.70, 0.90)

    # 在柱狀圖上顯示數值
    for i, (name, acc) in enumerate(all_results.items()):
        plt.text(
            i, acc + 0.005, f"{acc:.2%}", ha="center", fontsize=11, fontweight="bold"
        )

    plt.xticks(rotation=20, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        "output/result_images/model_comparison.png", dpi=300, bbox_inches="tight"
    )
    print("準確率比較圖已儲存: output/result_images/model_comparison.png")
    plt.close()

    # 5.2 訓練時間比較圖
    plt.figure(figsize=(14, 6))

    model_names = list(all_times.keys())
    times = list(all_times.values())

    bars = plt.bar(range(len(model_names)), times, color=colors, alpha=0.8)
    plt.ylabel("訓練時間 (秒)", fontsize=13, fontweight="bold")
    plt.xticks(range(len(model_names)), model_names, rotation=20, ha="right")
    plt.title("模型訓練時間比較", fontsize=16, fontweight="bold", pad=20)
    plt.grid(True, alpha=0.3, axis="y")

    # 在柱狀圖上顯示時間
    for i, t in enumerate(times):
        time_text = f"{t:.1f}s" if t < 60 else f"{t/60:.1f}m"
        plt.text(
            i,
            t + max(times) * 0.02,
            time_text,
            ha="center",
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(
        "output/result_images/training_time_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("訓練時間比較圖已儲存: output/result_images/training_time_comparison.png")
    plt.close()

    # 5.3 效能/時間權衡散佈圖
    plt.figure(figsize=(10, 6))
    accuracies = [all_results[name] for name in model_names]

    plt.scatter(
        times, accuracies, s=300, c=colors, alpha=0.6, edgecolors="black", linewidth=2
    )

    # 為每個點加上標籤
    for i, name in enumerate(model_names):
        short_name = name.split("(")[0].strip()  # 移除種子號碼
        plt.annotate(
            short_name,
            (times[i], accuracies[i]),
            xytext=(10, 10),
            textcoords="offset points",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.5),
            fontsize=10,
        )

    plt.xlabel("訓練時間 (秒)", fontsize=12)
    plt.ylabel("測試準確率", fontsize=12)
    plt.title("模型效能 vs 訓練時間權衡", fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "output/result_images/performance_time_tradeoff.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("效能/時間權衡圖已儲存: output/result_images/performance_time_tradeoff.png")
    plt.close()

    # ==================== 6. 生成摘要報告 ====================
    print("\n步驟 5: 生成摘要報告")
    print("-" * 70)

    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("模型比較摘要報告")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    # 按準確率排序
    summary_lines.append("1. 準確率排名:")
    for i, (name, acc) in enumerate(
        sorted(all_results.items(), key=lambda x: x[1], reverse=True), 1
    ):
        summary_lines.append(f"   {i}. {name:<35} {acc:.4f} ({acc:.2%})")

    summary_lines.append("")
    summary_lines.append("2. 訓練時間排名:")
    for i, (name, t) in enumerate(sorted(all_times.items(), key=lambda x: x[1]), 1):
        summary_lines.append(f"   {i}. {name:<35} {t:.2f} 秒")

    summary_lines.append("")
    summary_lines.append("3. 最佳模型詳情:")
    summary_lines.append(f"   模型名稱: {best_model_name}")
    summary_lines.append(f"   準確率: {best_accuracy:.4f} ({best_accuracy:.2%})")
    summary_lines.append(f"   訓練時間: {all_times[best_model_name]:.2f} 秒")

    # 顯示到螢幕
    for line in summary_lines:
        print(line)

    # 儲存到檔案
    summary_file = "output/model_comparison_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\n摘要報告已儲存: {summary_file}")

    # ==================== 7. 儲存最佳模型資訊 ====================
    print("\n步驟 6: 儲存最佳模型資訊")
    print("-" * 70)

    best_model_info = {
        "model_name": best_model_name,
        "accuracy": best_accuracy,
        "training_time": all_times[best_model_name],
        "y_pred": all_predictions[best_model_name],
        "y_test": y_test,
        "label_encoder": le,
    }

    # 如果最佳模型是神經網路，加入額外資訊
    if "Neural" in best_model_name:
        best_model_info["model_path"] = "output/models/best_keras_model.keras"
        best_model_info["seed"] = keras_data["best_seed"]
        best_model_info["lr"] = keras_data["best_lr"]
        best_model_info["bs"] = keras_data["best_bs"]
        print(f"最佳模型為神經網路 (Seed={keras_data['best_seed']})")
    else:
        # 傳統模型
        best_model_info["model"] = trad_data["models"][best_model_name]
        print(f"最佳模型為傳統機器學習: {best_model_name}")

    best_model_file = "output/models/final_best_model_info.pkl"
    with open(best_model_file, "wb") as f:
        pickle.dump(best_model_info, f)

    print(f"最佳模型資訊已儲存: {best_model_file}")

    # ==================== 8. 完成 ====================
    print("\n" + "=" * 70)
    print("模型比較完成！")
    print("=" * 70)
    print("\n生成的檔案:")
    print("  - output/result_images/model_comparison.png")
    print("  - output/result_images/training_time_comparison.png")
    print("  - output/result_images/performance_time_tradeoff.png")
    print("  - output/model_comparison_summary.txt")
    print("  - output/models/final_best_model_info.pkl")


if __name__ == "__main__":
    main()
