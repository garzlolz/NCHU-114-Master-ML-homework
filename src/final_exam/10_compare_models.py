import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

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
    print("模型比較分析")
    print("=" * 70)

    # 建立輸出資料夾
    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/result_images", exist_ok=True) 
    # ==================== 1. 讀取所有模型結果 ====================
    print("\n步驟 1: 讀取所有模型結果")
    print("-" * 70)

    # 讀取傳統模型
    traditional_file = "output/models/traditional_models.pkl"
    if not os.path.exists(traditional_file):
        print(f"錯誤: 找不到 {traditional_file}")
        print("請先執行 07_train_traditional.py")
        return

    with open(traditional_file, "rb") as f:
        trad_data = pickle.load(f)

    # 讀取 Keras 模型
    keras_file = "output/models/keras_results.pkl"
    if not os.path.exists(keras_file):
        print(f"錯誤: 找不到 {keras_file}")
        print("請先執行 08_train_neural_network.py")
        return

    with open(keras_file, "rb") as f:
        keras_data = pickle.load(f)

    print("✓ 成功讀取所有模型結果")

    # ==================== 2. 整合結果 ====================
    print("\n步驟 2: 整合模型結果")
    print("-" * 70)

    # 整合準確率
    all_results = {}
    all_results.update(trad_data["results"])
    keras_name = f"Neural Network (Keras, lr={keras_data['best_lr']}, bs={keras_data['best_bs']})"
    all_results[keras_name] = keras_data["best_accuracy"]

    # 整合訓練時間
    all_times = {}
    all_times.update(trad_data["training_times"])
    all_times[keras_name] = keras_data["train_times"][(keras_data["best_lr"], keras_data["best_bs"])]

    # 找出最佳模型
    best_model_name = max(all_results, key=all_results.get)
    best_accuracy = all_results[best_model_name]

    print(f"最佳模型: {best_model_name}")
    print(f"準確率: {best_accuracy:.2%}")

    # ==================== 3. 生成比較圖表 ====================
    print("\n步驟 3: 生成比較圖表")
    print("-" * 70)

    # 3.1 準確率比較圖
    plt.figure(figsize=(14, 6))
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    bars = plt.bar(all_results.keys(), all_results.values(), color=colors, alpha=0.8)
    plt.ylabel("準確率", fontsize=12)
    plt.title("不同模型準確率比較", fontsize=16)
    plt.ylim(0, 1)
    
    for i, (name, acc) in enumerate(all_results.items()):
        plt.text(i, acc + 0.01, f"{acc:.2%}", ha="center", fontsize=11, fontweight="bold")
    
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("output/result_images/model_comparison.png", dpi=300, bbox_inches="tight")
    print("準確率比較圖已儲存到 output/result_images/model_comparison.png")
    plt.close()

    # 3.2 訓練時間比較圖
    fig, ax = plt.subplots(figsize=(14, 6))

    model_names = list(all_times.keys())
    times = list(all_times.values())

    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    bars = ax.bar(range(len(model_names)), times, color=colors, alpha=0.8)

    ax.set_ylabel("訓練時間 (秒)", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_title("模型訓練時間比較", fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, axis="y")

    for i, t in enumerate(times):
        if t < 60:
            time_text = f"{t:.1f}s"
        elif t < 3600:
            time_text = f"{t/60:.1f}m"
        else:
            time_text = f"{t/3600:.2f}h"

        ax.text(i, t + max(times) * 0.02, time_text, ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("output/result_images/training_time_comparison.png", dpi=300, bbox_inches="tight")
    print("訓練時間比較圖已儲存到 output/result_images/training_time_comparison.png")
    plt.close()

    # 3.3 效能/時間權衡圖
    plt.figure(figsize=(10, 6))
    
    accuracies = [all_results[name] for name in model_names]
    
    plt.scatter(times, accuracies, s=300, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
    
    for i, name in enumerate(model_names):
        short_name = name.split('(')[0].strip()  # 縮短名稱
        plt.annotate(short_name, (times[i], accuracies[i]), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.xlabel("訓練時間 (秒)", fontsize=12, fontweight='bold')
    plt.ylabel("測試準確率", fontsize=12, fontweight='bold')
    plt.title("模型效能 vs 訓練時間權衡", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/result_images/performance_time_tradeoff.png", dpi=300, bbox_inches="tight")
    print("效能/時間權衡圖已儲存到 output/result_images/performance_time_tradeoff.png")
    plt.close()

    # ==================== 4. 生成摘要報告 ====================
    print("\n步驟 4: 生成摘要報告")
    print("-" * 70)

    summary = []
    summary.append("=" * 70)
    summary.append("模型比較摘要報告")
    summary.append("=" * 70)
    summary.append("")
    
    summary.append("1. 準確率排名:")
    for i, (name, acc) in enumerate(sorted(all_results.items(), key=lambda x: x[1], reverse=True), 1):
        summary.append(f"   {i}. {name}: {acc:.2%}")
    
    summary.append("")
    summary.append("2. 訓練時間:")
    for name, t in all_times.items():
        if t < 60:
            time_str = f"{t:.2f} 秒"
        elif t < 3600:
            time_str = f"{t/60:.2f} 分鐘"
        else:
            time_str = f"{t/3600:.2f} 小時"
        summary.append(f"   {name}: {time_str}")
    
    summary.append("")
    summary.append("3. 最佳模型:")
    summary.append(f"   模型: {best_model_name}")
    summary.append(f"   準確率: {best_accuracy:.2%}")
    summary.append(f"   訓練時間: {all_times[best_model_name]:.2f} 秒")
    
    summary.append("")
    summary.append("=" * 70)

    # 輸出到終端
    for line in summary:
        print(line)

    # 儲存到檔案
    with open("output/model_comparison_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print("\n摘要報告已儲存到 output/model_comparison_summary.txt")

    # ==================== 5. 儲存最佳模型 ====================
    print("\n步驟 5: 儲存最佳模型")
    print("-" * 70)

    best_model_data = {
        "model_name": best_model_name,
        "accuracy": best_accuracy,
        "training_time": all_times[best_model_name],
    }

    # 根據最佳模型類型選擇儲存方式
    if "Keras" in best_model_name:
        best_model_data["model"] = keras_data["best_model"]
        best_model_data["y_pred"] = keras_data["best_y_pred"]
        best_model_data["label_encoder"] = keras_data["label_encoder"]
    else:
        model_type = best_model_name
        best_model_data["model"] = trad_data["models"][model_type]
        best_model_data["y_pred"] = trad_data["predictions"][model_type]
        best_model_data["label_encoder"] = trad_data["label_encoder"]

    best_model_file = "output/models/best_model.pkl"
    with open(best_model_file, "wb") as f:
        pickle.dump(best_model_data, f)
    print(f"最佳模型已儲存到 {best_model_file}")

    print("\n" + "=" * 70)
    print("模型比較分析完成")
    print("=" * 70)
    print("\n生成的檔案:")
    print("  - output/model_comparison.png (準確率比較)")
    print("  - output/training_time_comparison.png (訓練時間比較)")
    print("  - output/performance_time_tradeoff.png (效能/時間權衡)")
    print("  - output/model_comparison_summary.txt (摘要報告)")
    print(f"  - {best_model_file} (最佳模型)")


if __name__ == "__main__":
    main()
