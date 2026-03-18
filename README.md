# NCHU 1141 — 資管系在職碩專班 機器學習專題

> **電商商品多模態分類系統**
> 整合文字（TF-IDF）、圖片（HOG + 色彩直方圖）與價格特徵，使用神經網路與傳統機器學習模型進行商品類別預測。

---

## 目錄

- [專案簡介](#專案簡介)
- [系統架構](#系統架構)
- [硬體與環境](#硬體與環境)
- [快速開始](#快速開始)
  - [方案 A：本機 Conda (PyTorch + Keras 3)（推薦）](#方案-a本機-conda-pytorch--keras-3推薦)
  - [方案 B：NVIDIA Docker 容器 (TensorFlow)（備用）](#方案-bnvidia-docker-容器-tensorflow備用)
- [執行流程](#執行流程)
- [目錄結構](#目錄結構)
- [模型成果](#模型成果)
- [開發歷程摘要](#開發歷程摘要)

---

## 專案簡介

本專案以「大買家」電商平台的商品資料為對象，建立一套多模態（Multimodal）商品自動分類系統：

| 特徵類型                | 方法             | 維度            |
| ----------------------- | ---------------- | --------------- |
| 文字（品牌 + 商品名稱） | TF-IDF           | 500 維          |
| 文字（商品描述）        | TF-IDF           | 500 維          |
| 圖片（色彩 + HOG）      | 色彩直方圖 + HOG | 576 維          |
| 價格                    | 數值特徵         | 1 維            |
| **總計**                |                  | **約 1,077 維** |

模型方面同時訓練並比較以下架構：

- **神經網路**：Keras 3（PyTorch 後端），架構為 `512 → 256 → 128 → 64`
- **傳統機器學習**：Random Forest、Logistic Regression
- **集成學習**：加權投票（RF × 0.76 + NN × 0.24）

---

## 系統架構

```
資料爬取 → 圖片下載 → 資料清理
    ↓
特徵工程（TF-IDF + HOG + 價格）
    ↓
模型訓練（神經網路 / 傳統模型）
    ↓
模型評估與比較
    ↓
集成預測 / 單一商品預測
```

---

## 硬體與環境

| 項目        | 規格                                                       |
| ----------- | ---------------------------------------------------------- |
| 作業系統    | Windows 11 + WSL2                                          |
| CPU         | AMD Ryzen 7 9800X3D                                        |
| GPU         | NVIDIA RTX 5070（Blackwell 架構，Compute Capability 12.0） |
| CUDA Driver | 591.44（支援至 CUDA 13.1）                                 |
| 主要框架    | Keras 3 + PyTorch 後端                                     |

> **說明**：RTX 50 系列（Blackwell 架構）對舊版 TensorFlow 官方 Wheel 的相容性不佳（`CUDA_ERROR_INVALID_PTX`），故本專案主要採用 **Keras 3 + PyTorch 後端** 作為解決方案。

---

## 快速開始

### 方案 A：本機 Conda (PyTorch + Keras 3)（推薦）

#### 1. 建立環境

```bash
# 建立或更新 Conda 環境（包含所有非 PyTorch 相依套件）
conda env update -f environment.pytorch.yml

# 啟動環境
conda activate py312_keras_torch
```

#### 2. 手動安裝 PyTorch（CUDA 13.0 版本）

由於 PyTorch Nightly Build 不在預設 PyPI 索引上，需手動安裝：

```bash
# 在 (py312_keras_torch) 環境中執行
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

#### 3. 驗證安裝

```bash
# 確認 PyTorch CUDA 可用
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
# 預期輸出：CUDA 可用: True

# 確認 Keras 後端
python -c "import keras; print(f'Keras Backend: {keras.backend.backend()}')"
# 預期輸出：Keras Backend: torch
```

---

### 方案 B：NVIDIA Docker 容器 (TensorFlow)（備用）

若需使用 TensorFlow，建議透過 NVIDIA NGC 官方容器以解決架構相容性問題。

#### 容器設定（`.devcontainer/devcontainer.json`）

```json
{
  "name": "RTX5070-TensorFlow",
  "image": "nvcr.io/nvidia/tensorflow:25.01-tf2-py3",
  "runArgs": ["--gpus=all"],
  "workspaceFolder": "/workspace",
  "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind",
  "forwardPorts": [8888],
  "postCreateCommand": "pip install jupyter matplotlib ipykernel",
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python", "ms-toolsai.jupyter"]
    }
  }
}
```

#### 容器內補裝套件與中文字型

```bash
pip install seaborn imbalanced-learn opencv-python
apt-get update && apt-get install -y fonts-noto-cjk
```

---

## 執行流程

依序執行以下腳本（位於 `src/` 目錄）：

| 步驟 | 腳本                                 | 說明                              |
| ---- | ------------------------------------ | --------------------------------- |
| 01   | `01_crawl_savesafe.py`               | 爬取大買家商品資料                |
| 02   | `02_download_images.py`              | 下載商品圖片                      |
| 03   | `03_clear_sold_out_product.py`       | 移除售完商品                      |
| 04   | `04_extract_image_features.py`       | 提取圖片特徵（HOG + 色彩）        |
| 05   | `05_tune_tfidf_params.py`            | TF-IDF 參數調優                   |
| 06   | `06_prepare_features.py`             | 整合文字、圖片與價格特徵          |
| 07   | `07_train_traditional_model.py`      | 訓練傳統機器學習模型（RF, LR）    |
| 08   | `08_train_neural_network.pytorch.py` | 訓練神經網路（Keras 3 + PyTorch） |
| 09   | `09_compare_models.pytorch.py`       | 比較各模型效能                    |
| 10   | `10_ensemble_prediction.pytorch.py`  | 集成學習預測                      |
| 11   | `11_predict_single_product.py`       | 單一商品預測                      |
| 12   | `12_baseline_lookup_table.py`        | 查表法 Baseline                   |

```bash
cd src
python 08_train_neural_network.pytorch.py
```

---

## 目錄結構

```
NCHU-114-ML/
├── environment.pytorch.yml          # Conda 環境定義（PyTorch 方案）
├── environment.tensorflow.yml       # Conda 環境定義（TensorFlow 方案，備用）
├── PROJECT.md                       # 專案說明與環境建置指南
├── DEV-LOG.md                       # 開發日誌
└── src/
    ├── 01_crawl_savesafe.py
    ├── 02_download_images.py
    ├── 03_clear_sold_out_product.py
    ├── 04_extract_image_features.py
    ├── 05_tune_tfidf_params.py
    ├── 06_prepare_features.py
    ├── 06_prepare_features.name_only.py
    ├── 07_train_traditional_model.py
    ├── 07_train_traditional_model.RandomizedSearchCV.py
    ├── 08_train_neural_network.py
    ├── 08_train_neural_network.pytorch.py
    ├── 08_train_neural_network.pytorch.grid_search.py
    ├── 09_compare_models.py
    ├── 09_compare_models.pytorch.py
    ├── 10_ensemble_prediction.pytorch.py
    ├── 11_predict_single_product.py
    ├── 12_baseline_lookup_table.py
    ├── input/                        # 預測輸入資料
    ├── output/
    │   ├── images/                   # 商品圖片
    │   ├── models/                   # 訓練完成的模型檔案
    │   └── result_images/            # 訓練曲線、混淆矩陣等圖表
    ├── test_images/
    └── utils/
        └── cross_platform_config.py  # 跨平台設定工具
```

---

## 模型成果

| 模型                     | 測試準確率 | 備註                                     |
| ------------------------ | ---------- | ---------------------------------------- |
| Random Forest            | ~83%       | 傳統模型最佳                             |
| Keras NN（PyTorch 後端） | **84.75%** | 架構：512→256→128→64，無 Label Smoothing |
| 集成（加權投票）         | ~83.72%    | RF×0.76 + NN×0.24                        |
| Baseline（查表法）       | -          | 對照組                                   |

**關鍵發現：**

- Label Smoothing（0.1）使準確率從 84.75% 下降至 77%，已放棄使用
- SMOTE 增強應只對訓練子集執行，驗證集與測試集必須保持純淨，避免資料洩漏
- 最佳 Random Seed `232268` 可達驗證集準確率 **82.69%**

---

## 開發歷程摘要

| 日期       | 里程碑                                                                      |
| ---------- | --------------------------------------------------------------------------- |
| 2025-12-07 | 專案初始化、特徵定義（文字 + 圖片 + 價格），初次訓練                        |
| 2025-12-08 | 修正 XLA 過慢與 OOM 問題（`jit_compile=False`）                             |
| 2025-12-09 | 完成資料清理，產出 cleaned dataset                                          |
| 2025-12-10 | TF-IDF 參數網格搜尋，優化文字特徵                                           |
| 2025-12-11 | WSL2 + RTX 5070 環境建置，腳本拆分重構                                      |
| 2025-12-12 | 集成學習實驗（加權投票 83.72% vs Stacking 82.63%）                          |
| 2025-12-13 | 修正 `workers` 參數錯誤，確認可重現性策略                                   |
| 2025-12-14 | 遷移至 Keras 3 + PyTorch 後端，Seed Mining 達 82.69%                        |
| 2025-12-16 | 規劃 Grid Search，定義滑動視窗時間序列輸入                                  |
| 2025-12-17 | Grid Search 重構、資料洩漏修正、放棄 Label Smoothing，最終準確率 **84.75%** |

詳細紀錄請參閱 [DEV-LOG.md](DEV-LOG.md)。
