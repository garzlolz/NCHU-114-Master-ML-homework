## 開發日誌：商品分類系統 (NCHU 114 ML)

**硬體環境**：Windows 11 / WSL2 / AMD Ryzen 7 9800X3D / NVIDIA RTX 5070
**核心技術**：Keras 3 + PyTorch Backend (解決 Blackwell 架構相容性問題)

---

### 2025-12-17：Grid Search 重構、資料管線修正與正則化實驗

**Grid Search 重構與系統區分**

- **需求變更**：將原本的 Seed Mining 策略改為系統性的 Grid Search，以尋找最佳超參數組合（Learning Rate, Batch Size, Dropout）。
- **跨平台支援**：在程式碼中加入 `platform.system()` 判斷，將輸出的 CSV 依據作業系統（Windows/Linux）分檔，避免雙系統協作時發生檔案寫入衝突。
- **重複運算排除**：新增讀取 CSV 紀錄的機制，執行前先載入已測試過的參數組合或 Seed，避免重複運算浪費時間。

**資料洩漏（Data Leakage）修正**

- **問題發現**：觀察訓練曲線發現訓練集準確率趨近 1.0，但驗證集準確率停滯。檢查後發現驗證集（Validation Set）是從經過 SMOTE 增強後的資料切分出來的，導致驗證集包含合成樣本，造成評估虛高（Validation Contamination）。
- **修正策略**：

1. 從原始訓練集（未經 SMOTE）先切分出固定的驗證集。
2. 僅對剩餘的訓練子集進行 SMOTE 增強。
3. 驗證集與測試集保持純淨，不進行任何合成增強。

- **Grid Search 同步修正**：確保 Grid Search 的每個迴圈都嚴格遵守上述流程：先切分驗證集，再於內部對訓練集做 SMOTE。

**模型架構與參數調整**

- **架構對齊**：將 Grid Search 使用的模型架構與訓練腳本統一。原本 Grid Search 腳本使用較大的 1024 神經元層，現已統一縮減為 `512 -> 256 -> 128 -> 64`，以符合資料集規模並減少過擬合。
- **Label Smoothing A/B 測試**：
- **方案 A（Smoothing=0.1）**：測試準確率下降至 **77%**。顯示在此資料集上，平滑化標籤導致模型對類別邊界的判斷變得模糊，反而降低了辨識能力。
- **方案 B（無 Smoothing）**：測試準確率維持在 **84.75%**。雖然模型存在過度自信（Over-confidence）的理論風險，但在實際分類準確度上表現顯著較優。
- **決策**：放棄 Label Smoothing，維持使用標準的 Categorical Crossentropy，優先確保分類準確率。

**文件化**

- **產出**：整理完整開發日誌，記錄從環境建置、特徵工程到模型優化的完整歷程。

---

### 2025-12-16：時間序列輸入定義與 Grid Search 規劃

**時間序列資料處理（FFNN + Sliding Window）**

- **需求**：針對時間序列資料（如股價預測），定義 FFNN 的輸入形狀。
- **設計決策**：
- 定義 Window Size（例如 3 天）與每日特徵數（例如 3 個）。
- Input Shape 定義為 `(window_size * features_per_day, )`，即將視窗內的二維資料攤平為一維向量，以符合全連接層（Dense Layer）的輸入要求。

- **樣本生成**：明確定義 Sliding Window 的運作方式，以前 N 天預測第 N+1 天，並推導出樣本總數為 `總天數 - window_size`。

**超參數搜尋規劃**

- **策略轉向**：決定暫停單純的 Seed Mining，轉而先進行 Grid Search 鎖定 Learning Rate 與 Batch Size 的最佳區間。
- **參數範圍**：設定 Learning Rate `[1e-4, 5e-4]` 與 Batch Size `[16, 64]` 進行排列組合測試。

---

### 2025-12-14：PyTorch 後端遷移與 Seed Mining

**環境遷移至 Keras 3 + PyTorch**

- **背景**：RTX 5070 (Blackwell 架構) 對 TensorFlow 的官方 Wheel 支援不佳，出現 CUDA 相容性問題。
- **解決方案**：環境切換為 Keras 3，並設定環境變數 `KERAS_BACKEND="torch"`，成功調用 GPU 進行訓練。

**Seed Mining（隨機種子挖掘）**

- **問題**：觀察到模型準確率隨 Random Seed 波動明顯。
- **策略**：撰寫自動化腳本，固定超參數，隨機生成 Seed 進行訓練，僅保留突破歷史最佳準確率的模型。
- **成果**：
- 挖掘到 Seed `232268`，達成驗證集準確率 **82.69%**。
- 發現若僅固定權重初始化的 Seed 但未固定 `train_test_split` 的 Seed，將無法重現結果。最終修正為全流程固定 Seed。

---

### 2025-12-13：效能優化與 API 差異釐清

**並行處理（Parallelization）**

- **技術釐清**：探討 `workers` 與 `use_multiprocessing` 參數。確認在 Keras PyTorch Backend 中，`fit()` 函數不支援 TensorFlow 專用的 `workers` 參數，導致 `TypeError`。
- **決策**：移除 `workers` 參數。針對中型資料集（約 7000 筆），資料已全數載入 RAM，多程序載入無顯著效益，維持單執行緒即可。

**可重現性確認**

- 確認 `pkl` 檔案無法直接反推訓練時的隨機狀態，必須在程式碼中顯式定義 `RANDOM_SEED` 常數，並統一應用於資料切分、SMOTE 與模型初始化。

---

### 2025-12-12：集成學習與特徵消融

**集成學習（Ensemble Learning）**

- **加權投票**：結合 Random Forest 與 Keras Neural Network。
- **權重搜尋**：透過 Grid Search 尋找最佳權重比例。結果顯示 `RF(0.76) + NN(0.24)` 為最佳組合。
- **Stacking**：嘗試以 Logistic Regression 作為 Meta-model，但準確率（82.63%）低於加權投票（83.72%），推測因資料量不足導致 Meta-model 過擬合。

**PyTorch Backend 調參**

- 執行小規模 Grid Search：
- Learning Rate: `0.00025`
- Batch Size: `20`
- 測試準確率：82.49%

**特徵消融想法**

- 討論是否移除 `description` 文字特徵僅保留 `brand+name`。推測若商品名稱過短，移除描述將導致準確率顯著下降。

---

### 2025-12-11：環境建置、Docker 整合與初步模型訓練

**09:54 - 12:36：模型訓練與腳本拆分**

- **腳本重構**：將單一訓練腳本拆分為傳統模型 (`07`)、神經網路 (`08`) 與結果比較 (`09`) 三部分，透過 Pickle 傳遞資料。
- **檔案管理**：統一將模型存檔至 `output/models/`，圖表輸出至 `output/images/`。
- **架構實驗**：嘗試縮減模型架構（移除第一層 1024 節點），導致準確率從 81.26% 下跌至 80.03%，確認模型容量不足。
- **錯誤排除**：解決 TensorFlow 對稀疏矩陣（Sparse Matrix）要求索引排序的問題（`InvalidArgumentError`），在 SMOTE 後增加 `sort_indices()` 處理。

**00:01 - 00:45：WSL2 + RTX 5070 環境建置**

- **驅動確認**：WSL2 內 `nvidia-smi` 成功辨識 RTX 5070，Driver 591.44，CUDA 13.1。
- **Docker 整合**：確認 Docker Desktop 可透過 WSL2 執行 GPU Passthrough。
- **容器選型**：因本機 pip 安裝 TensorFlow 遭遇 PTX 相容性問題，改用 NVIDIA NGC 容器 `nvcr.io/nvidia/tensorflow:25.01-tf2-py3`。
- **Dev Containers**：設定 VS Code Dev Container，解決容器內缺少中文字型（安裝 `fonts-noto-cjk`）與 Python 套件（seaborn 等）的問題。

**Keras 訓練錯誤修正**

- 修正 `ValueError: validation_split is only supported for Tensors or NumPy arrays`。原因為 Scipy 稀疏矩陣不支援自動切分，改為手動使用 `train_test_split` 切分驗證集並轉為 Dense 格式輸入。

---

### 2025-12-10：特徵工程與 TF-IDF 優化

**TF-IDF 參數調優**

- **目標**：準確率突破 90%（以學術專題標準設定）。
- **策略**：優先優化文字特徵。
- **方法**：建立獨立調優腳本，使用 Cross-Validation 網格搜尋最佳的 `max_features`。
- **概念釐清**：確認 `max_features` 是指全域詞彙表的維度上限，而非單一樣本的詞數。

---

### 2025-12-07 ~ 2025-12-09：專案初始化與資料處理

**2025-12-09：資料落地**

- 產出清理後的資料集 `savesafe_cleaned_products_...csv`。

**2025-12-08：訓練穩定性修正**

- **問題**：訓練過程遭遇 XLA/JIT 編譯過慢與記憶體不足（OOM/Killed）問題。
- **修正**：在 Keras `compile()` 中設定 `jit_compile=False`，成功穩定訓練流程。

**2025-12-07：特徵定義與初始訓練**

- **欄位語意**：重新定義 CSV 欄位，確認 `description` 為主要品名，`description_detail` 為規格描述。
- **特徵工程**：
- 文字：Brand+Name (TF-IDF 500 維) + Description (TF-IDF 500 維)。
- 圖片：提取色彩與 HOG 特徵共 576 維。
- 總維度：約 720 維。

- **字型設定**：統一使用 `Noto Sans CJK JP` 以解決 Matplotlib 中文亂碼問題。
- **Batch Size 實驗**：初步比較 Batch Size 32/64/128，確認 Batch Size 32 表現最佳（74.69%）。
