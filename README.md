# NCHU, 資管系碩專班 114年ML功課
## 專題目錄
```text
. (專案根目錄)
├── README.md            # 專案說明文件
└── src/                 # 來源碼 (Source) 資料夾
    ├── final_exam/      # 專題程式存放位置
    │   │── output/      # 輸出
    │   │   ├── images   # 圖檔 
    │   │   ├── bank-full.csv 
    │   └── 01...-.txt 
    └──   └── 10...-.py
```


## 一、環境目標

在 Windows 11 + WSL2 + RTX 5070 上，用 **NVIDIA TensorFlow 容器**（`nvcr.io/nvidia/tensorflow:25.01-tf2-py3`）跑作業專案 `NCHU-114-Master-ML-homework`，讓 Keras 模型用 GPU 訓練，避免本機 pip/conda 的 CUDA 相容問題。

***

## 二、安裝與測試步驟

1. **確認 WSL 看得到 GPU**
   ```bash
   nvidia-smi
   ```
   看到 RTX 5070、Driver 591.44、CUDA Version 13.1。

2. **測試 Docker + GPU**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.0-devel-ubuntu22.04 nvidia-smi
   ```
   容器內同樣顯示 RTX 5070，代表 Docker GPU passthrough 正常。

3. **測試 NVIDIA TensorFlow 容器**
   ```bash
   docker run -it --rm --gpus all nvcr.io/nvidia/tensorflow:25.01-tf2-py3 \
     python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
   ```
   顯示 TensorFlow 2.17.0，且偵測到 `/physical_device:GPU:0`，證明容器內 TF 可以用 RTX 5070。

***

## 三、本機 pip 版 TensorFlow 失敗原因

在 WSL conda 環境 `wsl_ml_hw` 中用 pip 安裝了 `tensorflow==2.20.0` 和一堆 `nvidia-*-cu12` 套件，訓練時出現：

- `TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0`
- `CUDA_ERROR_INVALID_PTX`、`CUDA_ERROR_INVALID_HANDLE`

原因是 pip 版 TF 2.20 沒有為 Blackwell 的 compute capability 12.0 編好對應的 CUDA binary，對 RTX 50 系列在新 driver 上相容性不好。
解法是改用 **NVIDIA 官方 TensorFlow 容器**，裡面已針對新架構編譯好。

***

## 四、改用 VS Code Dev Container 的設定

1. 在專案根目錄建立 `.devcontainer/devcontainer.json`：
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
         "extensions": [
           "ms-python.python",
           "ms-toolsai.jupyter"
         ]
       }
     }
   }
   ```
   這會讓 VS Code 把目前專案資料夾掛載到容器的 `/workspace`，並自動開啟有 GPU 的 TensorFlow 開發環境。

2. 在 VS Code 中：
   - `Ctrl + Shift + P` → `Dev Containers: Reopen in Container`  
   - 進入容器後終端提示變成 `root@...:/workspace#`，`python -c "import tensorflow as tf; ..."` 可看到 GPU。

***

## 五、在容器裡補裝需要的套件與字型

1. **補裝 Python 套件**（因為容器和本機環境獨立）：
   ```bash
   cd /workspace
   pip install seaborn imbalanced-learn opencv-python
   ```
   其他如 pandas、matplotlib、scikit-learn 在 image 裡已預裝。

2. **安裝中文字型 Noto Sans CJK**（給 matplotlib 用）：
   ```bash
   apt-get update
   apt-get install -y fonts-noto-cjk
   ls /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
   ```
   程式裡 `FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"` 就能正常載入。

***

## 六、最後的訓練方式

在已進入 Dev Container 的 VS Code 終端：

```bash
cd /workspace/src/final_exam
python 07_train_model.py
```

這樣：
- 程式碼仍在主機 `~/repos/NCHU-114-Master-ML-homework` 裡（透過掛載）。  
- 訓練在容器環境中進行，使用 NVIDIA 官方編譯好的 TensorFlow + CUDA 12.8，穩定支援 RTX 5070（Blackwell）。
- 不再依賴本機 conda 的 TensorFlow，避免 `CUDA_ERROR_INVALID_PTX` 類問題。
