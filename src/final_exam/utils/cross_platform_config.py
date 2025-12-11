# utils/cross_platform_config.py 內容
import platform
import matplotlib.pyplot as plt

def set_matplotlib_font():
    system = platform.system()

    if system == "Windows":
        # Windows 環境：使用微軟正黑體
        font_list = ["Microsoft JhengHei", "Arial Unicode MS", "SimHei"]
        font_name = "Microsoft JhengHei"
    elif system == "Linux":
        # WSL/Docker/Linux 環境：使用 NotoSans 或思源
        # 您可以將 NotoSansCJK-Regular.ttc 複製到專案資料夾，並在此處指定相對路徑
        # 或者使用 Linux 系統已知的字體名稱
        font_list = ["Noto Sans CJK TC", "DejaVu Sans", "SimHei"] 
        font_name = "Noto Sans CJK TC"
    else:
        font_list = ["Arial"]
        font_name = "Arial"

    plt.rcParams["font.sans-serif"] = font_list
    plt.rcParams["axes.unicode_minus"] = False
    print(f"當前系統 ({system}) 使用字體：{font_name}")
    return font_name 

# 執行設定
# set_matplotlib_font()