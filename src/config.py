"""
全局配置模块
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

DATA_CONFIG = {
    "stocks_tran_path": Path("./data/raw/train.csv"),
    "stocks_test_path": Path("./data/raw/test.csv"),
    "processed_data_path": Path("./data/processed"),
    "external_data_path": Path("./data/external"),
    "logs_path": Path("./logs"),
    "outputs_path": Path("./outputs"),
    "cache_path": Path("./cache"),
}


# 模型配置
MODEL_CONFIG = {
    "save_path": os.getenv("MODEL_SAVE_PATH", str(OUTPUTS_DIR / "models")),
    "prediction_output_path": os.getenv("PREDICTION_OUTPUT_PATH", str(OUTPUTS_DIR / "predictions")),
    "random_seed": int(os.getenv("RANDOM_SEED", 42)),
    "train_test_split_ratio": float(os.getenv("TRAIN_TEST_SPLIT_RATIO", 0.8)),
    "cross_validation_folds": int(os.getenv("CROSS_VALIDATION_FOLDS", 5))
}

# 特征工程配置
FEATURE_CONFIG = {
    "selection_method": os.getenv("FEATURE_SELECTION_METHOD", "mutual_info"),
    "max_features": int(os.getenv("MAX_FEATURES", 100)),
}

# 日志配置
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "file": os.getenv("LOG_FILE", str(LOGS_DIR / "app.log"))
}

# 沪深300成分股数量
HS300_STOCK_COUNT = 300

# 预测相关配置
PREDICTION_CONFIG = {
    "top_stocks_count": 10,  # 涨跌幅最大/最小的股票数量
    "prediction_days": 1,    # 预测天数
}

# 数据文件路径
DATA_PATHS = {
    "raw_data": DATA_DIR / "raw",
    "processed_data": DATA_DIR / "processed",
    "external_data": DATA_DIR / "external",
    "stocks": DATA_DIR / "external" / "stocks.csv",
    "prices": DATA_DIR / "processed" / "prices.csv",
    "features": DATA_DIR / "processed" / "features.csv"
} 