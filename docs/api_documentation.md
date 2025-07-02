# API 文档

## 概述

本文档描述了沪深300股价预测项目中各个模块的API接口。

## 数据模块 (src.data)

### DataLoader

股票数据加载器，负责从各种数据源获取股票数据。

#### 类方法

##### `__init__(self)`
初始化数据加载器，设置Tushare API连接。

##### `load_hs300_stocks(self) -> pd.DataFrame`
加载沪深300成分股列表。

**返回值:**
- `pd.DataFrame`: 包含股票代码、权重等信息的DataFrame

**示例:**
```python
loader = DataLoader()
stocks = loader.load_hs300_stocks()
print(stocks.head())
```

##### `load_stock_data(self, stock_codes: List[str], start_date: str, end_date: str, source: str = "tushare") -> Dict[str, pd.DataFrame]`
批量加载股票历史数据。

**参数:**
- `stock_codes`: 股票代码列表
- `start_date`: 开始日期 (YYYY-MM-DD)
- `end_date`: 结束日期 (YYYY-MM-DD)  
- `source`: 数据源 ("tushare" 或 "yahoo")

**返回值:**
- `Dict[str, pd.DataFrame]`: 股票代码到数据的映射

**示例:**
```python
data = loader.load_stock_data(
    stock_codes=['000001.SZ', '000002.SZ'],
    start_date='2024-01-01',
    end_date='2024-01-31'
)
```

##### `load_market_data(self, start_date: str, end_date: str) -> pd.DataFrame`
加载市场整体数据（如沪深300指数）。

**参数:**
- `start_date`: 开始日期
- `end_date`: 结束日期

**返回值:**
- `pd.DataFrame`: 市场数据

##### `save_data(self, data: pd.DataFrame, filename: str) -> None`
保存数据到文件。

**参数:**
- `data`: 要保存的数据
- `filename`: 文件名

##### `load_saved_data(self, filename: str) -> Optional[pd.DataFrame]`
加载已保存的数据。

**参数:**
- `filename`: 文件名

**返回值:**
- `Optional[pd.DataFrame]`: 加载的数据，如果文件不存在则返回None

### DataPreprocessor

数据预处理器，负责数据清洗、缺失值处理、特征生成等。

#### 类方法

##### `__init__(self)`
初始化预处理器。

##### `clean_stock_data(self, data: pd.DataFrame) -> pd.DataFrame`
清洗股票数据，处理异常值和数据一致性问题。

**参数:**
- `data`: 原始股票数据

**返回值:**
- `pd.DataFrame`: 清洗后的数据

##### `handle_missing_values(self, data: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame`
处理缺失值。

**参数:**
- `data`: 输入数据
- `method`: 处理方法 ("forward_fill", "backward_fill", "interpolate", "drop")

**返回值:**
- `pd.DataFrame`: 处理后的数据

##### `detect_outliers(self, data: pd.DataFrame, columns: List[str], method: str = "iqr", threshold: float = 3.0) -> pd.DataFrame`
检测异常值。

**参数:**
- `data`: 输入数据
- `columns`: 要检测的列
- `method`: 检测方法 ("iqr", "zscore")
- `threshold`: 阈值

**返回值:**
- `pd.DataFrame`: 标记了异常值的数据

##### `normalize_data(self, data: pd.DataFrame, columns: List[str], method: str = "standard") -> pd.DataFrame`
数据标准化/归一化。

**参数:**
- `data`: 输入数据
- `columns`: 要标准化的列
- `method`: 标准化方法 ("standard", "minmax")

**返回值:**
- `pd.DataFrame`: 标准化后的数据

##### `calculate_returns(self, data: pd.DataFrame, price_column: str = "close", periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame`
计算收益率。

**参数:**
- `data`: 输入数据
- `price_column`: 价格列名
- `periods`: 计算周期列表

**返回值:**
- `pd.DataFrame`: 包含收益率的数据

##### `add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame`
添加技术指标。

**参数:**
- `data`: 输入数据

**返回值:**
- `pd.DataFrame`: 包含技术指标的数据

##### `create_time_features(self, data: pd.DataFrame) -> pd.DataFrame`
创建时间特征。

**参数:**
- `data`: 输入数据

**返回值:**
- `pd.DataFrame`: 包含时间特征的数据

## 特征模块 (src.features)

### FeatureEngineer

特征工程器，负责生成用于机器学习的特征。

#### 类方法

##### `create_features(self, data: pd.DataFrame) -> pd.DataFrame`
为给定的股票数据创建特征。

**参数:**
- `data`: 股票数据

**返回值:**
- `pd.DataFrame`: 包含所有特征的数据

### FeatureSelector

特征选择器，从大量特征中选择最有用的特征。

#### 类方法

##### `select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', k: int = 100) -> List[str]`
选择最佳特征。

**参数:**
- `X`: 特征矩阵
- `y`: 目标变量
- `method`: 选择方法
- `k`: 选择的特征数量

**返回值:**
- `List[str]`: 选中的特征名称列表

## 模型模块 (src.models)

### BaseModel

所有模型的基类。

#### 抽象方法

##### `fit(self, X: pd.DataFrame, y: pd.Series) -> None`
训练模型。

##### `predict(self, X: pd.DataFrame) -> np.ndarray`
生成预测。

### XGBoostModel, LightGBMModel, CatBoostModel

具体的机器学习模型实现。

#### 类方法

##### `__init__(self, **params)`
初始化模型参数。

##### `fit(self, X: pd.DataFrame, y: pd.Series) -> None`
训练模型。

##### `predict(self, X: pd.DataFrame) -> np.ndarray`
生成预测。

##### `get_feature_importance(self) -> pd.DataFrame`
获取特征重要性。

### EnsembleModel

集成模型，组合多个基础模型的预测结果。

#### 类方法

##### `__init__(self, models: List[BaseModel], weights: List[float] = None)`
初始化集成模型。

**参数:**
- `models`: 基础模型列表
- `weights`: 模型权重列表

##### `fit(self, X: pd.DataFrame, y: pd.Series) -> None`
训练所有基础模型。

##### `predict(self, X: pd.DataFrame) -> np.ndarray`
生成集成预测。

## 工具模块 (src.utils)

### setup_logger

设置日志记录器。

**参数:**
- `name`: 日志记录器名称
- `log_file`: 日志文件路径
- `log_level`: 日志级别
- `console_output`: 是否输出到控制台

**返回值:**
- `logging.Logger`: 配置好的日志记录器

**示例:**
```python
from src.utils import setup_logger

logger = setup_logger(
    name="my_module",
    log_file="logs/app.log",
    log_level="INFO"
)
logger.info("This is a log message")
```

## 配置模块 (src.config)

### 全局配置变量

- `PROJECT_ROOT`: 项目根目录
- `DATA_DIR`: 数据目录
- `OUTPUTS_DIR`: 输出目录
- `LOGS_DIR`: 日志目录
- `MODEL_CONFIG`: 模型配置
- `FEATURE_CONFIG`: 特征配置
- `PREDICTION_CONFIG`: 预测配置
- `DATA_PATHS`: 数据文件路径配置

## 错误处理

所有API方法都包含适当的错误处理，会记录错误日志并抛出相应的异常。常见异常类型：

- `FileNotFoundError`: 文件不存在
- `ValueError`: 参数值错误
- `ConnectionError`: 网络连接错误
- `APIError`: API调用错误

## 示例用法

### 完整的数据处理流程

```python
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor

# 1. 加载数据
loader = DataLoader()
stocks = loader.load_hs300_stocks()
stock_codes = stocks['con_code'].tolist()[:10]  # 前10只股票

# 2. 获取历史数据
data = loader.load_stock_data(
    stock_codes=stock_codes,
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# 3. 数据预处理
preprocessor = DataPreprocessor()
for code, df in data.items():
    cleaned = preprocessor.clean_stock_data(df)
    processed = preprocessor.handle_missing_values(cleaned)
    with_indicators = preprocessor.add_technical_indicators(processed)
    data[code] = with_indicators

print("数据处理完成")
```

### 模型训练流程

```python
from src.features.feature_engineer import FeatureEngineer
from src.models.tree_models import XGBoostModel
from src.training.trainer import ModelTrainer

# 1. 特征工程
feature_engineer = FeatureEngineer()
features = feature_engineer.create_features(processed_data)

# 2. 准备训练数据
X = features.drop(['target'], axis=1)
y = features['target']

# 3. 训练模型
trainer = ModelTrainer()
model = trainer.train_model(X, y, model_type='XGBoostModel')

print("模型训练完成")
``` 