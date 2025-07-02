# 股票价格预测项目 (CSI 300)

基于沪深300指数成分股的股票价格预测项目，使用机器学习方法预测股票价格涨跌，专门针对竞赛场景设计。

## 项目简介

这是一个完整的股票价格预测机器学习项目，专门用于预测沪深300指数成分股的价格变动。项目目标是预测未来一天涨跌幅最大的前10只和后10只股票。

## 核心功能

- **数据获取**: 支持从多个数据源获取股票数据（Tushare、Yahoo Finance等）
- **特征工程**: 丰富的技术指标和特征生成
- **模型训练**: 支持多种机器学习算法（XGBoost、LightGBM、CatBoost等）
- **集成学习**: 多种集成方法（投票、堆叠、混合等）
- **超参数优化**: 基于Optuna的自动超参数调优
- **模型评估**: 详细的模型性能评估和可视化
- **预测输出**: 生成符合竞赛要求的预测结果

## 项目结构

```
my-python-project/
├── src/                    # 源代码
│   ├── data/              # 数据处理模块
│   │   ├── data_loader.py
│   │   └── data_preprocessor.py
│   ├── features/          # 特征工程模块
│   │   ├── feature_engineer.py    # 特征生成
│   │   ├── feature_selector.py    # 特征选择
│   │   └── technical_features.py  # 技术指标
│   ├── models/            # 模型定义
│   │   ├── base_model.py          # 基础模型类
│   │   ├── tree_models.py         # 树模型
│   │   ├── linear_models.py       # 线性模型
│   │   └── ensemble_models.py     # 集成模型
│   ├── training/          # 训练模块
│   │   ├── trainer.py             # 训练器
│   │   └── hyperopt.py            # 超参数优化
│   ├── prediction/        # 预测模块
│   │   └── predictor.py           # 预测器
│   ├── evaluation/        # 评估模块
│   │   └── evaluator.py           # 评估器
│   └── utils/             # 工具模块
│       └── logger.py
├── configs/               # 配置文件
│   └── model_config.yaml
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后数据
│   └── external/         # 外部数据
├── outputs/              # 输出目录
│   ├── models/           # 训练好的模型
│   └── predictions/      # 预测结果
├── tests/                # 测试代码
├── notebooks/            # Jupyter notebooks
├── logs/                 # 日志文件
├── requirements.txt      # Python依赖
├── environment.yml       # Conda环境配置
├── Makefile             # 便捷命令
└── README.md
```

## 快速开始

### 1. 环境设置

#### 使用 Conda (推荐)
```bash
# 创建环境
conda env create -f environment.yml
conda activate stock-prediction
```

#### 使用 pip
```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 配置数据源

创建环境变量文件（可参考项目中的配置示例）：
```bash
# 获取Tushare Token (https://tushare.pro/)
export TUSHARE_TOKEN="your_token_here"

# 其他配置
export MODEL_SAVE_PATH="./outputs/models"
export PREDICTION_OUTPUT_PATH="./outputs/predictions"
```

### 3. 数据准备

```bash
# 使用Makefile（推荐）
make data

# 或直接运行脚本
python prepare_data.py
```

### 4. 模型训练

```bash
# 训练模型
make train

# 或直接运行
python train_model.py
```

### 5. 生成预测

```bash
# 生成预测结果
make predict

# 或直接运行
python predict.py
```

## 详细使用说明

### 数据处理

项目支持多种数据源：
- **Tushare**: 主要数据源，需要注册获取Token
- **Yahoo Finance**: 备用数据源
- **本地CSV文件**: 支持离线使用

```python
from src.data.data_loader import DataLoader

loader = DataLoader()
data = loader.load_stock_data(
    symbols=['000001.SZ', '000002.SZ'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)
```

### 特征工程

支持多种特征类型：
- **价格特征**: 收益率、对数收益率、价格比率等
- **技术指标**: MA、EMA、MACD、RSI、布林带等
- **成交量特征**: 成交量移动平均、成交量比率等
- **滞后特征**: 历史价格和指标的延迟特征

```python
from src.features.feature_engineer import FeatureEngineer

fe = FeatureEngineer()
features = fe.create_all_features(data)
```

### 模型训练

支持多种机器学习算法：

#### 树模型
- XGBoost
- LightGBM  
- CatBoost
- ExtraTrees

#### 线性模型
- 线性回归
- Ridge回归
- Lasso回归
- ElasticNet
- 贝叶斯Ridge

#### 集成方法
- 投票集成
- 堆叠集成
- 混合集成
- Bagging集成

```python
from src.training.trainer import Trainer
from src.models.tree_models import XGBoostModel

trainer = Trainer()
model = XGBoostModel()

# 训练单个模型
trainer.train_single_model(model, X_train, y_train)

# 训练多个模型并比较
results = trainer.train_multiple_models(models_config, X_train, y_train)
```

### 超参数优化

基于Optuna的自动超参数调优：

```python
from src.training.hyperopt import HyperOptimizer

optimizer = HyperOptimizer()
result = optimizer.optimize(
    model_class=XGBoostModel,
    X=X_train,
    y=y_train,
    n_trials=100
)
```

### 模型评估

支持多种评估指标：
- **回归指标**: RMSE、MAE、R²、MAPE等
- **金融指标**: 方向准确率、信息系数、夏普比率等
- **可视化**: 预测图表、残差分析、模型对比等

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator()
metrics = evaluator.evaluate_model(model, X_test, y_test)
evaluator.plot_predictions(y_test, predictions)
```

### 预测生成

生成符合竞赛要求的预测结果：

```python
from src.prediction.predictor import Predictor

predictor = Predictor()
predictions = predictor.predict_top_stocks(
    model=model,
    data=latest_data,
    top_n=10
)
```

## 配置说明

### 模型配置

在 `configs/model_config.yaml` 中配置模型参数：

```yaml
xgboost:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.1
  
lightgbm:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.1
```

### 特征配置

在 `src/config.py` 中配置特征工程参数：

```python
FEATURE_CONFIG = {
    'price_windows': [1, 5, 10, 20],
    'technical_indicators': ['sma', 'ema', 'macd', 'rsi'],
    'lag_periods': [1, 3, 5]
}
```

## 性能指标

项目包含详细的性能评估：

### 回归指标
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差  
- **R²**: 决定系数
- **MAPE**: 平均绝对百分比误差

### 金融指标
- **方向准确率**: 预测涨跌方向的准确性
- **信息系数(IC)**: 预测值与真实值的相关性
- **夏普比率**: 风险调整后的收益率
- **最大回撤**: 投资组合的最大损失

## 开发指南

### 代码规范

使用以下工具确保代码质量：

```bash
# 代码格式化
make format

# 代码检查
make lint

# 运行测试
make test
```

### 添加新模型

1. 继承 `BaseModel` 类
2. 实现必要的方法
3. 添加配置文件
4. 编写单元测试

```python
from src.models.base_model import BaseModel

class MyModel(BaseModel):
    def build_model(self):
        # 实现模型构建
        pass
    
    def train(self, X, y):
        # 实现模型训练
        pass
```

### 添加新特征

在 `FeatureEngineer` 类中添加新的特征生成方法：

```python
def create_my_features(self, data):
    # 实现新特征逻辑
    return features_df
```

## 常见问题

### Q: 如何获取Tushare Token？
A: 访问 https://tushare.pro/ 注册账号并获取Token。

### Q: 模型训练速度慢怎么办？
A: 可以调整以下参数：
- 减少特征数量
- 使用特征选择
- 调整模型参数
- 使用更少的交叉验证折数

### Q: 如何处理缺失数据？
A: 项目自动处理缺失数据：
- 前向填充
- 线性插值
- 删除缺失率过高的特征

### Q: 预测结果不准确怎么办？
A: 尝试以下方法：
- 增加更多特征
- 使用集成方法
- 调整超参数
- 使用更长的历史数据

## 扩展功能

### 实时预测

项目支持实时数据获取和预测：

```python
# 实时预测示例
predictor = Predictor()
real_time_data = loader.get_real_time_data()
predictions = predictor.predict_real_time(real_time_data)
```

### API接口

使用FastAPI创建预测API：

```bash
# 启动API服务
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 可视化界面

使用Streamlit创建交互式界面：

```bash
# 启动可视化界面
streamlit run dashboard/app.py
```

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交变更
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License

## 联系方式

如有问题，请提交 Issue 或联系项目维护者。

## 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 基础功能实现
- 多种模型支持
- 完整的评估体系

### 计划功能
- [ ] 深度学习模型支持
- [ ] 情感分析特征
- [ ] 宏观经济指标
- [ ] 风险管理模块
- [ ] 回测系统 