# 项目设计文档：股票价格预测 (沪深300)

## 1. 概述

本文档概述了一个用于预测沪深300指数成分股价格变动的机器学习项目的设计。该项目专为竞赛场景设计，专注于预测单日内表现最好和最差的股票。

系统采用模块化架构设计，将数据处理、特征工程、建模、训练和评估等关注点分离到不同的组件中。这种设计提高了代码的可维护性、可扩展性和可重用性。

## 2. 系统架构

该项目遵循标准的机器学习流水线架构，由以下关键模块组成：

*   **数据模块 (`src/data`)**: 负责数据加载、预处理和验证。
*   **特征模块 (`src/features`)**: 处理从原始数据中创建丰富的特征集。
*   **模型模块 (`src/models`)**: 定义用于预测的机器学习模型。
*   **训练模块 (`src/training`)**: 管理模型训练过程，包括超参数优化。
*   **预测模块 (`src/prediction`)**: 按要求格式生成最终预测。
*   **评估模块 (`src/evaluation`)**: 评估已训练模型的性能。
*   **工具模块 (`src/utils`)**: 提供日志记录等通用工具。

## 3. 数据管理

### 3.1. 数据源

系统设计用于从多个来源获取数据：

*   **Tushare**: 历史股票数据的主要来源。
*   **Yahoo Finance**: 备用数据源。
*   **本地CSV文件**: 用于离线开发和测试。

### 3.2. 数据加载 (`src/data/data_loader.py`)

`DataLoader` 类负责：

*   从配置的数据源获取数据。
*   缓存数据以提高性能并减少API调用。
*   在不同数据源之间标准化列名。
*   进行基本的数据验证以确保数据质量。

### 3.3. 数据预处理 (`src/data/data_preprocessor.py`)

`DataPreprocessor` 类处理：

*   **数据清洗**: 删除重复记录并处理不一致性。
*   **缺失值插补**: 使用前向填充、后向填充或插值等方法。
*   **异常值检测**: 使用IQR或Z分数等方法识别和标记潜在的异常值。
*   **数据归一化/标准化**: 使用 `StandardScaler` 或 `MinMaxScaler` 缩放数值特征。

## 4. 特征工程

### 4.1. 特征创建 (`src/features/feature_engineer.py`)

`FeatureEngineer` 类是创建全面特征集的核心组件，包括：

*   **价格特征**: 收益率、对数收益率、价格比率等。
*   **成交量特征**: 成交量变化、成交量移动平均线等。
*   **技术指标**: 使用 `talib` 库生成广泛的指标。
*   **滞后特征**: 价格和其他特征的历史值。
*   **时间特征**: 星期几、月份、季度等。

### 4.2. 技术指标 (`src/features/technical_features.py`)

`TechnicalFeatures` 类是一个专用组件，用于计算各种技术指标，分为：

*   移动平均指标
*   动量指标
*   波动率指标
*   趋势指标
*   成交量指标

### 4.3. 特征选择 (`src/features/feature_selector.py`)

`FeatureSelector` 类提供了多种选择最相关特征的方法，包括：

*   **基于相关性的选择**
*   **互信息**
*   **递归特征消除 (RFE)**
*   **Lasso 正则化**
*   **基于模型的特征重要性**

## 5. 建模

### 5.1. 模型抽象 (`src/models/base_model.py`)

`BaseModel` 类是一个抽象基类，为项目中的所有模型定义了通用接口。这确保了一致性，并使新模型的集成变得容易。该接口包括以下方法：

*   `build_model()`: 创建模型实例。
*   `fit()`: 训练模型。
*   `predict()`: 生成预测。
*   `evaluate()`: 评估模型性能。
*   `save_model()` 和 `load_model()`: 用于模型持久化。

### 5.2. 模型实现

该项目包括多种模型的实现：

*   **树模型 (`src/models/tree_models.py`)**:
    *   XGBoost (`XGBoostModel`)
    *   LightGBM (`LightGBMModel`)
    *   CatBoost (`CatBoostModel`)
    *   ExtraTrees (`ExtraTreesModel`)
*   **线性模型 (`src/models/linear_models.py`)**:
    *   线性回归 (`LinearModel`)
    *   Ridge (`RidgeModel`)
    *   Lasso (`LassoModel`)
    *   ElasticNet (`ElasticNetModel`)
    *   贝叶斯岭回归 (`BayesianRidgeModel`)
*   **集成模型 (`src/models/ensemble_models.py`)**:
    *   投票回归器 (`EnsembleModel`)
    *   堆叠 (`StackingEnsemble`)
    *   混合 (`BlendingEnsemble`)
    *   Bagging (`BaggingEnsemble`)

## 6. 训练与评估

### 6.1. 模型训练 (`src/training/trainer.py`)

`Trainer` 类负责协调模型训练过程：

*   **数据分割**: 将数据分为训练集、验证集和测试集，支持随机和基于时间序列的分割。
*   **单模型和多模型训练**: 提供训练单个模型或一组模型的方法。
*   **交叉验证**: 支持k折和时间序列交叉验证。
*   **模型持久化**: 保存和加载训练好的模型。

### 6.2. 超参数优化 (`src/training/hyperopt.py`)

`HyperOptimizer` 类（如README中所述）将使用像Optuna这样的库来执行自动超参数调整，以找到给定模型的最佳参数集。

### 6.3. 模型评估 (`src/evaluation/evaluator.py`)

`Evaluator` 类负责对模型性能进行全面评估：

*   **回归指标**: MSE, RMSE, MAE, R-squared等。
*   **金融指标**: 方向准确率、信息系数 (IC)、夏普比率等。
*   **可视化**: 生成图表用于：
    *   预测值与实际值对比
    *   残差分析
    *   特征重要性
    *   模型比较

## 7. 预测

`Predictor` 类 (`src/prediction/predictor.py`) 将负责：

*   加载训练好的模型。
*   获取最新的市场数据。
*   为目标变量（例如，单日回报率）生成预测。
*   根据预测对股票进行排名，以识别表现最佳和最差的股票。

## 8. 配置

该项目被设计为高度可配置的：

*   **`configs/model_config.yaml`**: 定义不同机器学习模型的参数。
*   **`src/config.py`**: 包含数据路径、特征工程参数和其他全局设置的配置。

## 9. 快速入门和使用

该项目包含一个 `Makefile` 和多个脚本来简化工作流程：

*   `make data` 或 `python scripts/prepare_data.py`: 下载和预处理数据。
*   `make train` 或 `python scripts/train_model.py`: 训练模型。
*   `make predict` 或 `python scripts/predict.py`: 生成预测。

## 10. 未来增强

该项目被设计为可扩展的。潜在的未来增强功能包括：

*   **深度学习模型**: 集成LSTM或Transformers等模型。
*   **替代数据**: 整合来自新闻或社交媒体的情感分析，或宏观经济指标。
*   **高级风险管理**: 实施更复杂的风险管理和投资组合构建技术。
*   **实时预测**: 构建实时预测流水线。
*   **API和仪表板**: 通过REST API公开预测功能，并创建交互式仪表板进行可视化。

本设计文档全面概述了项目的架构和组件。系统的模块化和可配置性为未来的开发和实验提供了灵活性。
