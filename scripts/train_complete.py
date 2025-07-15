#!/usr/bin/env python3
"""
完整的股票预测模型训练流程

这个脚本演示了完整的训练流程，包括：
1. 数据加载和预处理
2. 特征工程
3. 模型训练
4. 模型评估
5. 模型保存
6. 预测结果可视化
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.features.feature_engineer import FeatureEngineer
from src.models.tree_models import XGBoostModel, LightGBMModel, CatBoostModel
from src.models.linear_models import RidgeModel, LassoModel
from src.models.ensemble_models import EnsembleModel, StackingEnsemble
from src.models.lstm_model import LSTMModel, GRUModel
from src.training.trainer import Trainer
from src.evaluation.evaluator import ModelEvaluator
from src.config import DATA_PATHS, MODEL_CONFIG


def load_and_preprocess_data():
    """加载和预处理数据"""
    print("1. 开始数据加载和预处理...")
    
    # 初始化数据加载器
    loader = DataLoader(enable_cache=True)
    
    # 加载训练数据
    train_data = loader.load_saved_data("train.csv", "raw")
    if train_data is None:
        print("错误：无法加载训练数据")
        return None
    
    print(f"   原始数据形状: {train_data.shape}")
    
    # 标准化列名
    train_data = loader.rename_columns(train_data, loader.column_mapping)
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    cleaned_data = preprocessor.clean_stock_data(train_data)
    processed_data = preprocessor.handle_missing_values(cleaned_data)
    
    print(f"   清洗后数据形状: {processed_data.shape}")
    
    return processed_data


def create_features(data):
    """创建特征"""
    print("2. 开始特征工程...")
    
    preprocessor = DataPreprocessor()
    
    # 添加技术指标
    technical_data = preprocessor.add_technical_indicators(data)
    
    # 添加时间特征
    feature_data = preprocessor.create_time_features(technical_data)
    
    # 计算收益率作为目标变量
    feature_data = preprocessor.calculate_returns(
        feature_data, 
        price_column="close",
        periods=[1, 3, 5]
    )
    
    # 删除包含NaN的行
    feature_data = feature_data.dropna()
    
    print(f"   特征工程后数据形状: {feature_data.shape}")
    print(f"   特征数量: {len(feature_data.columns)}")
    
    return feature_data


def prepare_training_data(data, target_col='return_1d', test_size=0.2):
    """准备训练数据"""
    print("3. 准备训练数据...")
    
    # 分离特征和目标
    feature_cols = [col for col in data.columns if not col.startswith('return_')]
    X = data[feature_cols]
    y = data[target_col]
    
    print(f"   特征维度: {X.shape}")
    print(f"   目标变量: {target_col}")
    
    return X, y


def train_traditional_models(X, y):
    """训练传统机器学习模型"""
    print("4. 开始训练传统模型...")
    
    # 初始化训练器
    trainer = Trainer()
    
    # 准备数据
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        X, y, test_size=0.2, validation_size=0.1, time_series_split=True
    )
    
    # 定义模型
    models = {
        'XGBoost': XGBoostModel(MODEL_CONFIG.get('xgboost', {})),
        'LightGBM': LightGBMModel(MODEL_CONFIG.get('lightgbm', {})),
        'CatBoost': CatBoostModel(MODEL_CONFIG.get('catboost', {})),
        'Ridge': RidgeModel(),
        'Lasso': LassoModel()
    }
    
    # 训练所有模型
    trained_models = trainer.train_multiple_models(
        models, X_train, y_train, X_val, y_val
    )
    
    # 评估模型
    results = {}
    for name, model in trained_models.items():
        test_metrics = model.evaluate(X_test, y_test)
        results[name] = {
            'test_metrics': test_metrics,
            'model': model
        }
        print(f"   {name} - RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")
    
    return trainer, results, (X_test, y_test)


def train_lstm_model(data, target_col='close'):
    """训练LSTM模型"""
    print("5. 开始训练LSTM模型...")
    
    # 为LSTM准备数据
    feature_cols = [col for col in data.columns if col != target_col]
    lstm_data = data[feature_cols + [target_col]].copy()
    
    # 初始化LSTM模型
    lstm_config = {
        'seq_length': 30,
        'lstm_units': [64, 32],
        'dropout_rate': 0.2,
        'epochs': 50,
        'batch_size': 32,
        'patience': 10
    }
    
    lstm_model = LSTMModel(lstm_config)
    
    # 训练模型
    lstm_model.fit(lstm_data, verbose=0)
    
    print("   LSTM模型训练完成")
    
    return lstm_model


def train_ensemble_model(X, y):
    """训练集成模型"""
    print("6. 开始训练集成模型...")
    
    trainer = Trainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        X, y, test_size=0.2, validation_size=0.1, time_series_split=True
    )
    
    # 创建基础模型
    base_models = {
        'XGBoost': XGBoostModel(),
        'LightGBM': LightGBMModel(),
        'CatBoost': CatBoostModel(),
        'Ridge': RidgeModel()
    }
    
    # 训练基础模型
    trained_base = trainer.train_multiple_models(
        base_models, X_train, y_train, X_val, y_val
    )
    
    # 训练堆叠集成模型
    stacking_model = StackingEnsemble()
    for name, model in trained_base.items():
        stacking_model.add_model(model)
    
    stacking_model.fit(X_train, y_train)
    
    # 评估集成模型
    stacking_metrics = stacking_model.evaluate(X_test, y_test)
    print(f"   堆叠集成模型 - RMSE: {stacking_metrics['rmse']:.4f}, R²: {stacking_metrics['r2']:.4f}")
    
    return stacking_model, stacking_metrics


def visualize_results(trainer, X_test, y_test):
    """可视化结果"""
    print("7. 开始结果可视化...")
    
    # 获取最佳模型
    try:
        best_model_name, best_model = trainer.get_best_model(metric='rmse')
        print(f"   最佳模型: {best_model_name}")
        
        # 预测
        y_pred = best_model.predict(X_test)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('模型评估结果', fontsize=16)
        
        # 1. 实际值 vs 预测值
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('实际值')
        axes[0, 0].set_ylabel('预测值')
        axes[0, 0].set_title('实际值 vs 预测值')
        
        # 2. 残差图
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('预测值')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title('残差图')
        
        # 3. 时间序列图
        test_index = range(len(y_test))
        axes[1, 0].plot(test_index, y_test, label='实际值', alpha=0.7)
        axes[1, 0].plot(test_index, y_pred, label='预测值', alpha=0.7)
        axes[1, 0].set_xlabel('时间')
        axes[1, 0].set_ylabel('收益率')
        axes[1, 0].set_title('时间序列对比')
        axes[1, 0].legend()
        
        # 4. 特征重要性（如果可用）
        importance = best_model.get_feature_importance()
        if importance:
            top_features = dict(list(importance.items())[:10])
            axes[1, 1].barh(list(top_features.keys()), list(top_features.values()))
            axes[1, 1].set_xlabel('重要性')
            axes[1, 1].set_title('Top 10 特征重要性')
        
        plt.tight_layout()
        
        # 保存图表
        output_dir = DATA_PATHS['outputs'] / 'plots'
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   图表已保存到: {output_dir / 'model_evaluation.png'}")
        
    except Exception as e:
        print(f"   可视化错误: {e}")


def save_models(trainer):
    """保存所有模型"""
    print("8. 开始保存模型...")
    
    # 创建模型保存目录
    model_dir = DATA_PATHS['models']
    model_dir.mkdir(exist_ok=True)
    
    # 保存所有模型
    trainer.save_models(str(model_dir))
    
    # 保存训练摘要
    summary_df = trainer.get_training_summary()
    if not summary_df.empty:
        summary_path = model_dir / 'training_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"   训练摘要已保存到: {summary_path}")
    
    print(f"   所有模型已保存到: {model_dir}")


def main():
    """主函数"""
    print("=" * 60)
    print("股票预测模型完整训练流程")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 加载和预处理数据
        data = load_and_preprocess_data()
        if data is None:
            return
        
        # 2. 创建特征
        feature_data = create_features(data)
        
        # 3. 准备训练数据
        X, y = prepare_training_data(feature_data)
        
        # 4. 训练传统模型
        trainer, results, (X_test, y_test) = train_traditional_models(X, y)
        
        # 5. 训练LSTM模型（可选）
        # lstm_model = train_lstm_model(feature_data)
        
        # 6. 训练集成模型
        stacking_model, stacking_metrics = train_ensemble_model(X, y)
        
        # 7. 可视化结果
        visualize_results(trainer, X_test, y_test)
        
        # 8. 保存模型
        save_models(trainer)
        
        print("\n" + "=" * 60)
        print("训练流程完成！")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 打印最终结果
        print("\n最终模型性能:")
        for name, result in results.items():
            print(f"  {name}: RMSE={result['test_metrics']['rmse']:.4f}, "
                  f"R²={result['test_metrics']['r2']:.4f}")
        print(f"  堆叠集成: RMSE={stacking_metrics['rmse']:.4f}, "
              f"R²={stacking_metrics['r2']:.4f}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()