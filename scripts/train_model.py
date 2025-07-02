#!/usr/bin/env python3
"""
模型训练脚本

用于训练股价预测模型
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import joblib

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATHS, MODEL_CONFIG, LOG_CONFIG
from src.features.feature_engineer import FeatureEngineer
from src.features.feature_selector import FeatureSelector
from src.models.ensemble_models import EnsembleModel
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_CONFIG["file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """加载预处理后的数据"""
    try:
        data_path = DATA_PATHS["processed_data"] / "combined_stock_data.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {data_path}")
        
        data = pd.read_csv(data_path)
        data['date'] = pd.to_datetime(data['date'])
        
        logger.info(f"Loaded data with shape: {data.shape}")
        logger.info(f"Date range: {data['date'].min()} to {data['date'].max()}")
        logger.info(f"Number of unique stocks: {data['stock_code'].nunique()}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def prepare_features_and_targets(data):
    """准备特征和目标变量"""
    try:
        # 初始化特征工程器
        feature_engineer = FeatureEngineer()
        
        # 为每只股票计算特征
        stock_features = []
        
        for stock_code in data['stock_code'].unique():
            stock_data = data[data['stock_code'] == stock_code].copy()
            stock_data = stock_data.sort_values('date')
            
            # 生成特征
            features = feature_engineer.create_features(stock_data)
            stock_features.append(features)
        
        # 合并所有特征
        all_features = pd.concat(stock_features, ignore_index=True)
        
        # 移除包含NaN的行
        all_features = all_features.dropna()
        
        logger.info(f"Generated features with shape: {all_features.shape}")
        
        return all_features
        
    except Exception as e:
        logger.error(f"Error preparing features and targets: {e}")
        raise


def select_features(features_data):
    """特征选择"""
    try:
        feature_selector = FeatureSelector()
        
        # 分离特征和目标变量
        target_col = 'target_return_1d'  # 假设这是目标变量
        if target_col not in features_data.columns:
            logger.error(f"Target column {target_col} not found")
            return None, None, None
        
        feature_columns = [col for col in features_data.columns 
                          if col not in ['date', 'stock_code', target_col]]
        
        X = features_data[feature_columns]
        y = features_data[target_col]
        
        # 进行特征选择
        selected_features = feature_selector.select_features(X, y, method='mutual_info')
        
        logger.info(f"Selected {len(selected_features)} features from {len(feature_columns)}")
        
        return X[selected_features], y, selected_features
        
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        raise


def train_models(X, y):
    """训练模型"""
    try:
        # 初始化训练器
        trainer = ModelTrainer()
        
        # 分割数据
        X_train, X_test, y_train, y_test = trainer.split_data(
            X, y, 
            test_size=1-MODEL_CONFIG["train_test_split_ratio"], 
            random_state=MODEL_CONFIG["random_seed"]
        )
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
        # 定义要训练的模型
        models = {
            'xgboost': {
                'type': 'XGBoostModel',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': MODEL_CONFIG["random_seed"]
                }
            },
            'lightgbm': {
                'type': 'LightGBMModel', 
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': MODEL_CONFIG["random_seed"]
                }
            },
            'ensemble': {
                'type': 'EnsembleModel',
                'params': {
                    'random_state': MODEL_CONFIG["random_seed"]
                }
            }
        }
        
        trained_models = {}
        
        for model_name, model_config in models.items():
            logger.info(f"Training {model_name} model...")
            
            # 训练模型
            model = trainer.train_model(
                X_train, y_train,
                model_type=model_config['type'],
                model_params=model_config['params']
            )
            
            if model is not None:
                trained_models[model_name] = model
                logger.info(f"Successfully trained {model_name}")
            else:
                logger.warning(f"Failed to train {model_name}")
        
        return trained_models, (X_train, X_test, y_train, y_test)
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise


def evaluate_models(models, data_splits, selected_features):
    """评估模型"""
    try:
        X_train, X_test, y_train, y_test = data_splits
        
        evaluator = ModelEvaluator()
        evaluation_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name} model...")
            
            # 预测
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # 评估
            train_metrics = evaluator.calculate_regression_metrics(y_train, y_pred_train)
            test_metrics = evaluator.calculate_regression_metrics(y_test, y_pred_test)
            
            evaluation_results[model_name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }
            
            logger.info(f"{model_name} - Train R2: {train_metrics['r2']:.4f}, Test R2: {test_metrics['r2']:.4f}")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error evaluating models: {e}")
        raise


def save_models_and_results(models, evaluation_results, selected_features):
    """保存模型和结果"""
    try:
        # 创建模型保存目录
        model_dir = Path(MODEL_CONFIG["save_path"])
        model_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        for model_name, model in models.items():
            model_path = model_dir / f"{model_name}_{timestamp}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} model to {model_path}")
        
        # 保存评估结果
        results_path = model_dir / f"evaluation_results_{timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        # 保存特征列表
        features_path = model_dir / f"selected_features_{timestamp}.json"
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(selected_features, f, indent=2, ensure_ascii=False)
        
        # 保存训练配置
        config = {
            'timestamp': timestamp,
            'model_config': MODEL_CONFIG,
            'selected_features_count': len(selected_features),
            'models_trained': list(models.keys())
        }
        
        config_path = model_dir / f"training_config_{timestamp}.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved training results with timestamp: {timestamp}")
        
        return timestamp
        
    except Exception as e:
        logger.error(f"Error saving models and results: {e}")
        raise


def main():
    """主函数"""
    logger.info("Starting model training process...")
    
    try:
        # 1. 加载数据
        data = load_data()
        
        # 2. 准备特征和目标变量
        features_data = prepare_features_and_targets(data)
        
        # 3. 特征选择
        X, y, selected_features = select_features(features_data)
        
        if X is None:
            logger.error("Feature selection failed")
            return
        
        # 4. 训练模型
        models, data_splits = train_models(X, y)
        
        if not models:
            logger.error("No models were trained successfully")
            return
        
        # 5. 评估模型
        evaluation_results = evaluate_models(models, data_splits, selected_features)
        
        # 6. 保存模型和结果
        timestamp = save_models_and_results(models, evaluation_results, selected_features)
        
        # 7. 输出最佳模型
        best_model_name = max(evaluation_results.keys(), 
                             key=lambda x: evaluation_results[x]['test_metrics']['r2'])
        
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best R2 score: {evaluation_results[best_model_name]['test_metrics']['r2']:.4f}")
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise


if __name__ == "__main__":
    main() 