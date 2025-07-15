"""
模型训练器

负责管理模型训练流程
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime

from ..models.base_model import BaseModel
from ..utils.logger import get_logger


class Trainer:
    """模型训练器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.models = {}
        self.training_history = {}
        
    def prepare_data(self, 
                    X: pd.DataFrame, 
                    y: pd.Series,
                    test_size: float = 0.2,
                    validation_size: float = 0.1,
                    time_series_split: bool = False,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """准备训练、验证和测试数据"""
        
        if time_series_split:
            # 时间序列分割
            n_samples = len(X)
            train_end = int(n_samples * (1 - test_size - validation_size))
            val_end = int(n_samples * (1 - test_size))
            
            X_train = X.iloc[:train_end]
            X_val = X.iloc[train_end:val_end]
            X_test = X.iloc[val_end:]
            
            y_train = y.iloc[:train_end]
            y_val = y.iloc[train_end:val_end]
            y_test = y.iloc[val_end:]
        else:
            # 随机分割
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            if validation_size > 0:
                val_ratio = validation_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_ratio, random_state=random_state
                )
            else:
                X_train, X_val = X_temp, pd.DataFrame()
                y_train, y_val = y_temp, pd.Series()
        
        self.logger.info(f"数据分割完成 - 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_single_model(self, 
                          model: BaseModel,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None,
                          model_name: Optional[str] = None,
                          **kwargs) -> BaseModel:
        """训练单个模型"""
        
        if model_name is None:
            model_name = model.__class__.__name__
        
        self.logger.info(f"开始训练模型: {model_name}")
        
        # 训练模型
        start_time = datetime.now()
        
        if X_val is not None and y_val is not None and len(X_val) > 0:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val, **kwargs)
        else:
            model.fit(X_train, y_train, **kwargs)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # 评估模型
        train_metrics = model.evaluate(X_train, y_train)
        
        val_metrics = {}
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_metrics = model.evaluate(X_val, y_val)
        
        # 记录训练历史
        self.training_history[model_name] = {
            'training_time': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_config': model.config,
            'feature_count': len(model.feature_names) if model.feature_names else 0
        }
        
        # 保存模型
        self.models[model_name] = model
        
        self.logger.info(f"模型 {model_name} 训练完成")
        self.logger.info(f"训练指标: {train_metrics}")
        if val_metrics:
            self.logger.info(f"验证指标: {val_metrics}")
        
        return model
    
    def train_multiple_models(self, 
                             models: Dict[str, BaseModel],
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_val: Optional[pd.DataFrame] = None,
                             y_val: Optional[pd.Series] = None,
                             **kwargs) -> Dict[str, BaseModel]:
        """训练多个模型"""
        
        self.logger.info(f"开始训练 {len(models)} 个模型")
        
        trained_models = {}
        
        for model_name, model in models.items():
            try:
                trained_model = self.train_single_model(
                    model, X_train, y_train, X_val, y_val, model_name, **kwargs
                )
                trained_models[model_name] = trained_model
            except Exception as e:
                self.logger.error(f"训练模型 {model_name} 时出错: {str(e)}")
                continue
        
        self.logger.info(f"成功训练 {len(trained_models)} 个模型")
        
        return trained_models
    
    def cross_validate_model(self, 
                            model: BaseModel,
                            X: pd.DataFrame,
                            y: pd.Series,
                            cv: int = 5,
                            time_series: bool = False) -> Dict[str, float]:
        """交叉验证模型"""
        
        self.logger.info(f"开始交叉验证模型: {model.__class__.__name__}")
        
        if time_series:
            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=cv)
            splits = tscv.split(X)
        else:
            # 标准k折交叉验证
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            splits = kf.split(X)
        
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 创建模型副本
            model_copy = model.__class__(model.config, model.random_seed)
            
            # 训练模型
            model_copy.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model_copy.predict(X_val)
            
            fold_scores = {
                'mse': mean_squared_error(y_val, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                'mae': mean_absolute_error(y_val, y_pred),
                'r2': r2_score(y_val, y_pred)
            }
            
            scores.append(fold_scores)
            self.logger.info(f"第 {fold+1} 折验证完成: RMSE={fold_scores['rmse']:.4f}")
        
        # 计算平均分数和标准差
        avg_scores = {}
        for metric in scores[0].keys():
            values = [score[metric] for score in scores]
            avg_scores[f'{metric}_mean'] = np.mean(values)
            avg_scores[f'{metric}_std'] = np.std(values)
        
        self.logger.info(f"交叉验证完成: RMSE={avg_scores['rmse_mean']:.4f}±{avg_scores['rmse_std']:.4f}")
        
        return avg_scores
    
    def get_best_model(self, metric: str = 'rmse', validation: bool = True) -> Tuple[str, BaseModel]:
        """获取最佳模型"""
        
        if not self.training_history:
            raise ValueError("没有训练过的模型")
        
        best_score = float('inf') if metric in ['mse', 'rmse', 'mae'] else float('-inf')
        best_model_name = None
        
        for model_name, history in self.training_history.items():
            metrics = history['val_metrics'] if validation and history['val_metrics'] else history['train_metrics']
            
            if metric not in metrics:
                continue
            
            score = metrics[metric]
            
            if metric in ['mse', 'rmse', 'mae']:
                if score < best_score:
                    best_score = score
                    best_model_name = model_name
            else:  # r2 等指标越大越好
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError(f"没有找到包含指标 {metric} 的模型")
        
        self.logger.info(f"最佳模型: {best_model_name}, {metric}={best_score:.4f}")
        
        return best_model_name, self.models[best_model_name]
    
    def save_models(self, save_dir: str) -> None:
        """保存所有训练好的模型"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{model_name}.pkl")
            model.save_model(model_path)
            
        # 保存训练历史
        history_path = os.path.join(save_dir, "training_history.pkl")
        joblib.dump(self.training_history, history_path)
        
        self.logger.info(f"所有模型已保存到: {save_dir}")
    
    def load_models(self, save_dir: str) -> Dict[str, BaseModel]:
        """加载训练好的模型"""
        
        models = {}
        
        # 加载训练历史
        history_path = os.path.join(save_dir, "training_history.pkl")
        if os.path.exists(history_path):
            self.training_history = joblib.load(history_path)
        
        # 加载模型文件
        for filename in os.listdir(save_dir):
            if filename.endswith('.pkl') and filename != 'training_history.pkl':
                model_name = filename[:-4]  # 移除.pkl扩展名
                model_path = os.path.join(save_dir, filename)
                
                # 根据模型名称创建相应的模型实例
                try:
                    model = self._create_model_instance(model_name)
                    model.load_model(model_path)
                    models[model_name] = model
                    self.models[model_name] = model
                except Exception as e:
                    self.logger.error(f"加载模型 {model_name} 失败: {str(e)}")
        
        self.logger.info(f"成功加载 {len(models)} 个模型")
        
        return models
    
    def _create_model_instance(self, model_name: str) -> BaseModel:
        """根据模型名称创建模型实例"""
        from ..models import (
            XGBoostModel, LightGBMModel, CatBoostModel, ExtraTreesModel,
            LinearModel, RidgeModel, LassoModel, 
            ElasticNetModel, BayesianRidgeModel, PolynomialModel,
            EnsembleModel, StackingEnsemble, LSTMModel, GRUModel
        )
        
        model_classes = {
            'XGBoostModel': XGBoostModel,
            'LightGBMModel': LightGBMModel,
            'CatBoostModel': CatBoostModel,
            'ExtraTreesModel': ExtraTreesModel,
            'LinearModel': LinearModel,
            'RidgeModel': RidgeModel,
            'LassoModel': LassoModel,
            'ElasticNetModel': ElasticNetModel,
            'BayesianRidgeModel': BayesianRidgeModel,
            'PolynomialModel': PolynomialModel,
            'EnsembleModel': EnsembleModel,
            'StackingEnsemble': StackingEnsemble,
            'LSTMModel': LSTMModel,
            'GRUModel': GRUModel
        }
        
        for class_name, model_class in model_classes.items():
            if class_name in model_name:
                return model_class()
        
        # 默认返回XGBoost模型
        return XGBoostModel()
    
    def get_training_summary(self) -> pd.DataFrame:
        """获取训练结果摘要"""
        
        if not self.training_history:
            return pd.DataFrame()
        
        summary_data = []
        
        for model_name, history in self.training_history.items():
            row = {
                'model_name': model_name,
                'training_time': history['training_time'],
                'feature_count': history['feature_count']
            }
            
            # 添加训练指标
            for metric, value in history['train_metrics'].items():
                row[f'train_{metric}'] = value
            
            # 添加验证指标
            for metric, value in history['val_metrics'].items():
                row[f'val_{metric}'] = value
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        return summary_df 