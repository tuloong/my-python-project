"""
基础模型类

定义所有机器学习模型的通用接口
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score


class BaseModel(ABC):
    """所有模型的基类"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        self.config = config or {}
        self.random_seed = random_seed
        self.model = None
        self.is_fitted = False
        self.feature_names = []
        self.training_metrics = {}
        
    @abstractmethod
    def build_model(self) -> Any:
        """构建模型实例"""
        pass
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseModel':
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """评估模型"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        predictions = self.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions)
        }
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """交叉验证"""
        if self.model is None:
            self.build_model()
        
        # 获取交叉验证分数
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_squared_error')
        
        return {
            'cv_mse_mean': -scores.mean(),
            'cv_mse_std': scores.std(),
            'cv_rmse_mean': np.sqrt(-scores.mean()),
            'cv_rmse_std': np.sqrt(scores.std())
        }
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """获取特征重要性"""
        if not self.is_fitted:
            return None
            
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(self.model, 'coef_'):
            importance = dict(zip(self.feature_names, np.abs(self.model.coef_)))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        else:
            return None
    
    def save_model(self, filepath: str) -> None:
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型和元数据
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'config': self.config,
            'model_type': self.__class__.__name__
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> 'BaseModel':
        """加载模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data.get('feature_names', [])
        self.training_metrics = model_data.get('training_metrics', {})
        self.config = model_data.get('config', {})
        self.is_fitted = True
        
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        if self.model is None:
            return {}
        return self.model.get_params() if hasattr(self.model, 'get_params') else {}
    
    def set_params(self, **params) -> 'BaseModel':
        """设置模型参数"""
        if self.model is None:
            self.build_model()
        
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        
        return self
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """预测并返回不确定性（如果模型支持）"""
        predictions = self.predict(X)
        
        # 对于不支持不确定性的模型，返回零不确定性
        uncertainty = np.zeros_like(predictions)
        
        return predictions, uncertainty
    
    def get_training_metrics(self) -> Dict[str, float]:
        """获取训练过程中的指标"""
        return self.training_metrics
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__() 