"""
线性模型实现

包含线性回归、Ridge、Lasso等线性模型
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, LassoCV, RidgeCV, ElasticNetCV
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from .base_model import BaseModel


class LinearModel(BaseModel):
    """线性回归模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.scaler = StandardScaler()
        self.use_scaling = config.get('use_scaling', True) if config else True
    
    def build_model(self) -> LinearRegression:
        """构建线性回归模型"""
        self.model = LinearRegression()
        return self.model
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'LinearModel':
        """训练线性回归模型"""
        if self.model is None:
            self.build_model()
        
        self.feature_names = list(X.columns)
        
        # 数据标准化
        if self.use_scaling:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # 训练模型
        self.model.fit(X_scaled, y, **kwargs)
        self.is_fitted = True
        
        # 计算训练指标
        train_pred = self.predict(X)
        self.training_metrics = self.evaluate(X, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 应用相同的预处理
        if self.use_scaling:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)


class RidgeModel(BaseModel):
    """Ridge回归模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.scaler = StandardScaler()
        self.default_params = {
            'alpha': 1.0,
            'random_state': random_seed
        }
        self.use_scaling = config.get('use_scaling', True) if config else True
    
    def build_model(self) -> Ridge:
        """构建Ridge回归模型"""
        params = {**self.default_params, **self.config}
        self.model = Ridge(**params)
        return self.model
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'RidgeModel':
        """训练Ridge回归模型"""
        if self.model is None:
            self.build_model()
        
        self.feature_names = list(X.columns)
        
        # 数据标准化
        if self.use_scaling:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # 训练模型
        self.model.fit(X_scaled, y, **kwargs)
        self.is_fitted = True
        
        # 计算训练指标
        self.training_metrics = self.evaluate(X, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 应用相同的预处理
        if self.use_scaling:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)


class LassoModel(BaseModel):
    """Lasso回归模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.scaler = StandardScaler()
        self.default_params = {
            'alpha': 1.0,
            'random_state': random_seed,
            'max_iter': 1000
        }
        self.use_scaling = config.get('use_scaling', True) if config else True
    
    def build_model(self) -> Lasso:
        """构建Lasso回归模型"""
        params = {**self.default_params, **self.config}
        self.model = Lasso(**params)
        return self.model
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'LassoModel':
        """训练Lasso回归模型"""
        if self.model is None:
            self.build_model()
        
        self.feature_names = list(X.columns)
        
        # 数据标准化
        if self.use_scaling:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # 训练模型
        self.model.fit(X_scaled, y, **kwargs)
        self.is_fitted = True
        
        # 计算训练指标
        self.training_metrics = self.evaluate(X, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 应用相同的预处理
        if self.use_scaling:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)


class ElasticNetModel(BaseModel):
    """ElasticNet回归模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.scaler = StandardScaler()
        self.default_params = {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'random_state': random_seed,
            'max_iter': 1000
        }
        self.use_scaling = config.get('use_scaling', True) if config else True
    
    def build_model(self) -> ElasticNet:
        """构建ElasticNet回归模型"""
        params = {**self.default_params, **self.config}
        self.model = ElasticNet(**params)
        return self.model
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'ElasticNetModel':
        """训练ElasticNet回归模型"""
        if self.model is None:
            self.build_model()
        
        self.feature_names = list(X.columns)
        
        # 数据标准化
        if self.use_scaling:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # 训练模型
        self.model.fit(X_scaled, y, **kwargs)
        self.is_fitted = True
        
        # 计算训练指标
        self.training_metrics = self.evaluate(X, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 应用相同的预处理
        if self.use_scaling:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)


class BayesianRidgeModel(BaseModel):
    """贝叶斯Ridge回归模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.scaler = StandardScaler()
        self.default_params = {
            'alpha_1': 1e-6,
            'alpha_2': 1e-6,
            'lambda_1': 1e-6,
            'lambda_2': 1e-6
        }
        self.use_scaling = config.get('use_scaling', True) if config else True
    
    def build_model(self) -> BayesianRidge:
        """构建贝叶斯Ridge回归模型"""
        params = {**self.default_params, **self.config}
        self.model = BayesianRidge(**params)
        return self.model
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BayesianRidgeModel':
        """训练贝叶斯Ridge回归模型"""
        if self.model is None:
            self.build_model()
        
        self.feature_names = list(X.columns)
        
        # 数据标准化
        if self.use_scaling:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # 训练模型
        self.model.fit(X_scaled, y, **kwargs)
        self.is_fitted = True
        
        # 计算训练指标
        self.training_metrics = self.evaluate(X, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 应用相同的预处理
        if self.use_scaling:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """预测并返回不确定性"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 应用相同的预处理
        if self.use_scaling:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # 贝叶斯模型可以返回预测不确定性
        predictions, std = self.model.predict(X_scaled, return_std=True)
        
        return predictions, std


class PolynomialModel(BaseModel):
    """多项式回归模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.degree = config.get('degree', 2) if config else 2
        self.include_bias = config.get('include_bias', False) if config else False
        self.alpha = config.get('alpha', 1.0) if config else 1.0
        
    def build_model(self) -> Pipeline:
        """构建多项式回归模型"""
        # 创建多项式特征 + Ridge回归的管道
        self.model = Pipeline([
            ('poly', PolynomialFeatures(
                degree=self.degree, 
                include_bias=self.include_bias
            )),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=self.alpha, random_state=self.random_seed))
        ])
        return self.model
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'PolynomialModel':
        """训练多项式回归模型"""
        if self.model is None:
            self.build_model()
        
        self.feature_names = list(X.columns)
        
        # 训练模型
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        
        # 计算训练指标
        self.training_metrics = self.evaluate(X, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X) 