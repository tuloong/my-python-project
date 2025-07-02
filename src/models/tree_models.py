"""
树模型实现

包含XGBoost、LightGBM、CatBoost等树模型
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost回归模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.default_params = {
            'objective': 'reg:squarederror',
            'random_state': random_seed,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    
    def build_model(self) -> xgb.XGBRegressor:
        """构建XGBoost模型"""
        params = {**self.default_params, **self.config}
        self.model = xgb.XGBRegressor(**params)
        return self.model
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None,
            early_stopping_rounds: Optional[int] = 50,
            verbose: bool = False,
            **kwargs) -> 'XGBoostModel':
        """训练XGBoost模型"""
        if self.model is None:
            self.build_model()
        
        self.feature_names = list(X.columns)
        
        # 设置验证集
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X, y), (X_val, y_val)]
        
        # 训练模型
        self.model.fit(
            X, y,
            eval_set=eval_set if eval_set else None,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            **kwargs
        )
        
        self.is_fitted = True
        
        # 记录训练指标
        if hasattr(self.model, 'evals_result_'):
            self.training_metrics.update(self.model.evals_result_)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: pd.DataFrame, n_estimators: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """预测并估计不确定性"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 使用不同数量的树进行预测来估计不确定性
        predictions = []
        n_est = n_estimators or self.model.n_estimators
        
        for i in range(10, n_est + 1, max(1, n_est // 10)):
            pred = self.model.predict(X, ntree_limit=i)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        
        return mean_pred, uncertainty


class LightGBMModel(BaseModel):
    """LightGBM回归模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.default_params = {
            'objective': 'regression',
            'random_state': random_seed,
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'verbose': -1
        }
    
    def build_model(self) -> lgb.LGBMRegressor:
        """构建LightGBM模型"""
        params = {**self.default_params, **self.config}
        self.model = lgb.LGBMRegressor(**params)
        return self.model
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            early_stopping_rounds: Optional[int] = 50,
            verbose: bool = False,
            **kwargs) -> 'LightGBMModel':
        """训练LightGBM模型"""
        if self.model is None:
            self.build_model()
        
        self.feature_names = list(X.columns)
        
        # 设置验证集
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X, y), (X_val, y_val)]
        
        # 训练模型
        self.model.fit(
            X, y,
            eval_set=eval_set if eval_set else None,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            **kwargs
        )
        
        self.is_fitted = True
        
        # 记录训练指标
        if hasattr(self.model, 'evals_result_'):
            self.training_metrics.update(self.model.evals_result_)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)


class CatBoostModel(BaseModel):
    """CatBoost回归模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.default_params = {
            'loss_function': 'RMSE',
            'random_seed': random_seed,
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3,
            'verbose': False
        }
    
    def build_model(self) -> cb.CatBoostRegressor:
        """构建CatBoost模型"""
        params = {**self.default_params, **self.config}
        self.model = cb.CatBoostRegressor(**params)
        return self.model
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            early_stopping_rounds: Optional[int] = 50,
            verbose: bool = False,
            cat_features: Optional[list] = None,
            **kwargs) -> 'CatBoostModel':
        """训练CatBoost模型"""
        if self.model is None:
            self.build_model()
        
        self.feature_names = list(X.columns)
        
        # 设置验证集
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        
        # 训练模型
        self.model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            cat_features=cat_features,
            **kwargs
        )
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """预测并返回不确定性"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # CatBoost支持预测不确定性
        predictions = self.model.predict(X)
        
        # 使用virtual ensembles来估计不确定性
        try:
            uncertainty = self.model.get_prediction_diff(X, ntree_end=self.model.tree_count_)
            uncertainty = np.abs(uncertainty)
        except:
            # 如果不支持，返回零不确定性
            uncertainty = np.zeros_like(predictions)
        
        return predictions, uncertainty


class ExtraTreesModel(BaseModel):
    """ExtraTrees回归模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.default_params = {
            'n_estimators': 100,
            'random_state': random_seed,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto'
        }
    
    def build_model(self):
        """构建ExtraTrees模型"""
        from sklearn.ensemble import ExtraTreesRegressor
        params = {**self.default_params, **self.config}
        self.model = ExtraTreesRegressor(**params)
        return self.model
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'ExtraTreesModel':
        """训练ExtraTrees模型"""
        if self.model is None:
            self.build_model()
        
        self.feature_names = list(X.columns)
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """预测并返回不确定性"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 使用所有树的预测结果计算不确定性
        predictions = []
        for tree in self.model.estimators_:
            pred = tree.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        
        return mean_pred, uncertainty 