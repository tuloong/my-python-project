"""
集成模型实现

包含多种模型的集成方法
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import (
    VotingRegressor, BaggingRegressor, 
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.model_selection import cross_val_predict
from .base_model import BaseModel
from .tree_models import XGBoostModel, LightGBMModel, CatBoostModel
from .linear_models import RidgeModel, LassoModel


class EnsembleModel(BaseModel):
    """集成模型基类"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.base_models = []
        self.model_weights = []
        self.ensemble_method = config.get('ensemble_method', 'voting') if config else 'voting'
        
    def add_model(self, model: BaseModel, weight: float = 1.0) -> 'EnsembleModel':
        """添加基础模型"""
        self.base_models.append(model)
        self.model_weights.append(weight)
        return self
        
    def build_model(self) -> Any:
        """构建集成模型"""
        if not self.base_models:
            self._create_default_models()
            
        if self.ensemble_method == 'voting':
            # 创建投票回归器
            estimators = [(f'model_{i}', model.model or model.build_model()) 
                         for i, model in enumerate(self.base_models)]
            self.model = VotingRegressor(estimators=estimators)
        else:
            # 使用自定义集成方法
            self.model = None
            
        return self.model
    
    def _create_default_models(self):
        """创建默认的基础模型"""
        # 添加默认模型
        models = [
            XGBoostModel(random_seed=self.random_seed),
            LightGBMModel(random_seed=self.random_seed),
            RidgeModel(random_seed=self.random_seed)
        ]
        
        for model in models:
            self.add_model(model)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'EnsembleModel':
        """训练集成模型"""
        self.feature_names = list(X.columns)
        
        if self.ensemble_method == 'voting':
            if self.model is None:
                self.build_model()
            self.model.fit(X, y, **kwargs)
        else:
            # 训练所有基础模型
            for model in self.base_models:
                model.fit(X, y, **kwargs)
                
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if self.ensemble_method == 'voting':
            return self.model.predict(X)
        elif self.ensemble_method == 'weighted_average':
            return self._weighted_average_predict(X)
        elif self.ensemble_method == 'stacking':
            return self._stacking_predict(X)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _weighted_average_predict(self, X: pd.DataFrame) -> np.ndarray:
        """加权平均预测"""
        predictions = []
        weights = []
        
        for model, weight in zip(self.base_models, self.model_weights):
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(weight)
        
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化权重
        
        return np.average(predictions, axis=0, weights=weights)
    
    def _stacking_predict(self, X: pd.DataFrame) -> np.ndarray:
        """堆叠预测（需要预先训练元学习器）"""
        if not hasattr(self, 'meta_learner'):
            raise ValueError("Meta learner not trained for stacking")
            
        # 获取所有基础模型的预测作为特征
        meta_features = []
        for model in self.base_models:
            pred = model.predict(X)
            meta_features.append(pred)
        
        meta_X = np.column_stack(meta_features)
        return self.meta_learner.predict(meta_X)


class StackingEnsemble(EnsembleModel):
    """堆叠集成模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.ensemble_method = 'stacking'
        self.meta_learner = None
        self.cv_folds = config.get('cv_folds', 5) if config else 5
        
    def build_model(self) -> Any:
        """构建堆叠模型"""
        if not self.base_models:
            self._create_default_models()
            
        # 元学习器使用Ridge回归
        meta_learner_config = self.config.get('meta_learner', {})
        self.meta_learner = RidgeModel(meta_learner_config, self.random_seed)
        
        return self.meta_learner
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'StackingEnsemble':
        """训练堆叠模型"""
        self.feature_names = list(X.columns)
        
        # 第一层：训练基础模型并生成交叉验证预测
        meta_features = []
        
        for model in self.base_models:
            # 使用交叉验证生成训练集预测
            if model.model is None:
                model.build_model()
            
            cv_predictions = cross_val_predict(
                model.model, X, y, cv=self.cv_folds, method='predict'
            )
            meta_features.append(cv_predictions)
            
            # 在全部数据上训练模型（用于最终预测）
            model.fit(X, y, **kwargs)
        
        # 第二层：训练元学习器
        meta_X = pd.DataFrame(np.column_stack(meta_features))
        meta_X.columns = [f'model_{i}_pred' for i in range(len(self.base_models))]
        
        if self.meta_learner is None:
            self.build_model()
        self.meta_learner.fit(meta_X, y)
        
        self.is_fitted = True
        return self


class BlendingEnsemble(EnsembleModel):
    """混合集成模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.ensemble_method = 'blending'
        self.meta_learner = None
        self.holdout_ratio = config.get('holdout_ratio', 0.2) if config else 0.2
        
    def build_model(self) -> Any:
        """构建混合模型"""
        if not self.base_models:
            self._create_default_models()
            
        # 元学习器使用Ridge回归
        meta_learner_config = self.config.get('meta_learner', {})
        self.meta_learner = RidgeModel(meta_learner_config, self.random_seed)
        
        return self.meta_learner
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BlendingEnsemble':
        """训练混合模型"""
        self.feature_names = list(X.columns)
        
        # 分割数据：一部分用于训练基础模型，一部分用于训练元学习器
        split_idx = int(len(X) * (1 - self.holdout_ratio))
        
        X_train, X_holdout = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_holdout = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 第一层：在训练集上训练基础模型
        meta_features = []
        
        for model in self.base_models:
            # 训练基础模型
            model.fit(X_train, y_train, **kwargs)
            
            # 在holdout集上生成预测
            holdout_pred = model.predict(X_holdout)
            meta_features.append(holdout_pred)
        
        # 第二层：训练元学习器
        meta_X = pd.DataFrame(np.column_stack(meta_features))
        meta_X.columns = [f'model_{i}_pred' for i in range(len(self.base_models))]
        
        if self.meta_learner is None:
            self.build_model()
        self.meta_learner.fit(meta_X, y_holdout)
        
        # 重新在全部数据上训练基础模型
        for model in self.base_models:
            model.fit(X, y, **kwargs)
        
        self.is_fitted = True
        return self


class BaggingEnsemble(BaseModel):
    """Bagging集成模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.base_estimator = config.get('base_estimator') if config else None
        self.n_estimators = config.get('n_estimators', 10) if config else 10
        self.max_samples = config.get('max_samples', 1.0) if config else 1.0
        self.max_features = config.get('max_features', 1.0) if config else 1.0
        
    def build_model(self) -> BaggingRegressor:
        """构建Bagging模型"""
        from sklearn.tree import DecisionTreeRegressor
        
        base_est = self.base_estimator or DecisionTreeRegressor(random_state=self.random_seed)
        
        self.model = BaggingRegressor(
            base_estimator=base_est,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=self.random_seed
        )
        return self.model
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaggingEnsemble':
        """训练Bagging模型"""
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
        
        # 使用所有估计器的预测来计算不确定性
        predictions = []
        for estimator in self.model.estimators_:
            pred = estimator.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        
        return mean_pred, uncertainty


class AdaBoostEnsemble(BaseModel):
    """AdaBoost集成模型（回归版本）"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        self.default_params = {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'loss': 'linear',
            'random_state': random_seed
        }
        
    def build_model(self):
        """构建AdaBoost模型"""
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        
        params = {**self.default_params, **self.config}
        
        # 使用决策树作为基学习器
        base_estimator = DecisionTreeRegressor(max_depth=3, random_state=self.random_seed)
        
        self.model = AdaBoostRegressor(
            base_estimator=base_estimator,
            **params
        )
        return self.model
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'AdaBoostEnsemble':
        """训练AdaBoost模型"""
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