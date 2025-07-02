"""
特征选择器

提供多种特征选择方法
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import warnings


class FeatureSelector:
    """特征选择器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.selected_features = []
        self.feature_scores = {}
        
    def select_by_correlation(self, 
                            X: pd.DataFrame, 
                            y: pd.Series, 
                            threshold: float = 0.1) -> List[str]:
        """基于相关性选择特征"""
        correlations = X.corrwith(y).abs()
        selected = correlations[correlations >= threshold].index.tolist()
        
        self.feature_scores['correlation'] = correlations.to_dict()
        return selected
    
    def select_by_mutual_info(self, 
                             X: pd.DataFrame, 
                             y: pd.Series, 
                             k: int = 50) -> List[str]:
        """基于互信息选择特征"""
        # 处理缺失值
        X_clean = X.fillna(X.median())
        
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(X_clean, y)
        
        feature_scores = dict(zip(X.columns, selector.scores_))
        selected = X.columns[selector.get_support()].tolist()
        
        self.feature_scores['mutual_info'] = feature_scores
        return selected
    
    def select_by_f_score(self, 
                         X: pd.DataFrame, 
                         y: pd.Series, 
                         k: int = 50) -> List[str]:
        """基于F分数选择特征"""
        # 处理缺失值
        X_clean = X.fillna(X.median())
        
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X_clean, y)
        
        feature_scores = dict(zip(X.columns, selector.scores_))
        selected = X.columns[selector.get_support()].tolist()
        
        self.feature_scores['f_score'] = feature_scores
        return selected
    
    def select_by_rfe(self, 
                     X: pd.DataFrame, 
                     y: pd.Series, 
                     n_features: int = 50,
                     estimator=None) -> List[str]:
        """基于递归特征消除选择特征"""
        if estimator is None:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # 处理缺失值
        X_clean = X.fillna(X.median())
        
        selector = RFE(estimator, n_features_to_select=n_features)
        selector.fit(X_clean, y)
        
        selected = X.columns[selector.get_support()].tolist()
        
        # 获取特征排名
        feature_ranking = dict(zip(X.columns, selector.ranking_))
        self.feature_scores['rfe_ranking'] = feature_ranking
        
        return selected
    
    def select_by_lasso(self, 
                       X: pd.DataFrame, 
                       y: pd.Series, 
                       alpha: float = None) -> List[str]:
        """基于Lasso回归选择特征"""
        # 处理缺失值
        X_clean = X.fillna(X.median())
        
        if alpha is None:
            # 使用交叉验证选择最优alpha
            lasso = LassoCV(cv=5, random_state=42)
        else:
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=alpha, random_state=42)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso.fit(X_clean, y)
        
        # 选择系数非零的特征
        selected = X.columns[np.abs(lasso.coef_) > 0].tolist()
        
        feature_scores = dict(zip(X.columns, np.abs(lasso.coef_)))
        self.feature_scores['lasso'] = feature_scores
        
        return selected
    
    def select_by_importance(self, 
                           X: pd.DataFrame, 
                           y: pd.Series, 
                           estimator=None,
                           threshold: float = 0.001) -> List[str]:
        """基于模型重要性选择特征"""
        if estimator is None:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # 处理缺失值
        X_clean = X.fillna(X.median())
        
        selector = SelectFromModel(estimator, threshold=threshold)
        selector.fit(X_clean, y)
        
        selected = X.columns[selector.get_support()].tolist()
        
        # 获取特征重要性
        if hasattr(estimator, 'feature_importances_'):
            feature_scores = dict(zip(X.columns, estimator.feature_importances_))
            self.feature_scores['importance'] = feature_scores
        
        return selected
    
    def remove_correlated_features(self, 
                                 X: pd.DataFrame, 
                                 threshold: float = 0.9) -> List[str]:
        """移除高相关性特征"""
        corr_matrix = X.corr().abs()
        
        # 找出高相关性的特征对
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # 找出需要删除的特征
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        # 返回保留的特征
        selected = [col for col in X.columns if col not in to_drop]
        
        return selected
    
    def select_features(self, 
                       X: pd.DataFrame, 
                       y: pd.Series, 
                       method: str = 'mutual_info',
                       **kwargs) -> List[str]:
        """综合特征选择"""
        if method == 'correlation':
            selected = self.select_by_correlation(X, y, **kwargs)
        elif method == 'mutual_info':
            selected = self.select_by_mutual_info(X, y, **kwargs)
        elif method == 'f_score':
            selected = self.select_by_f_score(X, y, **kwargs)
        elif method == 'rfe':
            selected = self.select_by_rfe(X, y, **kwargs)
        elif method == 'lasso':
            selected = self.select_by_lasso(X, y, **kwargs)
        elif method == 'importance':
            selected = self.select_by_importance(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # 移除高相关性特征
        corr_threshold = kwargs.get('corr_threshold', 0.9)
        if corr_threshold < 1.0:
            X_selected = X[selected]
            selected = self.remove_correlated_features(X_selected, corr_threshold)
        
        self.selected_features = selected
        return selected
    
    def get_feature_scores(self, method: str = None) -> Dict[str, float]:
        """获取特征评分"""
        if method is None:
            return self.feature_scores
        else:
            return self.feature_scores.get(method, {})
    
    def get_selected_features(self) -> List[str]:
        """获取选择的特征"""
        return self.selected_features 