"""
超参数优化器

支持多种超参数优化算法
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
import optuna
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error
import joblib
import os
from datetime import datetime

from ..models.base_model import BaseModel
from ..utils.logger import get_logger


class HyperOptimizer:
    """超参数优化器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.study = None
        self.best_params = {}
        self.optimization_history = []
        
    def define_search_space(self, model_type: str) -> Dict[str, Any]:
        """定义搜索空间"""
        
        search_spaces = {
            'xgboost': {
                'n_estimators': ('int', 50, 500),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('float', 0.01, 0.3),
                'subsample': ('float', 0.6, 1.0),
                'colsample_bytree': ('float', 0.6, 1.0),
                'reg_alpha': ('float', 0, 10),
                'reg_lambda': ('float', 0, 10),
            },
            'lightgbm': {
                'n_estimators': ('int', 50, 500),
                'max_depth': ('int', 3, 15),
                'learning_rate': ('float', 0.01, 0.3),
                'num_leaves': ('int', 10, 300),
                'subsample': ('float', 0.6, 1.0),
                'colsample_bytree': ('float', 0.6, 1.0),
                'reg_alpha': ('float', 0, 10),
                'reg_lambda': ('float', 0, 10),
            },
            'catboost': {
                'iterations': ('int', 50, 500),
                'depth': ('int', 3, 10),
                'learning_rate': ('float', 0.01, 0.3),
                'l2_leaf_reg': ('float', 1, 10),
            },
            'ridge': {
                'alpha': ('float', 0.1, 100),
            },
            'lasso': {
                'alpha': ('float', 0.001, 10),
                'max_iter': ('int', 500, 2000),
            },
            'random_forest': {
                'n_estimators': ('int', 50, 300),
                'max_depth': ('int', 5, 20),
                'min_samples_split': ('int', 2, 20),
                'min_samples_leaf': ('int', 1, 10),
                'max_features': ('categorical', ['auto', 'sqrt', 'log2']),
            }
        }
        
        return search_spaces.get(model_type.lower(), {})
    
    def suggest_parameters(self, trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """根据搜索空间生成参数建议"""
        
        params = {}
        
        for param_name, param_config in search_space.items():
            param_type = param_config[0]
            
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config[1])
            elif param_type == 'log_uniform':
                params[param_name] = trial.suggest_loguniform(param_name, param_config[1], param_config[2])
        
        return params
    
    def objective_function(self, 
                          trial: optuna.Trial,
                          model_class: type,
                          X: pd.DataFrame,
                          y: pd.Series,
                          search_space: Dict[str, Any],
                          cv: int = 5,
                          time_series: bool = False,
                          random_seed: int = 42) -> float:
        """目标函数"""
        
        # 生成参数建议
        params = self.suggest_parameters(trial, search_space)
        
        # 创建模型
        model = model_class(config=params, random_seed=random_seed)
        model.build_model()
        
        # 交叉验证
        if time_series:
            cv_splitter = TimeSeriesSplit(n_splits=cv)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_seed)
        
        # 计算交叉验证分数
        scores = cross_val_score(
            model.model, X, y, 
            cv=cv_splitter, 
            scoring='neg_mean_squared_error'
        )
        
        # 返回平均RMSE
        rmse = np.sqrt(-scores.mean())
        
        # 记录历史
        self.optimization_history.append({
            'trial': trial.number,
            'params': params,
            'rmse': rmse,
            'std': np.sqrt(scores.std())
        })
        
        return rmse
    
    def optimize(self, 
                model_class: type,
                X: pd.DataFrame,
                y: pd.Series,
                model_type: str = None,
                n_trials: int = 100,
                cv: int = 5,
                time_series: bool = False,
                random_seed: int = 42,
                search_space: Dict[str, Any] = None,
                direction: str = 'minimize') -> Dict[str, Any]:
        """执行超参数优化"""
        
        if model_type is None:
            model_type = model_class.__name__.lower().replace('model', '')
        
        if search_space is None:
            search_space = self.define_search_space(model_type)
        
        if not search_space:
            self.logger.warning(f"未找到模型 {model_type} 的搜索空间，使用默认参数")
            return {}
        
        self.logger.info(f"开始超参数优化: {model_type}")
        self.logger.info(f"搜索空间: {search_space}")
        self.logger.info(f"试验次数: {n_trials}")
        
        # 创建Optuna研究
        study_name = f"{model_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=random_seed)
        )
        
        # 定义目标函数
        def objective(trial):
            return self.objective_function(
                trial, model_class, X, y, search_space, cv, time_series, random_seed
            )
        
        # 执行优化
        self.study.optimize(objective, n_trials=n_trials)
        
        # 获取最佳参数
        self.best_params = self.study.best_params
        best_score = self.study.best_value
        
        self.logger.info(f"超参数优化完成")
        self.logger.info(f"最佳参数: {self.best_params}")
        self.logger.info(f"最佳分数 (RMSE): {best_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': best_score,
            'n_trials': len(self.study.trials),
            'study': self.study
        }
    
    def optimize_multiple_models(self, 
                               model_configs: Dict[str, Dict[str, Any]],
                               X: pd.DataFrame,
                               y: pd.Series,
                               n_trials: int = 50,
                               cv: int = 5,
                               time_series: bool = False,
                               random_seed: int = 42) -> Dict[str, Dict[str, Any]]:
        """优化多个模型的超参数"""
        
        results = {}
        
        for model_name, config in model_configs.items():
            try:
                self.logger.info(f"开始优化模型: {model_name}")
                
                model_class = config['model_class']
                model_type = config.get('model_type', model_name)
                search_space = config.get('search_space')
                
                result = self.optimize(
                    model_class=model_class,
                    X=X,
                    y=y,
                    model_type=model_type,
                    n_trials=n_trials,
                    cv=cv,
                    time_series=time_series,
                    random_seed=random_seed,
                    search_space=search_space
                )
                
                results[model_name] = result
                
            except Exception as e:
                self.logger.error(f"优化模型 {model_name} 时出错: {str(e)}")
                continue
        
        return results
    
    def get_optimization_history(self) -> pd.DataFrame:
        """获取优化历史"""
        
        if not self.optimization_history:
            return pd.DataFrame()
        
        history_df = pd.DataFrame(self.optimization_history)
        
        # 展开参数字典为单独的列
        params_df = pd.json_normalize(history_df['params'])
        history_df = pd.concat([history_df.drop('params', axis=1), params_df], axis=1)
        
        return history_df
    
    def plot_optimization_history(self, save_path: str = None) -> None:
        """绘制优化历史"""
        
        if self.study is None:
            self.logger.warning("没有优化历史可绘制")
            return
        
        import matplotlib.pyplot as plt
        
        # 获取所有试验的分数
        trials = self.study.trials
        scores = [trial.value for trial in trials if trial.value is not None]
        trial_numbers = [trial.number for trial in trials if trial.value is not None]
        
        if not scores:
            self.logger.warning("没有有效的试验结果")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 优化过程
        axes[0, 0].plot(trial_numbers, scores)
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('Optimization Progress')
        
        # 添加最佳分数线
        best_score = min(scores)
        axes[0, 0].axhline(y=best_score, color='r', linestyle='--', label=f'Best: {best_score:.4f}')
        axes[0, 0].legend()
        
        # 2. 分数分布
        axes[0, 1].hist(scores, bins=30, alpha=0.7)
        axes[0, 1].axvline(x=best_score, color='r', linestyle='--', label=f'Best: {best_score:.4f}')
        axes[0, 1].set_xlabel('RMSE')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Score Distribution')
        axes[0, 1].legend()
        
        # 3. 参数重要性
        try:
            importance = optuna.importance.get_param_importances(self.study)
            if importance:
                params = list(importance.keys())
                importances = list(importance.values())
                
                axes[1, 0].barh(params, importances)
                axes[1, 0].set_xlabel('Importance')
                axes[1, 0].set_title('Parameter Importance')
        except:
            axes[1, 0].text(0.5, 0.5, 'Parameter importance\nnot available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. 收敛历史
        best_scores = []
        current_best = float('inf')
        for score in scores:
            if score < current_best:
                current_best = score
            best_scores.append(current_best)
        
        axes[1, 1].plot(trial_numbers, best_scores)
        axes[1, 1].set_xlabel('Trial')
        axes[1, 1].set_ylabel('Best RMSE')
        axes[1, 1].set_title('Convergence History')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"优化历史图表已保存: {save_path}")
        
        plt.show()
    
    def save_study(self, save_path: str) -> None:
        """保存优化研究"""
        
        if self.study is None:
            self.logger.warning("没有研究可保存")
            return
        
        study_data = {
            'study': self.study,
            'best_params': self.best_params,
            'optimization_history': self.optimization_history
        }
        
        joblib.dump(study_data, save_path)
        self.logger.info(f"优化研究已保存: {save_path}")
    
    def load_study(self, save_path: str) -> None:
        """加载优化研究"""
        
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"研究文件不存在: {save_path}")
        
        study_data = joblib.load(save_path)
        
        self.study = study_data['study']
        self.best_params = study_data['best_params']
        self.optimization_history = study_data['optimization_history']
        
        self.logger.info(f"优化研究已加载: {save_path}")
    
    def get_best_model(self, 
                      model_class: type,
                      random_seed: int = 42) -> BaseModel:
        """获取最佳参数的模型实例"""
        
        if not self.best_params:
            raise ValueError("没有最佳参数，请先执行优化")
        
        model = model_class(config=self.best_params, random_seed=random_seed)
        
        return model
    
    def suggest_next_trial(self) -> Dict[str, Any]:
        """建议下一次试验的参数"""
        
        if self.study is None:
            raise ValueError("没有进行中的研究")
        
        # 创建一个新的试验来获取参数建议
        trial = self.study.ask()
        
 