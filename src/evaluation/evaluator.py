"""
模型评估器

包含详细的评估指标和分析功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from scipy import stats
import warnings

from ..models.base_model import BaseModel
from ..utils.logger import get_logger


class Evaluator:
    """模型评估器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.evaluation_results = {}
        
    def evaluate_regression(self, 
                           y_true: pd.Series, 
                           y_pred: np.ndarray,
                           model_name: str = "model") -> Dict[str, float]:
        """回归模型基础评估"""
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }
        
        # 计算MAPE（平均绝对百分比误差）
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        except:
            # 如果y_true中有0值，手动计算MAPE
            non_zero_mask = y_true != 0
            if non_zero_mask.sum() > 0:
                metrics['mape'] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]))
            else:
                metrics['mape'] = np.inf
        
        # 计算调整R²
        n = len(y_true)
        p = 1  # 假设为单变量回归，实际使用时应传入特征数
        if n > p + 1:
            metrics['adj_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
        else:
            metrics['adj_r2'] = metrics['r2']
        
        # 计算相关系数
        metrics['pearson_corr'] = np.corrcoef(y_true, y_pred)[0, 1]
        metrics['spearman_corr'] = stats.spearmanr(y_true, y_pred)[0]
        
        # 计算最大误差
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        
        # 计算中位数绝对误差
        metrics['median_ae'] = np.median(np.abs(y_true - y_pred))
        
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def evaluate_financial_metrics(self, 
                                 y_true: pd.Series, 
                                 y_pred: np.ndarray,
                                 model_name: str = "model") -> Dict[str, float]:
        """金融相关的评估指标"""
        
        # 计算方向准确率（预测涨跌方向的准确性）
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        direction_accuracy = np.mean(true_direction == pred_direction)
        
        # 计算分层准确率（按收益率大小分层）
        quantiles = [0.2, 0.4, 0.6, 0.8]
        quantile_accuracies = {}
        
        for i, q in enumerate(quantiles):
            if i == 0:
                mask = y_true <= np.quantile(y_true, q)
            else:
                mask = (y_true > np.quantile(y_true, quantiles[i-1])) & (y_true <= np.quantile(y_true, q))
            
            if mask.sum() > 0:
                quantile_accuracies[f'q{i+1}_accuracy'] = np.mean(true_direction[mask] == pred_direction[mask])
        
        # 计算信息系数（IC）
        ic = np.corrcoef(y_true, y_pred)[0, 1]
        
        # 计算信息比率（IR）
        ic_std = np.std([np.corrcoef(y_true.iloc[i:i+20], y_pred[i:i+20])[0, 1] 
                        for i in range(0, len(y_true)-20, 5) if not np.isnan(np.corrcoef(y_true.iloc[i:i+20], y_pred[i:i+20])[0, 1])])
        ir = ic / ic_std if ic_std > 0 else 0
        
        # 计算累计收益（假设按预测排序买入前N只股票）
        top_n = min(10, len(y_true) // 10)  # 前10%的股票
        top_indices = np.argsort(y_pred)[-top_n:]
        portfolio_return = np.mean(y_true.iloc[top_indices])
        
        # 计算夏普比率（简化版本）
        returns = y_true.iloc[top_indices]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        financial_metrics = {
            'direction_accuracy': direction_accuracy,
            'information_coefficient': ic,
            'information_ratio': ir,
            'portfolio_return': portfolio_return,
            'sharpe_ratio': sharpe_ratio,
            **quantile_accuracies
        }
        
        # 合并到已有评估结果中
        if model_name in self.evaluation_results:
            self.evaluation_results[model_name].update(financial_metrics)
        else:
            self.evaluation_results[model_name] = financial_metrics
        
        return financial_metrics
    
    def evaluate_model(self, 
                      model: BaseModel,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      model_name: str = None) -> Dict[str, float]:
        """全面评估单个模型"""
        
        if model_name is None:
            model_name = model.__class__.__name__
        
        # 进行预测
        y_pred = model.predict(X_test)
        
        # 基础回归指标
        basic_metrics = self.evaluate_regression(y_test, y_pred, model_name)
        
        # 金融指标
        financial_metrics = self.evaluate_financial_metrics(y_test, y_pred, model_name)
        
        # 合并所有指标
        all_metrics = {**basic_metrics, **financial_metrics}
        
        self.logger.info(f"模型 {model_name} 评估完成")
        self.logger.info(f"RMSE: {all_metrics['rmse']:.4f}, R²: {all_metrics['r2']:.4f}, 方向准确率: {all_metrics['direction_accuracy']:.4f}")
        
        return all_metrics
    
    def evaluate_multiple_models(self, 
                                models: Dict[str, BaseModel],
                                X_test: pd.DataFrame,
                                y_test: pd.Series) -> pd.DataFrame:
        """评估多个模型"""
        
        self.logger.info(f"开始评估 {len(models)} 个模型")
        
        for model_name, model in models.items():
            try:
                self.evaluate_model(model, X_test, y_test, model_name)
            except Exception as e:
                self.logger.error(f"评估模型 {model_name} 时出错: {str(e)}")
                continue
        
        # 返回评估结果汇总
        return self.get_evaluation_summary()
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """获取评估结果汇总"""
        
        if not self.evaluation_results:
            return pd.DataFrame()
        
        summary_df = pd.DataFrame(self.evaluation_results).T
        
        # 按某个指标排序（如RMSE升序）
        if 'rmse' in summary_df.columns:
            summary_df = summary_df.sort_values('rmse')
        
        return summary_df
    
    def plot_predictions(self, 
                        y_true: pd.Series, 
                        y_pred: np.ndarray,
                        model_name: str = "Model",
                        save_path: str = None) -> None:
        """绘制预测结果图表"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 真实值 vs 预测值散点图
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'{model_name}: True vs Predicted')
        
        # 添加R²信息
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. 残差图
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title(f'{model_name}: Residuals vs Predicted')
        
        # 3. 残差直方图
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, density=True)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title(f'{model_name}: Residuals Distribution')
        
        # 添加正态分布拟合
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[1, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Normal fit (μ={mu:.3f}, σ={sigma:.3f})')
        axes[1, 0].legend()
        
        # 4. Q-Q图（正态性检验）
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'{model_name}: Q-Q Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"预测图表已保存: {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, 
                             models: List[str] = None,
                             metrics: List[str] = None,
                             save_path: str = None) -> None:
        """绘制模型对比图表"""
        
        if not self.evaluation_results:
            self.logger.warning("没有评估结果可供对比")
            return
        
        summary_df = self.get_evaluation_summary()
        
        if models is None:
            models = summary_df.index.tolist()
        
        if metrics is None:
            metrics = ['rmse', 'r2', 'direction_accuracy', 'information_coefficient']
        
        # 过滤可用的指标
        available_metrics = [m for m in metrics if m in summary_df.columns]
        
        if not available_metrics:
            self.logger.warning("没有可用的评估指标")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            data = summary_df.loc[models, metric]
            
            bars = axes[i].bar(range(len(models)), data.values)
            axes[i].set_xlabel('Models')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_title(f'Model Comparison: {metric.upper()}')
            axes[i].set_xticks(range(len(models)))
            axes[i].set_xticklabels(models, rotation=45, ha='right')
            
            # 添加数值标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"模型对比图表已保存: {save_path}")
        
        plt.show()
    
    def feature_importance_analysis(self, 
                                  model: BaseModel,
                                  model_name: str = None) -> pd.DataFrame:
        """特征重要性分析"""
        
        if model_name is None:
            model_name = model.__class__.__name__
        
        importance = model.get_feature_importance()
        
        if importance is None:
            self.logger.warning(f"模型 {model_name} 不支持特征重要性分析")
            return pd.DataFrame()
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': imp}
            for feature, imp in importance.items()
        ])
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        self.logger.info(f"模型 {model_name} 特征重要性分析完成")
        
        return importance_df
    
    def plot_feature_importance(self, 
                              model: BaseModel,
                              top_n: int = 20,
                              model_name: str = None,
                              save_path: str = None) -> None:
        """绘制特征重要性图表"""
        
        importance_df = self.feature_importance_analysis(model, model_name)
        
        if importance_df.empty:
            return
        
        # 取前N个重要特征
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, max(6, len(top_features) * 0.3)))
        
        bars = plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name or model.__class__.__name__}')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"特征重要性图表已保存: {save_path}")
        
        plt.show()
    
    def statistical_tests(self, 
                         y_true: pd.Series, 
                         y_pred: np.ndarray) -> Dict[str, Any]:
        """统计显著性测试"""
        
        residuals = y_true - y_pred
        
        tests = {}
        
        # Shapiro-Wilk正态性检验
        if len(residuals) <= 5000:  # Shapiro-Wilk测试样本限制
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            tests['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        
        # Jarque-Bera正态性检验
        jb_stat, jb_p = stats.jarque_bera(residuals)
        tests['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_p,
            'is_normal': jb_p > 0.05
        }
        
        # Durbin-Watson自相关检验
        from statsmodels.stats.diagnostic import durbin_watson
        dw_stat = durbin_watson(residuals)
        tests['durbin_watson'] = {
            'statistic': dw_stat,
            'no_autocorr': 1.5 < dw_stat < 2.5
        }
        
        # Breusch-Pagan异方差检验
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, np.column_stack([y_pred]))
            tests['breusch_pagan'] = {
                'statistic': bp_stat,
                'p_value': bp_p,
                'homoscedastic': bp_p > 0.05
            }
        except:
            pass
        
        return tests
    
    def generate_evaluation_report(self, 
                                 output_path: str,
                                 include_plots: bool = True) -> None:
        """生成评估报告"""
        
        if not self.evaluation_results:
            self.logger.warning("没有评估结果可生成报告")
            return
        
        summary_df = self.get_evaluation_summary()
        
        # 保存评估结果CSV
        csv_path = output_path.replace('.txt', '.csv')
        summary_df.to_csv(csv_path)
        
        # 生成文本报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("评估指标汇总:\n")
            f.write("-" * 30 + "\n")
            f.write(summary_df.to_string())
            f.write("\n\n")
            
            # 最佳模型
            if 'rmse' in summary_df.columns:
                best_model = summary_df['rmse'].idxmin()
                f.write(f"最佳模型 (基于RMSE): {best_model}\n")
                f.write(f"RMSE: {summary_df.loc[best_model, 'rmse']:.4f}\n")
                if 'r2' in summary_df.columns:
                    f.write(f"R²: {summary_df.loc[best_model, 'r2']:.4f}\n")
                if 'direction_accuracy' in summary_df.columns:
                    f.write(f"方向准确率: {summary_df.loc[best_model, 'direction_accuracy']:.4f}\n")
                f.write("\n")
            
            # 模型排名
            f.write("模型排名 (基于RMSE):\n")
            f.write("-" * 20 + "\n")
            for i, (model_name, row) in enumerate(summary_df.iterrows(), 1):
                f.write(f"{i}. {model_name}: RMSE={row.get('rmse', 'N/A'):.4f}\n")
        
        self.logger.info(f"评估报告已保存: {output_path}")
        
        if include_plots:
            plot_dir = output_path.replace('.txt', '_plots')
            import os
            os.makedirs(plot_dir, exist_ok=True)
            
            # 生成模型对比图
            comparison_path = os.path.join(plot_dir, 'model_comparison.png')
            self.plot_model_comparison(save_path=comparison_path) 