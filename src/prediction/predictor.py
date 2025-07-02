"""
预测器

负责生成股价预测结果并输出竞赛要求的格式
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os
from datetime import datetime, timedelta

from ..models.base_model import BaseModel
from ..utils.logger import get_logger


class Predictor:
    """股价预测器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.models = {}
        self.predictions = {}
        
    def load_models(self, model_dict: Dict[str, BaseModel]) -> None:
        """加载训练好的模型"""
        self.models = model_dict
        self.logger.info(f"加载了 {len(self.models)} 个模型")
    
    def predict_single_model(self, 
                           model: BaseModel,
                           X: pd.DataFrame,
                           model_name: str = None) -> np.ndarray:
        """使用单个模型进行预测"""
        
        if not model.is_fitted:
            raise ValueError(f"模型 {model_name or model.__class__.__name__} 未训练")
        
        predictions = model.predict(X)
        
        if model_name:
            self.predictions[model_name] = predictions
        
        return predictions
    
    def predict_ensemble(self, 
                        X: pd.DataFrame,
                        method: str = 'average',
                        weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """集成预测"""
        
        if not self.models:
            raise ValueError("没有加载任何模型")
        
        all_predictions = {}
        
        # 获取所有模型的预测
        for model_name, model in self.models.items():
            try:
                pred = self.predict_single_model(model, X, model_name)
                all_predictions[model_name] = pred
            except Exception as e:
                self.logger.warning(f"模型 {model_name} 预测失败: {str(e)}")
                continue
        
        if not all_predictions:
            raise ValueError("所有模型预测都失败了")
        
        # 集成预测结果
        if method == 'average':
            # 简单平均
            predictions_array = np.array(list(all_predictions.values()))
            ensemble_pred = np.mean(predictions_array, axis=0)
            
        elif method == 'weighted_average':
            # 加权平均
            if weights is None:
                weights = {name: 1.0 for name in all_predictions.keys()}
            
            weighted_sum = np.zeros_like(list(all_predictions.values())[0])
            total_weight = 0
            
            for model_name, pred in all_predictions.items():
                weight = weights.get(model_name, 1.0)
                weighted_sum += pred * weight
                total_weight += weight
            
            ensemble_pred = weighted_sum / total_weight
            
        elif method == 'median':
            # 中位数
            predictions_array = np.array(list(all_predictions.values()))
            ensemble_pred = np.median(predictions_array, axis=0)
            
        else:
            raise ValueError(f"不支持的集成方法: {method}")
        
        self.predictions['ensemble'] = ensemble_pred
        self.logger.info(f"集成预测完成，使用方法: {method}")
        
        return ensemble_pred
    
    def predict_stock_returns(self, 
                             X: pd.DataFrame,
                             stock_codes: List[str],
                             current_prices: Optional[pd.Series] = None,
                             ensemble_method: str = 'average',
                             weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """预测股票收益率"""
        
        # 获取集成预测结果
        if len(self.models) == 1:
            # 如果只有一个模型，直接使用
            model_name, model = list(self.models.items())[0]
            predictions = self.predict_single_model(model, X, model_name)
        else:
            # 多个模型使用集成方法
            predictions = self.predict_ensemble(X, ensemble_method, weights)
        
        # 创建结果DataFrame
        result_df = pd.DataFrame({
            'stock_code': stock_codes,
            'predicted_return': predictions
        })
        
        # 如果提供了当前价格，计算预测价格
        if current_prices is not None:
            result_df['current_price'] = current_prices.values
            result_df['predicted_price'] = result_df['current_price'] * (1 + result_df['predicted_return'])
        
        # 添加预测时间
        result_df['prediction_time'] = datetime.now()
        
        self.logger.info(f"完成 {len(stock_codes)} 只股票的收益率预测")
        
        return result_df
    
    def get_top_stocks(self, 
                      predictions_df: pd.DataFrame,
                      top_n: int = 10,
                      return_type: str = 'both') -> Dict[str, pd.DataFrame]:
        """获取涨跌幅最大的股票"""
        
        # 按预测收益率排序
        sorted_df = predictions_df.sort_values('predicted_return', ascending=False)
        
        results = {}
        
        if return_type in ['both', 'top']:
            # 涨幅最大的股票
            top_gainers = sorted_df.head(top_n).copy()
            top_gainers['rank'] = range(1, len(top_gainers) + 1)
            top_gainers['category'] = '涨幅最大'
            results['top_gainers'] = top_gainers
        
        if return_type in ['both', 'bottom']:
            # 跌幅最大的股票
            top_losers = sorted_df.tail(top_n).copy()
            top_losers = top_losers.sort_values('predicted_return', ascending=True)
            top_losers['rank'] = range(1, len(top_losers) + 1)
            top_losers['category'] = '跌幅最大'
            results['top_losers'] = top_losers
        
        self.logger.info(f"筛选出涨跌幅最大的前 {top_n} 只股票")
        
        return results
    
    def generate_competition_output(self, 
                                  predictions_df: pd.DataFrame,
                                  output_path: str,
                                  top_n: int = 10) -> None:
        """生成竞赛提交格式的输出文件"""
        
        # 获取涨跌幅最大的股票
        top_stocks = self.get_top_stocks(predictions_df, top_n)
        
        # 创建提交格式
        submission_data = []
        
        # 涨幅最大的股票
        if 'top_gainers' in top_stocks:
            for _, row in top_stocks['top_gainers'].iterrows():
                submission_data.append({
                    'stock_code': row['stock_code'],
                    'predicted_return': row['predicted_return'],
                    'rank': row['rank'],
                    'category': 'top_gainers'
                })
        
        # 跌幅最大的股票
        if 'top_losers' in top_stocks:
            for _, row in top_stocks['top_losers'].iterrows():
                submission_data.append({
                    'stock_code': row['stock_code'],
                    'predicted_return': row['predicted_return'],
                    'rank': row['rank'],
                    'category': 'top_losers'
                })
        
        # 保存结果
        submission_df = pd.DataFrame(submission_data)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为CSV文件
        submission_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # 同时保存详细的预测结果
        detailed_path = output_path.replace('.csv', '_detailed.csv')
        predictions_df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"竞赛提交文件已保存: {output_path}")
        self.logger.info(f"详细预测结果已保存: {detailed_path}")
        
        # 打印结果摘要
        self._print_prediction_summary(top_stocks)
    
    def _print_prediction_summary(self, top_stocks: Dict[str, pd.DataFrame]) -> None:
        """打印预测结果摘要"""
        
        print("\n" + "="*50)
        print("股价预测结果摘要")
        print("="*50)
        
        if 'top_gainers' in top_stocks:
            print("\n涨幅最大的股票:")
            print("-" * 30)
            for _, row in top_stocks['top_gainers'].iterrows():
                print(f"{row['rank']}. {row['stock_code']}: {row['predicted_return']:.4f} ({row['predicted_return']*100:.2f}%)")
        
        if 'top_losers' in top_stocks:
            print("\n跌幅最大的股票:")
            print("-" * 30)
            for _, row in top_stocks['top_losers'].iterrows():
                print(f"{row['rank']}. {row['stock_code']}: {row['predicted_return']:.4f} ({row['predicted_return']*100:.2f}%)")
        
        print("="*50)
    
    def analyze_predictions(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """分析预测结果"""
        
        analysis = {
            'total_stocks': len(predictions_df),
            'mean_return': predictions_df['predicted_return'].mean(),
            'std_return': predictions_df['predicted_return'].std(),
            'min_return': predictions_df['predicted_return'].min(),
            'max_return': predictions_df['predicted_return'].max(),
            'positive_returns': (predictions_df['predicted_return'] > 0).sum(),
            'negative_returns': (predictions_df['predicted_return'] < 0).sum(),
            'zero_returns': (predictions_df['predicted_return'] == 0).sum()
        }
        
        # 计算分位数
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        for q in quantiles:
            analysis[f'quantile_{int(q*100)}'] = predictions_df['predicted_return'].quantile(q)
        
        self.logger.info("预测结果分析完成")
        
        return analysis
    
    def save_predictions(self, 
                        predictions_df: pd.DataFrame,
                        output_dir: str,
                        filename: str = None) -> str:
        """保存预测结果"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.csv"
        
        output_path = os.path.join(output_dir, filename)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存预测结果
        predictions_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"预测结果已保存: {output_path}")
        
        return output_path
    
    def get_prediction_confidence(self, 
                                predictions_df: pd.DataFrame,
                                uncertainty_threshold: float = 0.05) -> pd.DataFrame:
        """获取预测置信度（如果模型支持不确定性估计）"""
        
        if not self.models:
            raise ValueError("没有加载任何模型")
        
        # 尝试获取不确定性估计
        uncertainty_estimates = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_with_uncertainty'):
                try:
                    X_last = predictions_df[['stock_code']].copy()  # 简化示例
                    _, uncertainty = model.predict_with_uncertainty(X_last)
                    uncertainty_estimates[model_name] = uncertainty
                except:
                    continue
        
        if uncertainty_estimates:
            # 计算平均不确定性
            avg_uncertainty = np.mean(list(uncertainty_estimates.values()), axis=0)
            
            predictions_df = predictions_df.copy()
            predictions_df['uncertainty'] = avg_uncertainty
            predictions_df['confidence'] = 1 - (avg_uncertainty / (avg_uncertainty.max() + 1e-8))
            predictions_df['high_confidence'] = predictions_df['uncertainty'] < uncertainty_threshold
            
            self.logger.info("添加了预测置信度信息")
        
        return predictions_df 