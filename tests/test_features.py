"""
特征工程测试

测试特征生成和选择功能
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.feature_engineer import FeatureEngineer
from src.features.feature_selector import FeatureSelector
from src.features.technical_features import TechnicalFeatures


class TestFeatures(unittest.TestCase):
    """特征工程测试类"""
    
    def setUp(self):
        """设置测试数据"""
        # 生成模拟股票数据
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        n_stocks = 10
        
        self.stock_data = []
        for i in range(n_stocks):
            stock_code = f'00000{i}'
            
            # 生成价格数据（随机游走）
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            
            # 生成成交量数据
            volumes = np.random.lognormal(15, 1, len(dates))
            
            stock_df = pd.DataFrame({
                'ts_code': stock_code,
                'trade_date': dates,
                'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                'close': prices,
                'vol': volumes,
                'amount': prices * volumes
            })
            
            self.stock_data.append(stock_df)
        
        self.data = pd.concat(self.stock_data, ignore_index=True)
        self.data['trade_date'] = pd.to_datetime(self.data['trade_date'])
    
    def test_feature_engineer_creation(self):
        """测试特征工程器创建"""
        fe = FeatureEngineer()
        self.assertIsNotNone(fe)
        self.assertIsInstance(fe.config, dict)
    
    def test_price_features(self):
        """测试价格特征生成"""
        fe = FeatureEngineer()
        
        # 生成价格特征
        features_df = fe.create_price_features(self.data)
        
        # 验证特征列存在
        expected_features = ['return_1d', 'return_5d', 'return_20d', 'log_return_1d']
        for feature in expected_features:
            self.assertIn(feature, features_df.columns)
        
        # 验证数据类型
        self.assertTrue(features_df['return_1d'].dtype in [np.float64, np.float32])
        
        # 验证无穷大值处理
        self.assertFalse(np.any(np.isinf(features_df['return_1d'])))
    
    def test_volume_features(self):
        """测试成交量特征生成"""
        fe = FeatureEngineer()
        
        # 生成成交量特征
        features_df = fe.create_volume_features(self.data)
        
        # 验证特征列存在
        expected_features = ['vol_ma_5', 'vol_ma_20', 'vol_ratio_5', 'vol_ratio_20']
        for feature in expected_features:
            self.assertIn(feature, features_df.columns)
    
    def test_technical_features_creation(self):
        """测试技术指标特征创建"""
        tf = TechnicalFeatures()
        
        # 取单只股票数据测试
        single_stock = self.data[self.data['ts_code'] == '000000'].copy()
        single_stock = single_stock.sort_values('trade_date').reset_index(drop=True)
        
        # 生成技术指标
        features_df = tf.create_technical_features(single_stock)
        
        # 验证基础技术指标
        basic_indicators = ['sma_5', 'sma_20', 'ema_12', 'ema_26', 'macd']
        for indicator in basic_indicators:
            self.assertIn(indicator, features_df.columns)
    
    def test_feature_selector_creation(self):
        """测试特征选择器创建"""
        fs = FeatureSelector()
        self.assertIsNotNone(fs)
        self.assertIsInstance(fs.config, dict)
    
    def test_correlation_selection(self):
        """测试相关性特征选择"""
        # 生成测试数据
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                        columns=[f'feature_{i}' for i in range(n_features)])
        y = pd.Series(np.random.randn(n_samples))
        
        # 创建一些高相关特征
        X['high_corr_1'] = y + np.random.randn(n_samples) * 0.1
        X['high_corr_2'] = y * 2 + np.random.randn(n_samples) * 0.2
        
        fs = FeatureSelector()
        
        # 执行相关性选择
        selected_features = fs.select_by_correlation(X, y, threshold=0.1, top_k=10)
        
        # 验证选择了特征
        self.assertGreater(len(selected_features), 0)
        self.assertLessEqual(len(selected_features), 10)
        
        # 验证高相关特征被选中
        self.assertIn('high_corr_1', selected_features)
        self.assertIn('high_corr_2', selected_features)
    
    def test_mutual_info_selection(self):
        """测试互信息特征选择"""
        # 生成测试数据
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                        columns=[f'feature_{i}' for i in range(n_features)])
        y = pd.Series(np.random.randn(n_samples))
        
        fs = FeatureSelector()
        
        # 执行互信息选择
        selected_features = fs.select_by_mutual_info(X, y, top_k=10)
        
        # 验证选择了特征
        self.assertGreater(len(selected_features), 0)
        self.assertLessEqual(len(selected_features), 10)
    
    def test_feature_engineering_pipeline(self):
        """测试特征工程流水线"""
        fe = FeatureEngineer()
        
        # 生成所有特征
        all_features = fe.create_all_features(self.data)
        
        # 验证输出
        self.assertIsInstance(all_features, pd.DataFrame)
        self.assertGreater(len(all_features.columns), len(self.data.columns))
        
        # 验证没有NaN值在关键列中
        key_features = [col for col in all_features.columns if 'return_' in col]
        for feature in key_features:
            # 允许前几行有NaN（由于滞后特征）
            non_null_ratio = all_features[feature].notna().sum() / len(all_features)
            self.assertGreater(non_null_ratio, 0.8)  # 至少80%的数据非空
    
    def test_lag_features(self):
        """测试滞后特征生成"""
        fe = FeatureEngineer()
        
        # 生成滞后特征
        lag_features = fe.create_lag_features(self.data, ['close', 'vol'], [1, 3, 5])
        
        # 验证滞后特征列
        expected_lag_cols = ['close_lag1', 'close_lag3', 'close_lag5', 
                           'vol_lag1', 'vol_lag3', 'vol_lag5']
        
        for col in expected_lag_cols:
            self.assertIn(col, lag_features.columns)


if __name__ == '__main__':
    unittest.main() 