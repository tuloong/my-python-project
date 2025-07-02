"""
特征工程器

负责从原始数据生成机器学习特征
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import talib


class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.features = []
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建价格相关特征"""
        result = df.copy()
        
        # 价格变化率
        for window in [1, 3, 5, 10, 20]:
            result[f'return_{window}d'] = result['close'].pct_change(window)
            result[f'high_low_ratio_{window}d'] = (result['high'] / result['low']).rolling(window).mean()
        
        # 价格位置（当前价格在N日内的分位数）
        for window in [10, 20, 50]:
            result[f'price_position_{window}d'] = result['close'].rolling(window).rank(pct=True)
        
        # 振幅
        result['amplitude'] = (result['high'] - result['low']) / result['close']
        
        return result
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建成交量相关特征"""
        result = df.copy()
        
        # 成交量变化率
        for window in [1, 3, 5, 10]:
            result[f'volume_change_{window}d'] = result['volume'].pct_change(window)
            result[f'volume_ma_{window}d'] = result['volume'].rolling(window).mean()
        
        # 成交量价格关系
        result['volume_price_corr'] = result['volume'].rolling(20).corr(result['close'])
        result['money_flow'] = result['volume'] * result['close']
        
        return result
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建技术指标特征"""
        result = df.copy()
        
        # 移动平均线
        for window in [5, 10, 20, 50]:
            result[f'sma_{window}'] = talib.SMA(result['close'].values, timeperiod=window)
            result[f'ema_{window}'] = talib.EMA(result['close'].values, timeperiod=window)
        
        # MACD
        exp1 = result['close'].ewm(span=12).mean()
        exp2 = result['close'].ewm(span=26).mean()
        result['macd'] = exp1 - exp2
        result['macd_signal'] = result['macd'].ewm(span=9).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        # RSI
        result['rsi'] = talib.RSI(result['close'].values, timeperiod=14)
        
        # 布林带
        result['bb_upper'], result['bb_middle'], result['bb_lower'] = talib.BBANDS(
            result['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        result['bb_position'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        # KDJ
        result['k'], result['d'] = talib.STOCH(
            result['high'].values, 
            result['low'].values, 
            result['close'].values,
            fastk_period=9, slowk_period=3, slowd_period=3
        )
        result['j'] = 3 * result['k'] - 2 * result['d']
        
        return result
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """创建滞后特征"""
        result = df.copy()
        
        # 滞后特征
        for lag in [1, 2, 3, 5, 10]:
            result[f'{target_col}_lag_{lag}'] = result[target_col].shift(lag)
        
        # 滚动统计特征
        for window in [5, 10, 20]:
            result[f'{target_col}_mean_{window}'] = result[target_col].rolling(window).mean()
            result[f'{target_col}_std_{window}'] = result[target_col].rolling(window).std()
            result[f'{target_col}_max_{window}'] = result[target_col].rolling(window).max()
            result[f'{target_col}_min_{window}'] = result[target_col].rolling(window).min()
        
        return result
    
    def create_time_features(self, df: pd.DataFrame, time_col: str = 'trade_date') -> pd.DataFrame:
        """创建时间特征"""
        result = df.copy()
        
        if time_col in result.columns:
            result[time_col] = pd.to_datetime(result[time_col])
            
            # 基本时间特征
            result['year'] = result[time_col].dt.year
            result['month'] = result[time_col].dt.month
            result['day'] = result[time_col].dt.day
            result['weekday'] = result[time_col].dt.weekday
            result['quarter'] = result[time_col].dt.quarter
            
            # 季节性特征
            result['is_month_start'] = result[time_col].dt.is_month_start
            result['is_month_end'] = result[time_col].dt.is_month_end
            result['is_quarter_start'] = result[time_col].dt.is_quarter_start
            result['is_quarter_end'] = result[time_col].dt.is_quarter_end
        
        return result
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """综合特征工程"""
        result = df.copy()
        
        # 创建各类特征
        result = self.create_price_features(result)
        result = self.create_volume_features(result)
        result = self.create_technical_indicators(result)
        result = self.create_lag_features(result)
        result = self.create_time_features(result)
        
        # 记录生成的特征列
        self.features = [col for col in result.columns if col not in df.columns]
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """获取生成的特征名称"""
        return self.features 