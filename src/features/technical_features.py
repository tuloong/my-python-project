"""
技术指标特征

专门用于计算金融市场技术指标
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, List


class TechnicalFeatures:
    """技术指标特征计算器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def calculate_ma_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算移动平均类指标"""
        result = df.copy()
        
        # 简单移动平均
        for period in [5, 10, 20, 30, 60]:
            result[f'sma_{period}'] = talib.SMA(result['close'].values, timeperiod=period)
            result[f'price_sma_{period}_ratio'] = result['close'] / result[f'sma_{period}']
        
        # 指数移动平均
        for period in [5, 10, 20, 30]:
            result[f'ema_{period}'] = talib.EMA(result['close'].values, timeperiod=period)
            result[f'price_ema_{period}_ratio'] = result['close'] / result[f'ema_{period}']
        
        # 加权移动平均
        for period in [5, 10, 20]:
            result[f'wma_{period}'] = talib.WMA(result['close'].values, timeperiod=period)
        
        # 三重指数移动平均
        result['tema'] = talib.TEMA(result['close'].values, timeperiod=30)
        
        return result
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量类指标"""
        result = df.copy()
        
        # RSI
        for period in [6, 12, 24]:
            result[f'rsi_{period}'] = talib.RSI(result['close'].values, timeperiod=period)
        
        # 动量指标
        for period in [10, 20]:
            result[f'momentum_{period}'] = talib.MOM(result['close'].values, timeperiod=period)
        
        # 变化率
        for period in [10, 20]:
            result[f'roc_{period}'] = talib.ROC(result['close'].values, timeperiod=period)
        
        # Williams %R
        for period in [14, 21]:
            result[f'willr_{period}'] = talib.WILLR(
                result['high'].values, 
                result['low'].values, 
                result['close'].values, 
                timeperiod=period
            )
        
        # 商品通道指数
        result['cci'] = talib.CCI(
            result['high'].values, 
            result['low'].values, 
            result['close'].values, 
            timeperiod=14
        )
        
        return result
    
    def calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动性指标"""
        result = df.copy()
        
        # 布林带
        for period in [20, 26]:
            upper, middle, lower = talib.BBANDS(
                result['close'].values, 
                timeperiod=period, 
                nbdevup=2, 
                nbdevdn=2
            )
            result[f'bb_upper_{period}'] = upper
            result[f'bb_middle_{period}'] = middle
            result[f'bb_lower_{period}'] = lower
            result[f'bb_width_{period}'] = (upper - lower) / middle
            result[f'bb_position_{period}'] = (result['close'] - lower) / (upper - lower)
        
        # 平均真实范围
        for period in [14, 21]:
            result[f'atr_{period}'] = talib.ATR(
                result['high'].values, 
                result['low'].values, 
                result['close'].values, 
                timeperiod=period
            )
        
        # 真实范围
        result['tr'] = talib.TRANGE(
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )
        
        return result
    
    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算趋势类指标"""
        result = df.copy()
        
        # MACD
        macd, signal, histogram = talib.MACD(
            result['close'].values, 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_histogram'] = histogram
        
        # ADX (平均趋向指数)
        result['adx'] = talib.ADX(
            result['high'].values, 
            result['low'].values, 
            result['close'].values, 
            timeperiod=14
        )
        
        # Aroon
        aroon_down, aroon_up = talib.AROON(
            result['high'].values, 
            result['low'].values, 
            timeperiod=14
        )
        result['aroon_down'] = aroon_down
        result['aroon_up'] = aroon_up
        result['aroon_oscillator'] = aroon_up - aroon_down
        
        # Parabolic SAR
        result['sar'] = talib.SAR(
            result['high'].values, 
            result['low'].values, 
            acceleration=0.02, 
            maximum=0.2
        )
        
        return result
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量指标"""
        result = df.copy()
        
        # 成交量移动平均
        for period in [5, 10, 20]:
            result[f'volume_sma_{period}'] = talib.SMA(result['volume'].values, timeperiod=period)
            result[f'volume_ratio_{period}'] = result['volume'] / result[f'volume_sma_{period}']
        
        # 资金流量指数
        result['mfi'] = talib.MFI(
            result['high'].values,
            result['low'].values,
            result['close'].values,
            result['volume'].values,
            timeperiod=14
        )
        
        # 成交量确认指标
        result['obv'] = talib.OBV(result['close'].values, result['volume'].values)
        
        # 成交量加权平均价
        result['vwap'] = (result['volume'] * result['close']).cumsum() / result['volume'].cumsum()
        
        return result
    
    def calculate_overlap_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算重叠研究指标"""
        result = df.copy()
        
        # 中位数价格
        result['medprice'] = talib.MEDPRICE(result['high'].values, result['low'].values)
        
        # 典型价格
        result['typprice'] = talib.TYPPRICE(
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )
        
        # 加权收盘价
        result['wclprice'] = talib.WCLPRICE(
            result['high'].values, 
            result['low'].values, 
            result['close'].values
        )
        
        return result
    
    def calculate_cycle_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算周期类指标"""
        result = df.copy()
        
        # Hilbert Transform - Instantaneous Trendline
        try:
            result['ht_trendline'] = talib.HT_TRENDLINE(result['close'].values)
        except:
            result['ht_trendline'] = np.nan
        
        # Hilbert Transform - Trend vs Cycle Mode
        try:
            result['ht_trendmode'] = talib.HT_TRENDMODE(result['close'].values)
        except:
            result['ht_trendmode'] = np.nan
        
        return result
    
    def calculate_pattern_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算模式识别指标"""
        result = df.copy()
        
        # 十字星
        result['doji'] = talib.CDLDOJI(
            result['open'].values,
            result['high'].values,
            result['low'].values,
            result['close'].values
        )
        
        # 锤子线
        result['hammer'] = talib.CDLHAMMER(
            result['open'].values,
            result['high'].values,
            result['low'].values,
            result['close'].values
        )
        
        # 上吊线
        result['hanging_man'] = talib.CDLHANGINGMAN(
            result['open'].values,
            result['high'].values,
            result['low'].values,
            result['close'].values
        )
        
        return result
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        result = df.copy()
        
        # 计算各类指标
        result = self.calculate_ma_indicators(result)
        result = self.calculate_momentum_indicators(result)
        result = self.calculate_volatility_indicators(result)
        result = self.calculate_trend_indicators(result)
        result = self.calculate_volume_indicators(result)
        result = self.calculate_overlap_indicators(result)
        result = self.calculate_cycle_indicators(result)
        result = self.calculate_pattern_indicators(result)
        
        return result
    
    def get_indicator_names(self) -> List[str]:
        """获取所有技术指标名称"""
        indicators = []
        
        # 移动平均类
        for period in [5, 10, 20, 30, 60]:
            indicators.extend([f'sma_{period}', f'price_sma_{period}_ratio'])
        for period in [5, 10, 20, 30]:
            indicators.extend([f'ema_{period}', f'price_ema_{period}_ratio'])
        for period in [5, 10, 20]:
            indicators.append(f'wma_{period}')
        indicators.append('tema')
        
        # 动量类
        for period in [6, 12, 24]:
            indicators.append(f'rsi_{period}')
        for period in [10, 20]:
            indicators.extend([f'momentum_{period}', f'roc_{period}'])
        for period in [14, 21]:
            indicators.append(f'willr_{period}')
        indicators.append('cci')
        
        # 波动性
        for period in [20, 26]:
            indicators.extend([
                f'bb_upper_{period}', f'bb_middle_{period}', f'bb_lower_{period}',
                f'bb_width_{period}', f'bb_position_{period}'
            ])
        for period in [14, 21]:
            indicators.append(f'atr_{period}')
        indicators.append('tr')
        
        # 趋势类
        indicators.extend(['macd', 'macd_signal', 'macd_histogram', 'adx', 
                          'aroon_down', 'aroon_up', 'aroon_oscillator', 'sar'])
        
        # 成交量类
        for period in [5, 10, 20]:
            indicators.extend([f'volume_sma_{period}', f'volume_ratio_{period}'])
        indicators.extend(['mfi', 'obv', 'vwap'])
        
        # 重叠研究
        indicators.extend(['medprice', 'typprice', 'wclprice'])
        
        # 周期类
        indicators.extend(['ht_trendline', 'ht_trendmode'])
        
        # 模式识别
        indicators.extend(['doji', 'hammer', 'hanging_man'])
        
        return indicators 