"""
数据预处理模块

负责数据清洗、缺失值处理、异常值检测等
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_loader import DataLoader

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        """初始化预处理器"""
        self.scalers = {}
        self.outlier_thresholds = {}
    
    def clean_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗股票数据
        
        Args:
            data: 原始股票数据
            
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        cleaned_data = data.copy()
        
        # 1. 删除重复行
        cleaned_data = cleaned_data.drop_duplicates()
        
        # 2. 确保日期列是datetime类型
        if 'date' in cleaned_data.columns:
            cleaned_data['date'] = pd.to_datetime(cleaned_data['date'])
            cleaned_data = cleaned_data.sort_values('date')
        
        # 3. 检查并处理价格数据的异常值
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in cleaned_data.columns:
                # 删除负价格或零价格
                cleaned_data = cleaned_data[cleaned_data[col] > 0]
                
                # 检查价格逻辑关系
                if col == 'high':
                    # 最高价应该 >= 开盘价、收盘价、最低价
                    mask = (cleaned_data['high'] >= cleaned_data['low'])
                    if 'open' in cleaned_data.columns:
                        mask &= (cleaned_data['high'] >= cleaned_data['open'])
                    if 'close' in cleaned_data.columns:
                        mask &= (cleaned_data['high'] >= cleaned_data['close'])
                    cleaned_data = cleaned_data[mask]
                
                elif col == 'low':
                    # 最低价应该 <= 开盘价、收盘价、最高价
                    mask = (cleaned_data['low'] <= cleaned_data['high'])
                    if 'open' in cleaned_data.columns:
                        mask &= (cleaned_data['low'] <= cleaned_data['open'])
                    if 'close' in cleaned_data.columns:
                        mask &= (cleaned_data['low'] <= cleaned_data['close'])
                    cleaned_data = cleaned_data[mask]
        
        # 4. 处理成交量异常值
        if 'volume' in cleaned_data.columns:
            # 删除负成交量
            cleaned_data = cleaned_data[cleaned_data['volume'] >= 0]
        
        # 5. 重置索引
        cleaned_data = cleaned_data.reset_index(drop=True)
        
        logger.info(f"Data cleaning completed. {len(data)} -> {len(cleaned_data)} records")
        
        return cleaned_data
    
    def handle_missing_values(
        self, 
        data: pd.DataFrame, 
        method: str = "forward_fill"
    ) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            data: 输入数据
            method: 处理方法 ("forward_fill", "backward_fill", "interpolate", "drop")
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        processed_data = data.copy()
        
        # 记录原始缺失值情况
        missing_before = processed_data.isnull().sum()
        logger.info(f"Missing values before processing: {missing_before.sum()}")
        
        if method == "forward_fill":
            processed_data = processed_data.fillna(method='ffill')
        elif method == "backward_fill":
            processed_data = processed_data.fillna(method='bfill')
        elif method == "interpolate":
            # 对数值列进行插值
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_columns] = processed_data[numeric_columns].interpolate()
        elif method == "drop":
            processed_data = processed_data.dropna()
        else:
            logger.warning(f"Unknown method {method}, using forward_fill")
            processed_data = processed_data.fillna(method='ffill')
        
        # 记录处理后缺失值情况
        missing_after = processed_data.isnull().sum()
        logger.info(f"Missing values after processing: {missing_after.sum()}")
        
        return processed_data
    
    def detect_outliers(
        self, 
        data: pd.DataFrame, 
        columns: List[str], 
        method: str = "iqr",
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        检测异常值
        
        Args:
            data: 输入数据
            columns: 要检测的列
            method: 检测方法 ("iqr", "zscore", "isolation_forest")
            threshold: 阈值
            
        Returns:
            pd.DataFrame: 标记了异常值的数据
        """
        result_data = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if method == "iqr":
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                outlier_mask = z_scores > threshold
                
            else:
                logger.warning(f"Unknown outlier detection method: {method}")
                continue
            
            # 标记异常值
            result_data[f'{col}_outlier'] = outlier_mask
            
            outlier_count = outlier_mask.sum()
            logger.info(f"Detected {outlier_count} outliers in column {col}")
        
        return result_data
    
    def normalize_data(
        self, 
        data: pd.DataFrame, 
        columns: List[str], 
        method: str = "standard"
    ) -> pd.DataFrame:
        """
        数据标准化/归一化
        
        Args:
            data: 输入数据
            columns: 要标准化的列
            method: 标准化方法 ("standard", "minmax")
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        normalized_data = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
            
            if method == "standard":
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                
                normalized_data[col] = self.scalers[col].fit_transform(
                    data[col].values.reshape(-1, 1)
                ).flatten()
                
            elif method == "minmax":
                if col not in self.scalers:
                    self.scalers[col] = MinMaxScaler()
                
                normalized_data[col] = self.scalers[col].fit_transform(
                    data[col].values.reshape(-1, 1)
                ).flatten()
                
            else:
                logger.warning(f"Unknown normalization method: {method}")
                continue
        
        logger.info(f"Normalized {len(columns)} columns using {method} method")
        
        return normalized_data
    
    def calculate_returns(
        self, 
        data: pd.DataFrame, 
        price_column: str = "close",
        periods: List[int] = [1, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        计算收益率
        
        Args:
            data: 输入数据
            price_column: 价格列名
            periods: 计算周期列表
            
        Returns:
            pd.DataFrame: 包含收益率的数据
        """
        result_data = data.copy()
        
        if price_column not in data.columns:
            logger.error(f"Price column {price_column} not found")
            return result_data
        
        for period in periods:
            # 简单收益率
            result_data[f'return_{period}d'] = (
                result_data[price_column].pct_change(periods=period)
            )
            
            # 对数收益率
            result_data[f'log_return_{period}d'] = (
                np.log(result_data[price_column] / result_data[price_column].shift(period))
            )
        
        logger.info(f"Calculated returns for periods: {periods}")
        
        return result_data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 包含技术指标的数据
        """
        result_data = data.copy()
        
        if 'close' not in data.columns:
            logger.error("Close price column not found")
            return result_data
        
        # 移动平均线
        for window in [5, 10, 20, 50]:
            result_data[f'ma_{window}'] = result_data['close'].rolling(window=window).mean()
        
        # 相对强弱指数 (RSI)
        delta = result_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result_data['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        result_data['bb_middle'] = result_data['close'].rolling(window=20).mean()
        bb_std = result_data['close'].rolling(window=20).std()
        result_data['bb_upper'] = result_data['bb_middle'] + 2 * bb_std
        result_data['bb_lower'] = result_data['bb_middle'] - 2 * bb_std
        
        # MACD
        if 'close' in result_data.columns:
            ema_12 = result_data['close'].ewm(span=12).mean()
            ema_26 = result_data['close'].ewm(span=26).mean()
            result_data['macd'] = ema_12 - ema_26
            result_data['macd_signal'] = result_data['macd'].ewm(span=9).mean()
            result_data['macd_histogram'] = result_data['macd'] - result_data['macd_signal']
        
        # 成交量移动平均
        if 'volume' in result_data.columns:
            result_data['volume_ma_10'] = result_data['volume'].rolling(window=10).mean()
            result_data['volume_ratio'] = result_data['volume'] / result_data['volume_ma_10']
        
        logger.info("Added technical indicators")
        
        return result_data
    
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建时间特征
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 包含时间特征的数据
        """
        result_data = data.copy()
        
        if 'date' not in data.columns:
            logger.error("Date column not found")
            return result_data
        
        # 确保date列是datetime类型
        result_data['date'] = pd.to_datetime(result_data['date'])
        
        # 基本时间特征
        result_data['year'] = result_data['date'].dt.year
        result_data['month'] = result_data['date'].dt.month
        result_data['day'] = result_data['date'].dt.day
        result_data['weekday'] = result_data['date'].dt.weekday
        result_data['quarter'] = result_data['date'].dt.quarter
        
        # 是否为月末、季末、年末
        result_data['is_month_end'] = result_data['date'].dt.is_month_end
        result_data['is_quarter_end'] = result_data['date'].dt.is_quarter_end
        result_data['is_year_end'] = result_data['date'].dt.is_year_end
        
        # 周期性特征（sin/cos编码）
        result_data['month_sin'] = np.sin(2 * np.pi * result_data['month'] / 12)
        result_data['month_cos'] = np.cos(2 * np.pi * result_data['month'] / 12)
        result_data['weekday_sin'] = np.sin(2 * np.pi * result_data['weekday'] / 7)
        result_data['weekday_cos'] = np.cos(2 * np.pi * result_data['weekday'] / 7)
        
        logger.info("Created time features")
        
        return result_data
    
    def main():
        """主函数，用于演示 DataPreprocessor 的功能"""
        
        # 初始化数据加载器
        loader = DataLoader(enable_cache=True)
        
        print("数据加载器初始化完成")
        print(f"缓存目录: {loader.cache_dir}")
        
        # 演示加载沪深300成分股
        try:
            print("\n正在获取沪深300成分股...")
            hs300_stocks = loader.load_saved_data("tran.csv", "raw")
            hs300_stocks = loader.rename_columns(hs300_stocks, loader.column_mapping)
            if hs300_stocks is not None and not hs300_stocks.empty:
                print(f"成功获取 {len(hs300_stocks)} 只沪深300成分股")
                print(hs300_stocks.head())
            else:
                print("沪深300成分股数据为空或文件不存在")
        except Exception as e:
            print(f"获取沪深300成分股失败: {e}")

    if __name__ == "__main__":
        main()