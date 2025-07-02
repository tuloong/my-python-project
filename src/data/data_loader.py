"""
数据加载器模块

负责从各种数据源加载股票数据，支持Tushare、Yahoo Finance等多种数据源
提供数据缓存、验证和预处理功能
"""

import pandas as pd
import numpy as np
import yfinance as yf
import tushare as ts
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
import logging
from datetime import datetime, timedelta
import json
import time
import warnings
from functools import wraps

from ..config import DATA_CONFIG, DATA_PATHS, HS300_STOCK_COUNT

logger = logging.getLogger(__name__)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"函数 {func.__name__} 在 {max_retries} 次尝试后仍然失败: {e}")
                        raise
                    logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}，{delay}秒后重试...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class DataLoader:
    """股票数据加载器
    
    支持多种数据源：
    - Tushare (主要数据源)
    - Yahoo Finance (备用数据源)
    - 本地CSV文件
    
    功能特性：
    - 自动重试机制
    - 数据缓存
    - 数据验证
    - 列名标准化
    - 错误处理
    """
    
    def __init__(self, enable_cache: bool = True):
        """初始化数据加载器
        
        Args:
            tushare_token: Tushare API token
            enable_cache: 是否启用数据缓存
        """
        self.enable_cache = enable_cache
        self.cache_dir = DATA_CONFIG.get("cache_path", Path("./cache"))
        self.cache_dir.mkdir(exist_ok=True)
        
        # 列名映射字典
        self.column_mapping = {
            # Tushare列名映射
            'ts_code': 'stock_code',
            'trade_date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount',
            'pct_chg': 'change_rate',
            'change': 'change',
            'turnover_rate': 'turnover_rate',
            
            # Yahoo Finance列名映射
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
            
            # 本地文件中文列名映射
            '股票代码': 'stock_code',
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌额': 'change',
            '换手率': 'turnover_rate',
            '涨跌幅': 'change_rate'
        }
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        if not self.enable_cache:
            return None
            
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None
            
        try:
            # 检查缓存是否过期
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age > max_age_hours * 3600:
                logger.info(f"缓存文件 {cache_key} 已过期，将重新获取数据")
                return None
                
            data = pd.read_pickle(cache_path)
            logger.info(f"从缓存加载数据: {cache_key}")
            return data
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str):
        """保存数据到缓存"""
        if not self.enable_cache or data.empty:
            return
            
        try:
            cache_path = self._get_cache_path(cache_key)
            data.to_pickle(cache_path)
            logger.info(f"数据已缓存: {cache_key}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        if df.empty:
            return df
            
        # 重命名列
        df_renamed = df.rename(columns=self.column_mapping)
        
        # 确保日期列格式正确
        if 'date' in df_renamed.columns:
            df_renamed['date'] = pd.to_datetime(df_renamed['date'], errors='coerce')
        
        # 确保数值列类型正确
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 
                          'change_rate', 'change', 'turnover_rate', 'amplitude']
        
        for col in numeric_columns:
            if col in df_renamed.columns:
                df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')
        
        return df_renamed
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """验证数据质量"""
        if df.empty:
            logger.warning("数据为空")
            return False
        
        required_columns = ['stock_code', 'date', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"缺少必要列: {missing_columns}")
            return False
        
        # 检查数据是否有异常值
        if 'close' in df.columns:
            if df['close'].isna().sum() > len(df) * 0.5:
                logger.warning("收盘价数据缺失超过50%")
                return False
        
        return True
    
    def load_saved_data(self, filename: str, data_type: str = 'raw') -> Optional[pd.DataFrame]:
        """
        加载已保存的数据
        
        Args:
            filename: 文件名
            data_type: 数据类型 ('raw', 'processed', 'external')
            
        Returns:
            pd.DataFrame or None: 加载的数据
        """
        try:
            if data_type == 'processed':
                file_path = DATA_PATHS["processed_data"] / filename
            elif data_type == 'raw':
                file_path = DATA_PATHS["raw_data"] / filename
            elif data_type == 'external':
                file_path = DATA_PATHS["external_data"] / filename
            else:
                raise ValueError(f"不支持的数据类型: {data_type}")
            
            if not file_path.exists():
                logger.warning(f"文件不存在: {file_path}")
                return None
            
            # 根据文件扩展名选择加载方式
            if filename.endswith('.csv'):
                data = pd.read_csv(file_path, encoding='utf-8-sig')
            elif filename.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            elif filename.endswith('.pkl'):
                data = pd.read_pickle(file_path)
            else:
                # 默认尝试CSV
                data = pd.read_csv(file_path, encoding='utf-8-sig')
            
            logger.info(f"数据加载成功: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return None
    
    def get_latest_trading_date(self) -> Optional[str]:
        """获取最新交易日"""
        if self.pro is None:
            # 使用简单的日期计算
            today = datetime.now()
            # 如果是周末，回退到周五
            while today.weekday() > 4:  # 0=Monday, 6=Sunday
                today -= timedelta(days=1)
            return today.strftime('%Y-%m-%d')
        
        try:
            trade_cal = self.pro.trade_cal(
                exchange='',
                start_date=(datetime.now() - timedelta(days=10)).strftime('%Y%m%d'),
                end_date=datetime.now().strftime('%Y%m%d')
            )
            latest_date = trade_cal[trade_cal['is_open'] == 1]['cal_date'].max()
            return pd.to_datetime(latest_date).strftime('%Y-%m-%d')
        except Exception as e:
            logger.error(f"获取最新交易日失败: {e}")
            return None
    
    def clear_cache(self, cache_key: Optional[str] = None):
        """清理缓存"""
        try:
            if cache_key:
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    cache_path.unlink()
                    logger.info(f"已清理缓存: {cache_key}")
            else:
                # 清理所有缓存
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                logger.info("已清理所有缓存")
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")

    def rename_columns(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """重命名列名"""
        return df.rename(columns=mapping)

def main():
    """主函数，用于演示 DataLoader 的功能"""
    
    # 初始化数据加载器
    loader = DataLoader(enable_cache=True)
    
    print("数据加载器初始化完成")
    print(f"缓存目录: {loader.cache_dir}")
    
    # 演示加载沪深300成分股
    try:
        print("\n正在获取沪深300成分股...")
        hs300_stocks = loader.load_saved_data("train.csv", "raw")
        hs300_stocks = loader.rename_columns(hs300_stocks, loader.column_mapping)
        if hs300_stocks is not None and not hs300_stocks.empty:
            print(f"成功获取 {len(hs300_stocks)} 只沪深300成分股")
            print(hs300_stocks.head())
        else:
            print("沪深300成分股数据为空或文件不存在")
    except Exception as e:
        print(f"获取沪深300成分股失败: {e}")
    
    # 演示获取最新交易日
    try:
        print("\n正在获取最新交易日...")
        latest_date = loader.get_latest_trading_date()
        print(f"最新交易日: {latest_date}")
    except Exception as e:
        print(f"获取最新交易日失败: {e}")
    
    print("\n数据加载器演示完成")

if __name__ == "__main__":
    main()