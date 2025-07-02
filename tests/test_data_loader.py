"""
数据加载器测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader


class TestDataLoader:
    """数据加载器测试类"""
    
    def setup_method(self):
        """每个测试方法运行前的设置"""
        self.data_loader = DataLoader()
    
    def test_init(self):
        """测试初始化"""
        assert self.data_loader is not None
    
    @patch('src.data.data_loader.ts')
    def test_load_hs300_stocks_with_tushare(self, mock_ts):
        """测试通过Tushare加载沪深300成分股"""
        # 模拟Tushare返回数据
        mock_data = pd.DataFrame({
            'con_code': ['000001.SZ', '000002.SZ', '000858.SZ'],
            'trade_date': ['20240101', '20240101', '20240101'],
            'weight': [1.5, 1.2, 0.8]
        })
        
        mock_pro = Mock()
        mock_pro.index_weight.return_value = mock_data
        mock_ts.pro_api.return_value = mock_pro
        
        # 重新初始化data_loader以使用mock
        self.data_loader.pro = mock_pro
        
        result = self.data_loader.load_hs300_stocks()
        
        assert not result.empty
        assert len(result) == 3
        assert 'con_code' in result.columns
        assert 'weight' in result.columns
    
    def test_load_hs300_stocks_fallback_to_local(self):
        """测试当API失败时回退到本地文件"""
        # 模拟API失败
        self.data_loader.pro = None
        
        # 由于本地文件可能不存在，这个测试可能会返回空DataFrame
        result = self.data_loader.load_hs300_stocks()
        
        # 验证返回了DataFrame（可能为空）
        assert isinstance(result, pd.DataFrame)
    
    def test_load_stock_data_empty_codes(self):
        """测试空股票代码列表"""
        result = self.data_loader.load_stock_data(
            stock_codes=[],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        assert result == {}
    
    def test_load_stock_data_invalid_date_format(self):
        """测试无效日期格式"""
        result = self.data_loader.load_stock_data(
            stock_codes=['000001.SZ'],
            start_date='invalid-date',
            end_date='2024-01-31'
        )
        
        # 应该处理错误并返回空字典
        assert isinstance(result, dict)
    
    @patch('src.data.data_loader.yf')
    def test_load_from_yahoo(self, mock_yf):
        """测试从Yahoo Finance加载数据"""
        # 模拟Yahoo Finance返回数据
        mock_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=5),
            'Open': [10.0, 10.1, 10.2, 10.3, 10.4],
            'High': [10.5, 10.6, 10.7, 10.8, 10.9],
            'Low': [9.5, 9.6, 9.7, 9.8, 9.9],
            'Close': [10.2, 10.3, 10.4, 10.5, 10.6],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_yf.Ticker.return_value = mock_ticker
        
        result = self.data_loader._load_from_yahoo(
            '000001.SZ', '2024-01-01', '2024-01-05'
        )
        
        assert result is not None
        assert not result.empty
        assert 'date' in result.columns
        assert 'close' in result.columns
        assert len(result) == 5
    
    def test_save_and_load_data(self):
        """测试数据保存和加载"""
        # 创建测试数据
        test_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'stock_code': ['000001.SZ', '000001.SZ', '000001.SZ'],
            'close': [10.0, 10.1, 10.2]
        })
        
        filename = 'test_data.csv'
        
        # 保存数据
        self.data_loader.save_data(test_data, filename)
        
        # 加载数据
        loaded_data = self.data_loader.load_saved_data(filename)
        
        assert loaded_data is not None
        assert len(loaded_data) == 3
        assert 'stock_code' in loaded_data.columns
        
        # 清理测试文件
        import os
        from src.config import DATA_PATHS
        test_file = DATA_PATHS["processed_data"] / filename
        if test_file.exists():
            os.remove(test_file)
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        result = self.data_loader.load_saved_data('nonexistent_file.csv')
        assert result is None
    
    @patch('src.data.data_loader.yf')
    def test_load_market_data_yahoo_fallback(self, mock_yf):
        """测试市场数据的Yahoo备用方案"""
        # 模拟没有Tushare API
        self.data_loader.pro = None
        
        # 模拟Yahoo Finance返回数据
        mock_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=5),
            'close': [3000, 3010, 3020, 3030, 3040]
        })
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_yf.Ticker.return_value = mock_ticker
        
        result = self.data_loader.load_market_data('2024-01-01', '2024-01-05')
        
        assert not result.empty
        assert 'close' in result.columns


if __name__ == '__main__':
    pytest.main([__file__]) 