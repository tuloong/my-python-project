#!/usr/bin/env python3
"""
数据加载器使用示例

展示如何使用改进后的DataLoader类加载和处理股票数据
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.utils.logger import setup_logger

def main():
    """主函数 - 演示数据加载器功能"""
    
    # 设置日志
    logger = setup_logger("data_loader_example", level=logging.INFO)
    
    # 初始化数据加载器
    # 注意：如果有Tushare token，请传入参数
    data_loader = DataLoader(
        tushare_token=None,  # 在这里添加您的Tushare token
        enable_cache=True
    )
    
    logger.info("=== 数据加载器功能演示开始 ===")
    
    # 1. 加载沪深300成分股
    logger.info("\n1. 加载沪深300成分股")
    hs300_stocks = data_loader.load_stocks()
    if not hs300_stocks.empty:
        logger.info(f"成功加载 {len(hs300_stocks)} 只沪深300成分股")
        logger.info(f"数据列: {list(hs300_stocks.columns)}")
        logger.info(f"前5只股票:\n{hs300_stocks.head()}")
    else:
        logger.warning("未能加载沪深300成分股数据")
    
    # 2. 获取最新交易日
    logger.info("\n2. 获取最新交易日")
    latest_date = data_loader.get_latest_trading_date()
    if latest_date:
        logger.info(f"最新交易日: {latest_date}")
    
    # 3. 加载单只股票历史数据
    logger.info("\n3. 加载股票历史数据")
    stock_codes = ['000001.SZ', '000002.SZ', '600000.SH']  # 示例股票代码
    
    try:
        stock_data = data_loader.load_stock_data(
            stock_codes=stock_codes,
            start_date='2024-01-01',
            end_date='2024-01-31',
            fields=['stock_code', 'date', 'open', 'high', 'low', 'close', 'volume']
        )
        
        for code, data in stock_data.items():
            if not data.empty:
                logger.info(f"股票 {code}: {len(data)} 条记录")
                logger.info(f"数据范围: {data['date'].min()} 到 {data['date'].max()}")
            else:
                logger.warning(f"股票 {code}: 无数据")
    
    except Exception as e:
        logger.error(f"加载股票数据失败: {e}")
    
    # 4. 加载市场指数数据
    logger.info("\n4. 加载市场指数数据")
    try:
        market_data = data_loader.load_market_data(
            start_date='2024-01-01',
            end_date='2024-01-31',
            index_code='000300.SH'  # 沪深300指数
        )
        
        if not market_data.empty:
            logger.info(f"沪深300指数数据: {len(market_data)} 条记录")
            logger.info(f"收盘价范围: {market_data['close'].min():.2f} - {market_data['close'].max():.2f}")
        else:
            logger.warning("未能获取沪深300指数数据")
    
    except Exception as e:
        logger.error(f"加载指数数据失败: {e}")
    
    # 5. 数据保存和加载示例
    logger.info("\n5. 数据保存和加载示例")
    if not hs300_stocks.empty:
        try:
            # 保存数据
            data_loader.save_data(hs300_stocks, "hs300_example.csv", data_type="processed")
            
            # 重新加载数据
            loaded_data = data_loader.load_saved_data("hs300_example.csv", data_type="processed")
            if loaded_data is not None:
                logger.info(f"重新加载的数据: {len(loaded_data)} 条记录")
            
        except Exception as e:
            logger.error(f"数据保存/加载失败: {e}")
    
    # 6. 缓存管理示例
    logger.info("\n6. 缓存管理")
    try:
        # 显示缓存信息
        cache_files = list(data_loader.cache_dir.glob("*.pkl"))
        logger.info(f"当前缓存文件数量: {len(cache_files)}")
        
        # 清理特定缓存
        # data_loader.clear_cache("hs300_stocks")
        
        # 清理所有缓存（谨慎使用）
        # data_loader.clear_cache()
        
    except Exception as e:
        logger.error(f"缓存管理失败: {e}")
    
    logger.info("\n=== 数据加载器功能演示完成 ===")

def demo_data_analysis():
    """数据分析示例"""
    logger = logging.getLogger("data_loader_example")
    
    logger.info("\n=== 数据分析示例 ===")
    
    data_loader = DataLoader(enable_cache=True)
    
    # 加载一些示例股票数据进行分析
    stock_codes = ['000001.SZ', '000002.SZ']
    stock_data = data_loader.load_stock_data(
        stock_codes=stock_codes,
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    
    for code, data in stock_data.items():
        if not data.empty:
            logger.info(f"\n股票 {code} 分析:")
            logger.info(f"  交易天数: {len(data)}")
            logger.info(f"  平均收盘价: {data['close'].mean():.2f}")
            logger.info(f"  价格波动率: {data['close'].std():.2f}")
            logger.info(f"  最高价: {data['high'].max():.2f}")
            logger.info(f"  最低价: {data['low'].min():.2f}")
            
            # 计算简单的技术指标
            if len(data) >= 5:
                data['ma5'] = data['close'].rolling(window=5).mean()
                logger.info(f"  5日均线最新值: {data['ma5'].iloc[-1]:.2f}")

if __name__ == "__main__":
    try:
        main()
        demo_data_analysis()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc() 