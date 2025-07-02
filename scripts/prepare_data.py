#!/usr/bin/env python3
"""
数据准备脚本

用于下载、清洗和预处理股票数据
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.config import DATA_PATHS, LOG_CONFIG

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_CONFIG["file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    logger.info("Starting data preparation process...")
    
    try:
        # 初始化数据加载器和预处理器
        data_loader = DataLoader()
        preprocessor = DataPreprocessor()
        
        # 1. 获取沪深300成分股列表
        logger.info("Loading HS300 stock list...")
        hs300_stocks = data_loader.load_hs300_stocks()
        
        if hs300_stocks.empty:
            logger.error("Failed to load HS300 stock list")
            return
        
        # 获取股票代码列表
        if 'con_code' in hs300_stocks.columns:
            stock_codes = hs300_stocks['con_code'].unique().tolist()
        else:
            logger.error("Stock code column not found in HS300 data")
            return
        
        logger.info(f"Found {len(stock_codes)} HS300 stocks")
        
        # 2. 设置数据获取的日期范围
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')  # 2年历史数据
        
        logger.info(f"Loading stock data from {start_date} to {end_date}")
        
        # 3. 批量下载股票数据
        batch_size = 50  # 每次处理50只股票
        all_stock_data = []
        
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: stocks {i+1}-{min(i+batch_size, len(stock_codes))}")
            
            try:
                # 加载数据
                stock_data = data_loader.load_stock_data(
                    stock_codes=batch_codes,
                    start_date=start_date,
                    end_date=end_date,
                    source="tushare"  # 优先使用tushare
                )
                
                # 处理每只股票的数据
                for stock_code, data in stock_data.items():
                    if data is not None and not data.empty:
                        # 数据清洗
                        cleaned_data = preprocessor.clean_stock_data(data)
                        
                        # 处理缺失值
                        processed_data = preprocessor.handle_missing_values(
                            cleaned_data, 
                            method="forward_fill"
                        )
                        
                        # 计算收益率
                        data_with_returns = preprocessor.calculate_returns(
                            processed_data,
                            price_column="close",
                            periods=[1, 5, 10, 20]
                        )
                        
                        # 添加技术指标
                        data_with_indicators = preprocessor.add_technical_indicators(
                            data_with_returns
                        )
                        
                        # 添加时间特征
                        final_data = preprocessor.create_time_features(
                            data_with_indicators
                        )
                        
                        # 添加股票代码列
                        final_data['stock_code'] = stock_code
                        
                        all_stock_data.append(final_data)
                        
                        logger.debug(f"Processed {stock_code}: {len(final_data)} records")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        # 4. 合并所有数据
        if all_stock_data:
            import pandas as pd
            combined_data = pd.concat(all_stock_data, ignore_index=True)
            logger.info(f"Combined data shape: {combined_data.shape}")
            
            # 5. 保存处理后的数据
            output_path = DATA_PATHS["processed_data"] / "combined_stock_data.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined_data.to_csv(output_path, index=False)
            logger.info(f"Saved combined data to {output_path}")
            
            # 6. 保存数据摘要
            summary = {
                'total_records': len(combined_data),
                'unique_stocks': combined_data['stock_code'].nunique(),
                'date_range': f"{combined_data['date'].min()} to {combined_data['date'].max()}",
                'features': list(combined_data.columns),
                'feature_count': len(combined_data.columns)
            }
            
            import json
            summary_path = DATA_PATHS["processed_data"] / "data_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved data summary to {summary_path}")
            logger.info("Data preparation completed successfully!")
            
        else:
            logger.error("No data was successfully processed")
            
    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
        raise


if __name__ == "__main__":
    main() 