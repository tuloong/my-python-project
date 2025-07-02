#!/usr/bin/env python3
"""
预测脚本

用于生成股价预测结果，输出涨幅最大和最小的股票
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import joblib

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATHS, MODEL_CONFIG, PREDICTION_CONFIG, LOG_CONFIG
from src.data.data_loader import DataLoader
from src.features.feature_engineer import FeatureEngineer
from src.prediction.predictor import StockPredictor

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


def load_latest_model():
    """加载最新的训练模型"""
    try:
        model_dir = Path(MODEL_CONFIG["save_path"])
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # 查找最新的模型文件
        model_files = list(model_dir.glob("*.joblib"))
        
        if not model_files:
            raise FileNotFoundError("No model files found")
        
        # 按修改时间排序，获取最新的
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        
        # 加载模型
        model = joblib.load(latest_model_file)
        
        logger.info(f"Loaded model from: {latest_model_file}")
        
        # 查找对应的特征文件
        timestamp = latest_model_file.stem.split('_')[-1]
        features_file = model_dir / f"selected_features_{timestamp}.json"
        
        if features_file.exists():
            with open(features_file, 'r', encoding='utf-8') as f:
                selected_features = json.load(f)
        else:
            logger.warning("Selected features file not found, using default")
            selected_features = None
        
        return model, selected_features
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def get_latest_data():
    """获取最新的股票数据"""
    try:
        data_loader = DataLoader()
        
        # 获取沪深300成分股列表
        hs300_stocks = data_loader.load_hs300_stocks()
        
        if hs300_stocks.empty:
            raise ValueError("Failed to load HS300 stock list")
        
        # 获取股票代码
        if 'con_code' in hs300_stocks.columns:
            stock_codes = hs300_stocks['con_code'].unique().tolist()
        else:
            logger.error("Stock code column not found")
            return None
        
        # 设置日期范围（获取最近30天的数据用于特征计算）
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        logger.info(f"Loading latest data from {start_date} to {end_date}")
        
        # 加载最新数据
        stock_data = data_loader.load_stock_data(
            stock_codes=stock_codes[:50],  # 限制数量以加快速度
            start_date=start_date,
            end_date=end_date,
            source="tushare"
        )
        
        return stock_data
        
    except Exception as e:
        logger.error(f"Error getting latest data: {e}")
        raise


def prepare_prediction_features(stock_data):
    """准备预测用的特征"""
    try:
        feature_engineer = FeatureEngineer()
        prediction_features = []
        
        for stock_code, data in stock_data.items():
            if data is None or data.empty:
                continue
            
            try:
                # 数据预处理
                from src.data.data_preprocessor import DataPreprocessor
                preprocessor = DataPreprocessor()
                
                # 清洗数据
                cleaned_data = preprocessor.clean_stock_data(data)
                
                # 处理缺失值
                processed_data = preprocessor.handle_missing_values(cleaned_data)
                
                # 计算收益率
                data_with_returns = preprocessor.calculate_returns(processed_data)
                
                # 添加技术指标
                data_with_indicators = preprocessor.add_technical_indicators(data_with_returns)
                
                # 添加时间特征
                final_data = preprocessor.create_time_features(data_with_indicators)
                
                # 生成预测特征
                features = feature_engineer.create_features(final_data)
                
                if not features.empty:
                    # 添加股票代码
                    features['stock_code'] = stock_code
                    
                    # 只保留最新的一行数据用于预测
                    latest_features = features.iloc[-1:].copy()
                    prediction_features.append(latest_features)
                
            except Exception as e:
                logger.warning(f"Error processing {stock_code}: {e}")
                continue
        
        if prediction_features:
            combined_features = pd.concat(prediction_features, ignore_index=True)
            logger.info(f"Prepared prediction features for {len(prediction_features)} stocks")
            return combined_features
        else:
            logger.error("No valid prediction features generated")
            return None
            
    except Exception as e:
        logger.error(f"Error preparing prediction features: {e}")
        raise


def make_predictions(model, features, selected_features):
    """生成预测"""
    try:
        if features is None or features.empty:
            logger.error("No features provided for prediction")
            return None
        
        # 准备特征矩阵
        if selected_features:
            # 确保所有选定特征都存在
            available_features = [f for f in selected_features if f in features.columns]
            if len(available_features) != len(selected_features):
                logger.warning(f"Missing {len(selected_features) - len(available_features)} features")
            
            X = features[available_features]
        else:
            # 如果没有特征选择信息，使用所有数值特征
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            exclude_cols = ['stock_code', 'date'] if 'date' in features.columns else ['stock_code']
            X = features[[col for col in numeric_columns if col not in exclude_cols]]
        
        # 处理缺失值
        X = X.fillna(0)
        
        # 生成预测
        predictions = model.predict(X)
        
        # 创建结果DataFrame
        result = pd.DataFrame({
            'stock_code': features['stock_code'],
            'predicted_return': predictions
        })
        
        logger.info(f"Generated predictions for {len(result)} stocks")
        
        return result
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise


def select_top_stocks(predictions):
    """选择涨跌幅最大的股票"""
    try:
        if predictions is None or predictions.empty:
            logger.error("No predictions provided")
            return None, None
        
        # 按预测收益率排序
        sorted_predictions = predictions.sort_values('predicted_return', ascending=False)
        
        # 选择涨幅最大的10只股票
        top_gainers = sorted_predictions.head(PREDICTION_CONFIG["top_stocks_count"])
        
        # 选择跌幅最大的10只股票（涨幅最小）
        top_losers = sorted_predictions.tail(PREDICTION_CONFIG["top_stocks_count"])
        
        logger.info("Selected top gainers and losers")
        logger.info(f"Top gainer predicted return: {top_gainers.iloc[0]['predicted_return']:.4f}")
        logger.info(f"Top loser predicted return: {top_losers.iloc[-1]['predicted_return']:.4f}")
        
        return top_gainers, top_losers
        
    except Exception as e:
        logger.error(f"Error selecting top stocks: {e}")
        raise


def save_predictions(predictions, top_gainers, top_losers):
    """保存预测结果"""
    try:
        # 创建输出目录
        output_dir = Path(MODEL_CONFIG["prediction_output_path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整预测结果
        full_predictions_path = output_dir / f"full_predictions_{timestamp}.csv"
        predictions.to_csv(full_predictions_path, index=False)
        
        # 保存涨幅最大股票
        top_gainers_path = output_dir / f"top_gainers_{timestamp}.csv"
        top_gainers.to_csv(top_gainers_path, index=False)
        
        # 保存跌幅最大股票
        top_losers_path = output_dir / f"top_losers_{timestamp}.csv"
        top_losers.to_csv(top_losers_path, index=False)
        
        # 保存竞赛格式结果
        competition_result = {
            'timestamp': timestamp,
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'top_gainers': top_gainers['stock_code'].tolist(),
            'top_losers': top_losers['stock_code'].tolist(),
            'top_gainers_returns': top_gainers['predicted_return'].tolist(),
            'top_losers_returns': top_losers['predicted_return'].tolist()
        }
        
        competition_path = output_dir / f"competition_submission_{timestamp}.json"
        with open(competition_path, 'w', encoding='utf-8') as f:
            json.dump(competition_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved prediction results with timestamp: {timestamp}")
        logger.info(f"Competition submission file: {competition_path}")
        
        return timestamp
        
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        raise


def main():
    """主函数"""
    logger.info("Starting prediction process...")
    
    try:
        # 1. 加载训练好的模型
        model, selected_features = load_latest_model()
        
        # 2. 获取最新数据
        stock_data = get_latest_data()
        
        if not stock_data:
            logger.error("Failed to get latest data")
            return
        
        # 3. 准备预测特征
        prediction_features = prepare_prediction_features(stock_data)
        
        if prediction_features is None:
            logger.error("Failed to prepare prediction features")
            return
        
        # 4. 生成预测
        predictions = make_predictions(model, prediction_features, selected_features)
        
        if predictions is None:
            logger.error("Failed to generate predictions")
            return
        
        # 5. 选择涨跌幅最大的股票
        top_gainers, top_losers = select_top_stocks(predictions)
        
        if top_gainers is None or top_losers is None:
            logger.error("Failed to select top stocks")
            return
        
        # 6. 保存预测结果
        timestamp = save_predictions(predictions, top_gainers, top_losers)
        
        # 7. 输出结果摘要
        logger.info("=== 预测结果摘要 ===")
        logger.info(f"预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"预测股票数量: {len(predictions)}")
        
        logger.info("\n涨幅最大的10只股票:")
        for i, (_, row) in enumerate(top_gainers.iterrows(), 1):
            logger.info(f"{i}. {row['stock_code']}: {row['predicted_return']:.4f}")
        
        logger.info("\n跌幅最大的10只股票:")
        for i, (_, row) in enumerate(top_losers.iterrows(), 1):
            logger.info(f"{i}. {row['stock_code']}: {row['predicted_return']:.4f}")
        
        logger.info("Prediction process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in prediction process: {e}")
        raise


if __name__ == "__main__":
    main() 