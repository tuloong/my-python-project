#!/usr/bin/env python3
"""
快速训练脚本

一个简化的训练流程，快速验证模型效果
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.tree_models import XGBoostModel, LightGBMModel
from src.training.trainer import Trainer


def quick_train():
    """快速训练流程"""
    print("🚀 快速训练开始...")
    
    # 1. 加载数据
    loader = DataLoader()
    data = loader.load_saved_data("train.csv", "raw")
    if data is None:
        print("❌ 无法加载数据")
        return
    
    # 2. 简单预处理
    data = loader.rename_columns(data, loader.column_mapping)
    preprocessor = DataPreprocessor()
    data = preprocessor.clean_stock_data(data)
    data = preprocessor.handle_missing_values(data)
    
    # 3. 添加基础特征
    data = preprocessor.add_technical_indicators(data)
    data = preprocessor.calculate_returns(data, periods=[1])
    data = data.dropna()
    
    # 4. 准备数据
    feature_cols = [col for col in data.columns if not col.startswith('return_')]
    X = data[feature_cols]
    y = data['return_1d']
    
    print(f"📊 数据形状: X={X.shape}, y={y.shape}")
    
    # 5. 训练模型
    trainer = Trainer()
    models = {
        'XGBoost': XGBoostModel(),
        'LightGBM': LightGBMModel()
    }
    
    trained = trainer.train_multiple_models(models, X, y)
    
    # 6. 显示结果
    print("\n📈 训练结果:")
    for name, model in trained.items():
        metrics = model.evaluate(X, y)
        print(f"   {name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    # 7. 保存最佳模型
    trainer.save_models('outputs/models')
    print("✅ 训练完成，模型已保存")


if __name__ == "__main__":
    quick_train()