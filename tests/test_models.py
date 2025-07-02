"""
模型测试

测试各种机器学习模型的功能
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

from src.models.tree_models import XGBoostModel, LightGBMModel, CatBoostModel
from src.models.linear_models import LinearRegressionModel, RidgeModel
from src.models.ensemble_models import VotingEnsemble, StackingEnsemble


class TestModels(unittest.TestCase):
    """模型测试类"""
    
    def setUp(self):
        """设置测试数据"""
        # 生成模拟数据
        X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
        
        self.X_train = pd.DataFrame(X[:800], columns=[f'feature_{i}' for i in range(20)])
        self.y_train = pd.Series(y[:800])
        self.X_test = pd.DataFrame(X[800:], columns=[f'feature_{i}' for i in range(20)])
        self.y_test = pd.Series(y[800:])
    
    def test_xgboost_model(self):
        """测试XGBoost模型"""
        model = XGBoostModel(random_seed=42)
        model.build_model()
        
        # 训练模型
        model.train(self.X_train, self.y_train)
        
        # 预测
        predictions = model.predict(self.X_test)
        
        # 验证预测结果
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertIsInstance(predictions, np.ndarray)
        
        # 检查特征重要性
        importance = model.get_feature_importance()
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), 20)
    
    def test_lightgbm_model(self):
        """测试LightGBM模型"""
        model = LightGBMModel(random_seed=42)
        model.build_model()
        
        # 训练模型
        model.train(self.X_train, self.y_train)
        
        # 预测
        predictions = model.predict(self.X_test)
        
        # 验证预测结果
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertIsInstance(predictions, np.ndarray)
    
    def test_catboost_model(self):
        """测试CatBoost模型"""
        model = CatBoostModel(random_seed=42)
        model.build_model()
        
        # 训练模型（CatBoost可能输出训练信息，我们忽略它）
        model.train(self.X_train, self.y_train)
        
        # 预测
        predictions = model.predict(self.X_test)
        
        # 验证预测结果
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertIsInstance(predictions, np.ndarray)
    
    def test_linear_regression_model(self):
        """测试线性回归模型"""
        model = LinearRegressionModel(random_seed=42)
        model.build_model()
        
        # 训练模型
        model.train(self.X_train, self.y_train)
        
        # 预测
        predictions = model.predict(self.X_test)
        
        # 验证预测结果
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertIsInstance(predictions, np.ndarray)
    
    def test_ridge_model(self):
        """测试Ridge回归模型"""
        model = RidgeModel(config={'alpha': 1.0}, random_seed=42)
        model.build_model()
        
        # 训练模型
        model.train(self.X_train, self.y_train)
        
        # 预测
        predictions = model.predict(self.X_test)
        
        # 验证预测结果
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertIsInstance(predictions, np.ndarray)
    
    def test_voting_ensemble(self):
        """测试投票集成模型"""
        # 创建基础模型
        models = {
            'xgb': XGBoostModel(random_seed=42),
            'lgb': LightGBMModel(random_seed=42),
            'lr': LinearRegressionModel(random_seed=42)
        }
        
        # 构建所有模型
        for model in models.values():
            model.build_model()
        
        # 创建集成模型
        ensemble = VotingEnsemble(models=models, random_seed=42)
        ensemble.build_model()
        
        # 训练集成模型
        ensemble.train(self.X_train, self.y_train)
        
        # 预测
        predictions = ensemble.predict(self.X_test)
        
        # 验证预测结果
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertIsInstance(predictions, np.ndarray)
    
    def test_model_persistence(self):
        """测试模型保存和加载"""
        import tempfile
        import os
        
        # 创建模型
        model = XGBoostModel(random_seed=42)
        model.build_model()
        model.train(self.X_train, self.y_train)
        
        # 保存模型
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.pkl')
            model.save_model(model_path)
            
            # 验证文件存在
            self.assertTrue(os.path.exists(model_path))
            
            # 加载模型
            new_model = XGBoostModel(random_seed=42)
            new_model.load_model(model_path)
            
            # 验证预测结果一致
            original_pred = model.predict(self.X_test)
            loaded_pred = new_model.predict(self.X_test)
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)


if __name__ == '__main__':
    unittest.main() 