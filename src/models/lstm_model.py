"""
LSTM模型实现

用于时间序列预测的LSTM神经网络模型
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from .base_model import BaseModel


class LSTMModel(BaseModel):
    """LSTM时间序列预测模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        
        # LSTM配置参数
        self.seq_length = config.get('seq_length', 60) if config else 60
        self.n_features = config.get('n_features', 1) if config else 1
        self.lstm_units = config.get('lstm_units', [50, 50]) if config else [50, 50]
        self.dropout_rate = config.get('dropout_rate', 0.2) if config else 0.2
        self.learning_rate = config.get('learning_rate', 0.001) if config else 0.001
        self.batch_size = config.get('batch_size', 32) if config else 32
        self.epochs = config.get('epochs', 100) if config else 100
        self.patience = config.get('patience', 10) if config else 10
        
        # 数据预处理
        self.scaler = MinMaxScaler()
        self.X_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        
        # 模型相关
        self.model = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """构建LSTM模型"""
        model = keras.Sequential()
        
        # 输入层
        model.add(layers.Input(shape=(self.seq_length, self.n_features)))
        
        # LSTM层
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(layers.LSTM(
                units, 
                return_sequences=return_sequences,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'
            ))
            model.add(layers.Dropout(self.dropout_rate))
        
        # 全连接层
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        
        # 编译模型
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, data: pd.DataFrame, target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """准备时间序列数据"""
        # 提取特征和目标
        feature_cols = [col for col in data.columns if col != target_col]
        X_data = data[feature_cols].values
        y_data = data[target_col].values.reshape(-1, 1)
        
        # 标准化数据
        X_scaled = self.X_scaler.fit_transform(X_data)
        y_scaled = self.y_scaler.fit_transform(y_data)
        
        # 创建序列
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X_scaled) - self.seq_length):
            X_sequences.append(X_scaled[i:(i + self.seq_length)])
            y_sequences.append(y_scaled[i + self.seq_length])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None, 
            validation_split: float = 0.2, verbose: int = 0, **kwargs) -> 'LSTMModel':
        """训练LSTM模型"""
        if y is None:
            # 假设X已经包含了目标列
            sequences, targets = self.prepare_sequences(X)
        else:
            # 合并X和y创建数据框
            data = X.copy()
            data['target'] = y
            sequences, targets = self.prepare_sequences(data, 'target')
        
        # 更新特征维度
        self.n_features = sequences.shape[2]
        
        # 构建模型
        if self.model is None:
            self.build_model()
        
        # 设置回调函数
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # 训练模型
        self.history = self.model.fit(
            sequences, targets,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            **kwargs
        )
        
        self.is_fitted = True
        self.feature_names = list(X.columns) if y is None else list(X.columns) + ['target']
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 准备预测数据
        if hasattr(self, 'feature_names') and len(X.columns) == len(self.feature_names):
            # 使用训练时的特征顺序
            X_data = X[self.feature_names[:-1]].values  # 排除目标列
        else:
            # 假设最后一列是目标列
            X_data = X.values
        
        # 标准化输入数据
        X_scaled = self.X_scaler.transform(X_data)
        
        # 创建序列
        sequences = []
        for i in range(len(X_scaled) - self.seq_length + 1):
            sequences.append(X_scaled[i:(i + self.seq_length)])
        
        if len(sequences) == 0:
            # 如果数据不足一个序列长度，重复最后一个值
            last_sequence = np.tile(X_scaled[-1:], (self.seq_length, 1))
            sequences = [last_sequence]
        
        sequences = np.array(sequences)
        
        # 预测
        predictions_scaled = self.model.predict(sequences)
        
        # 反标准化
        predictions = self.y_scaler.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def predict_with_uncertainty(self, X: pd.DataFrame, n_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """预测并返回不确定性（使用MC Dropout）"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 启用训练模式以使用Dropout
        predictions = []
        for _ in range(n_iterations):
            pred = self.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        
        return mean_pred, uncertainty
    
    def save_model(self, filepath: str) -> None:
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存Keras模型
        model_path = filepath.replace('.pkl', '_keras.h5')
        self.model.save(model_path)
        
        # 保存其他状态
        model_data = {
            'model_type': 'LSTMModel',
            'config': self.config,
            'seq_length': self.seq_length,
            'n_features': self.n_features,
            'X_scaler': self.X_scaler,
            'y_scaler': self.y_scaler,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics
        }
        
        import joblib
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> 'LSTMModel':
        """加载模型"""
        import joblib
        import os
        
        # 加载状态
        model_data = joblib.load(filepath)
        
        # 恢复属性
        self.config = model_data['config']
        self.seq_length = model_data['seq_length']
        self.n_features = model_data['n_features']
        self.X_scaler = model_data['X_scaler']
        self.y_scaler = model_data['y_scaler']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        
        # 加载Keras模型
        model_path = filepath.replace('.pkl', '_keras.h5')
        self.model = keras.models.load_model(model_path)
        
        self.is_fitted = True
        return self
    
    def get_training_history(self) -> Dict[str, list]:
        """获取训练历史"""
        if self.history is None:
            return {}
        
        return {
            'loss': self.history.history.get('loss', []),
            'val_loss': self.history.history.get('val_loss', []),
            'mae': self.history.history.get('mae', []),
            'val_mae': self.history.history.get('val_mae', [])
        }


class GRUModel(BaseModel):
    """GRU时间序列预测模型"""
    
    def __init__(self, config: Dict[str, Any] = None, random_seed: int = 42):
        super().__init__(config, random_seed)
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        
        # GRU配置参数
        self.seq_length = config.get('seq_length', 60) if config else 60
        self.n_features = config.get('n_features', 1) if config else 1
        self.gru_units = config.get('gru_units', [50, 50]) if config else [50, 50]
        self.dropout_rate = config.get('dropout_rate', 0.2) if config else 0.2
        self.learning_rate = config.get('learning_rate', 0.001) if config else 0.001
        self.batch_size = config.get('batch_size', 32) if config else 32
        self.epochs = config.get('epochs', 100) if config else 100
        self.patience = config.get('patience', 10) if config else 10
        
        # 数据预处理
        self.scaler = MinMaxScaler()
        self.X_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        
        # 模型相关
        self.model = None
        self.history = None
    
    def build_model(self) -> keras.Model:
        """构建GRU模型"""
        model = keras.Sequential()
        
        # 输入层
        model.add(layers.Input(shape=(self.seq_length, self.n_features)))
        
        # GRU层
        for i, units in enumerate(self.gru_units):
            return_sequences = i < len(self.gru_units) - 1
            model.add(layers.GRU(
                units, 
                return_sequences=return_sequences,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'
            ))
            model.add(layers.Dropout(self.dropout_rate))
        
        # 全连接层
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        
        # 编译模型
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, data: pd.DataFrame, target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """准备时间序列数据"""
        # 提取特征和目标
        feature_cols = [col for col in data.columns if col != target_col]
        X_data = data[feature_cols].values
        y_data = data[target_col].values.reshape(-1, 1)
        
        # 标准化数据
        X_scaled = self.X_scaler.fit_transform(X_data)
        y_scaled = self.y_scaler.fit_transform(y_data)
        
        # 创建序列
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X_scaled) - self.seq_length):
            X_sequences.append(X_scaled[i:(i + self.seq_length)])
            y_sequences.append(y_scaled[i + self.seq_length])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None, 
            validation_split: float = 0.2, verbose: int = 0, **kwargs) -> 'GRUModel':
        """训练GRU模型"""
        if y is None:
            # 假设X已经包含了目标列
            sequences, targets = self.prepare_sequences(X)
        else:
            # 合并X和y创建数据框
            data = X.copy()
            data['target'] = y
            sequences, targets = self.prepare_sequences(data, 'target')
        
        # 更新特征维度
        self.n_features = sequences.shape[2]
        
        # 构建模型
        if self.model is None:
            self.build_model()
        
        # 设置回调函数
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # 训练模型
        self.history = self.model.fit(
            sequences, targets,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            **kwargs
        )
        
        self.is_fitted = True
        self.feature_names = list(X.columns) if y is None else list(X.columns) + ['target']
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 准备预测数据
        if hasattr(self, 'feature_names') and len(X.columns) == len(self.feature_names):
            # 使用训练时的特征顺序
            X_data = X[self.feature_names[:-1]].values  # 排除目标列
        else:
            # 假设最后一列是目标列
            X_data = X.values
        
        # 标准化输入数据
        X_scaled = self.X_scaler.transform(X_data)
        
        # 创建序列
        sequences = []
        for i in range(len(X_scaled) - self.seq_length + 1):
            sequences.append(X_scaled[i:(i + self.seq_length)])
        
        if len(sequences) == 0:
            # 如果数据不足一个序列长度，重复最后一个值
            last_sequence = np.tile(X_scaled[-1:], (self.seq_length, 1))
            sequences = [last_sequence]
        
        sequences = np.array(sequences)
        
        # 预测
        predictions_scaled = self.model.predict(sequences)
        
        # 反标准化
        predictions = self.y_scaler.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def save_model(self, filepath: str) -> None:
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存Keras模型
        model_path = filepath.replace('.pkl', '_keras.h5')
        self.model.save(model_path)
        
        # 保存其他状态
        model_data = {
            'model_type': 'GRUModel',
            'config': self.config,
            'seq_length': self.seq_length,
            'n_features': self.n_features,
            'X_scaler': self.X_scaler,
            'y_scaler': self.y_scaler,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics
        }
        
        import joblib
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> 'GRUModel':
        """加载模型"""
        import joblib
        import os
        
        # 加载状态
        model_data = joblib.load(filepath)
        
        # 恢复属性
        self.config = model_data['config']
        self.seq_length = model_data['seq_length']
        self.n_features = model_data['n_features']
        self.X_scaler = model_data['X_scaler']
        self.y_scaler = model_data['y_scaler']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        
        # 加载Keras模型
        model_path = filepath.replace('.pkl', '_keras.h5')
        self.model = keras.models.load_model(model_path)
        
        self.is_fitted = True
        return self