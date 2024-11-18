import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass

@dataclass
class LSTMConfig:
    enabled: bool = False
    sequence_length: int = 60
    prediction_steps: int = 1
    epochs: int = 50
    batch_size: int = 32
    units: int = 50
    features: List[str] = None

    def __post_init__(self):
        if self.features is None:
            self.features = ['close', 'volume', 'rsi', 'macd']

class LSTMStrategy:
    def __init__(self, config: LSTMConfig):
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def create_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=self.config.units, 
                               return_sequences=True,
                               input_shape=(self.config.sequence_length, len(self.config.features))),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=self.config.units),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.config.prediction_steps)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Extract features
        data = df[self.config.features].values
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.config.sequence_length - self.config.prediction_steps + 1):
            X.append(scaled_data[i:(i + self.config.sequence_length)])
            y.append(scaled_data[i + self.config.sequence_length:i + self.config.sequence_length + self.config.prediction_steps, 0])
            
        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame) -> None:
        if len(df) < self.config.sequence_length + self.config.prediction_steps:
            raise ValueError("Not enough data points for training")

        X, y = self.prepare_data(df)
        
        self.model = self.create_model()
        self.model.fit(X, y, 
                      epochs=self.config.epochs, 
                      batch_size=self.config.batch_size,
                      verbose=0)
        
        self.is_trained = True

    def predict(self, df: pd.DataFrame) -> float:
        if not self.is_trained:
            raise ValueError("Model needs to be trained first")
            
        if len(df) < self.config.sequence_length:
            raise ValueError("Not enough data points for prediction")

        # Prepare the last sequence
        data = df[self.config.features].values
        scaled_data = self.scaler.transform(data)
        last_sequence = scaled_data[-self.config.sequence_length:]
        X = np.array([last_sequence])
        
        # Make prediction
        scaled_prediction = self.model.predict(X, verbose=0)[0][0]
        
        # Inverse transform the prediction
        dummy_array = np.zeros((1, len(self.config.features)))
        dummy_array[0, 0] = scaled_prediction
        prediction = self.scaler.inverse_transform(dummy_array)[0, 0]
        
        return prediction

    def generate_signal(self, df: pd.DataFrame, current_price: float) -> str:
        if not self.config.enabled:
            return "NONE"

        try:
            predicted_price = self.predict(df)
            threshold = 0.001  # 0.1% price movement threshold
            
            price_change_pct = (predicted_price - current_price) / current_price
            
            if price_change_pct > threshold:
                return "BUY"
            elif price_change_pct < -threshold:
                return "SELL"
            
        except (ValueError, Exception) as e:
            print(f"Error generating signal: {str(e)}")
            
        return "NONE"

    def save_model(self, path: str) -> None:
        if self.model:
            self.model.save(path)
            np.save(f"{path}_scaler.npy", self.scaler.scale_)

    def load_model(self, path: str) -> None:
        if tf.io.gfile.exists(path):
            self.model = tf.keras.models.load_model(path)
            self.scaler.scale_ = np.load(f"{path}_scaler.npy")
            self.is_trained = True