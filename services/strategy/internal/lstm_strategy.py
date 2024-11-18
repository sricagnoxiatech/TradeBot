import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
import logging

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
        self.scalers = {}  # Separate scaler for each feature
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
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
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Initialize scalers for each feature if not exists
        for feature in self.config.features:
            if feature not in self.scalers:
                self.scalers[feature] = MinMaxScaler()

        # Scale features independently
        scaled_features = []
        for feature in self.config.features:
            values = df[feature].values.reshape(-1, 1)
            scaled = self.scalers[feature].fit_transform(values)
            scaled_features.append(scaled)
        
        # Combine scaled features
        scaled_data = np.hstack(scaled_features)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.config.sequence_length - self.config.prediction_steps + 1):
            X.append(scaled_data[i:(i + self.config.sequence_length)])
            y.append(scaled_data[i + self.config.sequence_length:i + self.config.sequence_length + self.config.prediction_steps, 0])
            
        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame) -> None:
        if len(df) < self.config.sequence_length + self.config.prediction_steps:
            self.logger.warning("Not enough data points for training")
            return

        try:
            X, y = self.prepare_data(df)
            
            self.model = self.create_model()
            self.model.fit(X, y, 
                         epochs=self.config.epochs, 
                         batch_size=self.config.batch_size,
                         validation_split=0.1,
                         verbose=0)
            
            self.is_trained = True
            self.logger.info("LSTM model training completed successfully")
        except Exception as e:
            self.logger.error(f"Error during LSTM training: {str(e)}")
            self.is_trained = False

    def predict(self, df: pd.DataFrame) -> float:
        if not self.is_trained or not self.model:
            self.logger.warning("Model needs to be trained first")
            return None
            
        if len(df) < self.config.sequence_length:
            self.logger.warning("Not enough data points for prediction")
            return None

        try:
            # Prepare the last sequence
            scaled_features = []
            for feature in self.config.features:
                values = df[feature].values.reshape(-1, 1)
                scaled = self.scalers[feature].transform(values)
                scaled_features.append(scaled)
            
            scaled_data = np.hstack(scaled_features)
            last_sequence = scaled_data[-self.config.sequence_length:]
            X = np.array([last_sequence])
            
            # Make prediction
            scaled_prediction = self.model.predict(X, verbose=0)[0][0]
            
            # Inverse transform only the prediction (first feature - close price)
            prediction = self.scalers[self.config.features[0]].inverse_transform(
                scaled_prediction.reshape(-1, 1)
            )[0][0]
            
            return prediction
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            return None

    def generate_signal(self, df: pd.DataFrame, current_price: float) -> str:
        if not self.config.enabled:
            return "NONE"

        try:
            predicted_price = self.predict(df)
            if predicted_price is None:
                return "NONE"
                
            threshold = 0.001  # 0.1% price movement threshold
            price_change_pct = (predicted_price - current_price) / current_price
            
            if price_change_pct > threshold:
                return "BUY"
            elif price_change_pct < -threshold:
                return "SELL"
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            
        return "NONE"

    def save_model(self, path: str) -> None:
        if self.model:
            try:
                self.model.save(path)
                # Save scalers
                for feature, scaler in self.scalers.items():
                    np.save(f"{path}_scaler_{feature}.npy", scaler.scale_)
                self.logger.info(f"Model saved successfully to {path}")
            except Exception as e:
                self.logger.error(f"Error saving model: {str(e)}")

    def load_model(self, path: str) -> None:
        try:
            if tf.io.gfile.exists(path):
                self.model = tf.keras.models.load_model(path)
                # Load scalers
                for feature in self.config.features:
                    scaler_path = f"{path}_scaler_{feature}.npy"
                    if tf.io.gfile.exists(scaler_path):
                        self.scalers[feature] = MinMaxScaler()
                        self.scalers[feature].scale_ = np.load(scaler_path)
                self.is_trained = True
                self.logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.is_trained = False