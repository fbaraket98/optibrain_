from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np


class FullNeuralNetwork(BaseEstimator):
    def __init__(
        self, n_jobs=None, hidden_units=64, epochs=10, batch_size=32, task="regression"
    ):
        super().__init__()
        self.task = task
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = None
        self.n_jobs = n_jobs

    @classmethod
    def init(cls):
        return cls

    @classmethod
    def search_space(cls, data_size=None, task=None):
        return {
            "epochs": {
                "domain": (10, 100),
                "init_value": 10,
                "low_cost_init_value": 10,
            },
            "batch_size": {"domain": (16, 128), "init_value": 32},
            "hidden_units": {"domain": (16, 256), "init_value": 64},
        }

    @classmethod
    def cost_relative2lgbm(cls):
        return 10

    def _to_int(self, val):
        return int(val[1]) if isinstance(val, tuple) else int(val)

    def _build_model(self, input_dim):
        hidden_units = self._to_int(self.hidden_units)
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        model.add(layers.Dense(hidden_units, activation="relu"))
        model.add(
            layers.Dense(
                1, activation="sigmoid" if self.task == "classification" else "linear"
            )
        )
        model.add(layers.Dense(int(2)))
        return model

    def fit(self, X_train, y_train, **kwargs):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model = self._build_model(input_dim=X_scaled.shape[1])
        self.model.compile(
            optimizer="adam",
            loss=(
                "binary_crossentropy"
                if self.task == "classification"
                else "mean_squared_error"
            ),
            metrics=(
                ["accuracy"]
                if self.task == "classification"
                else ["mean_squared_error"]
            ),
        )
        self.model.fit(
            X_scaled,
            y_train,
            epochs=self._to_int(self.epochs),
            batch_size=self._to_int(self.batch_size),
            verbose=0,
        )

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return (
            (preds > 0.5).astype(int).flatten()
            if self.task == "classification"
            else preds.flatten()
        )

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return np.hstack([1 - preds, preds])

    @classmethod
    def size(cls, config):
        return 64

    def cleanup(self):
        self.model = None

    def get_params(self, deep=True):
        return {
            "hidden_units": self.hidden_units,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "task": self.task,
        }

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self
