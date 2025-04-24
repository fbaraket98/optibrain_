import tensorflow.keras as keras
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers


class FullNeuralNetwork(BaseEstimator):
    def __init__(
            self,
            n_jobs=None,
            hidden_units_l1=32,
            hidden_units_l2=32,
            act_l1="relu",
            act_l2="relu",
            bias_l1=True,
            bias_l2=True,
            optimizer="adam",
            epochs=50,
            batch_size=32,
            task="regression",
    ):
        super().__init__()
        self.n_jobs = n_jobs
        self.hidden_units_l1 = hidden_units_l1
        self.hidden_units_l2 = hidden_units_l2
        self.act_l1 = act_l1
        self.act_l2 = act_l2
        self.bias_l1 = bias_l1
        self.bias_l2 = bias_l2
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.task = task
        self.model = None
        self.scaler = StandardScaler()

    @classmethod
    def init(cls):
        return cls

    @classmethod
    def search_space(cls, data_size=None, task=None):
        return {
            "hidden_units_l1": {"domain": (16, 64), "init_value": 32},
            "hidden_units_l2": {"domain": (16, 64), "init_value": 32},
            "act_l1": {"domain": ["relu", "tanh"], "init_value": "relu"},
            "act_l2": {"domain": ["relu", "tanh"], "init_value": "relu"},
            "bias_l1": {"domain": [True, False], "init_value": True},
            "bias_l2": {"domain": [True, False], "init_value": True},
            "optimizer": {"domain": ["sgd", "rmsprop", "adam"], "init_value": "adam"},
            "epochs": {"domain": (10, 200), "init_value": 50},
            "batch_size": {"domain": (16, 128), "init_value": 32},
        }

    def _build_model(self, input_dim, output_dim):
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        model.add(
            layers.Dense(
                self.hidden_units_l1, activation=self.act_l1, use_bias=self.bias_l1
            )
        )
        model.add(
            layers.Dense(
                self.hidden_units_l2, activation=self.act_l2, use_bias=self.bias_l2
            )
        )
        model.add(layers.Dense(output_dim, activation="linear"))
        model.compile(
            optimizer=self.optimizer, loss="mean_squared_error", metrics=["mse"]
        )
        return model

    def fit(self, X, y, **kwargs):
        X_scaled = self.scaler.fit_transform(X)

        # 🔧 Convert y to a 2D ndarray
        if hasattr(y, "values"):
            y = y.values
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.model = self._build_model(X_scaled.shape[1], y.shape[1])
        self.model.fit(
            X_scaled,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_params(self, deep=True):
        return {
            "n_jobs": self.n_jobs,
            "hidden_units_l1": self.hidden_units_l1,
            "hidden_units_l2": self.hidden_units_l2,
            "act_l1": self.act_l1,
            "act_l2": self.act_l2,
            "bias_l1": self.bias_l1,
            "bias_l2": self.bias_l2,
            "optimizer": self.optimizer,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "task": self.task,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def cleanup(self):
        self.model = None

    @classmethod
    def cost_relative2lgbm(cls):
        return 10

    @classmethod
    def size(cls, config):
        return 64
