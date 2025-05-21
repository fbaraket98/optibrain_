# Copyright MewsLabs 2025
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import numpy as np
from sklearn.base import BaseEstimator
from smt.surrogate_models import KRG


class KRGModel(BaseEstimator):
    def __init__(self, n_jobs=None, task="regression", theta0=1.0):
        super().__init__()
        self.task = task
        self.theta0 = theta0
        self.model = None
        self.n_jobs = n_jobs

    @classmethod
    def search_space(cls, data_size=None, task=None):
        return {
            "theta0": {
                "domain": [1e-2, 1e1],
                "init_value": 1.0,
                "low_cost_init_value": 1.0,
            },
        }

    @classmethod
    def init(cls):
        return cls

    @classmethod
    def size(cls, config):
        return 1

    def fit(self, X, y, **kwargs):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        self.model = KRG(theta0=[self.theta0])
        self.model.set_training_values(X, y)
        self.model.train()
        return self

    def predict(self, X):
        X = np.array(X)
        return self.model.predict_values(X).flatten()

    def get_params(self, deep=True):
        return {
            "task": self.task,
            "theta0": self.theta0,
        }

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def cleanup(self):
        self.model = None

    @classmethod
    def cost_relative2lgbm(cls):
        return 10
