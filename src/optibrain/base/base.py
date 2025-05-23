# Copyright MewsLabs 2025
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


from typing import Optional, List, Dict

import keras
import pandas as pd
from palma.base.splitting_strategy import ValidationStrategy
import numpy as np
from revival import LiteModel
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
from optibrain.utils.engine import FlamlOptimizer
from optibrain.utils.project import Project


class SurrogateModeling:
    def __init__(self, estimator_list: List[str], problem: str, project_name="default"):
        self.__model = None
        self.__performance = None
        self.__config_estimator = None
        self.__best_time_train = None
        self.__best_config = None
        self.__metrics_for_best_config = None
        self.__supported_metrics = None
        self.__best_loss = None
        self.estimator_list = estimator_list
        self.problem = problem
        self.project_name = project_name
        self.prediction = None
        self.X_train = None
        self.y_train = None
        self.y_test = None
        self.X_test = None

    def find_best_model(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        learners: Optional[Dict[str, BaseEstimator]] = None,
        log_target=False,
    ):
        """Function that aims to select the best model, the user can also add learner to flaml
        Parameters
        X: pd.DataFrame
            X data for training
        y : pd.DataFrame
            y data for training
        learners : Dict
            Dictionary for new personalized learners
        log_target : bool
            True if you need to log-transforming the target
        """
        if self.problem == "regression":
            metric = "r2"
        else:
            metric = "roc_auc"

        if log_target:
            y = np.log(y)
            y = pd.DataFrame(y)

        engine_parameters = {
            "time_budget": 50,
            "metric": metric,
            "log_training_metric": True,
            "estimator_list": self.estimator_list,
        }
        splitting_strategy = ValidationStrategy(
            splitter=ShuffleSplit(
                n_splits=10,
                random_state=1,
            )
        )
        X, y = splitting_strategy(X, y)
        self.X_train = X.loc[splitting_strategy.train_index]
        self.X_test = X.loc[splitting_strategy.test_index]
        self.y_train = y.loc[splitting_strategy.train_index]
        self.y_test = y.loc[splitting_strategy.test_index]

        # Project creation
        project = Project(problem=self.problem, project_name=self.project_name)
        project.start(
            X,
            y,
            splitter=ShuffleSplit(n_splits=10, random_state=42),
        )
        # Create and start optimizer
        if learners is not None:
            optimizer = FlamlOptimizer(engine_parameters, learners)
        else:
            optimizer = FlamlOptimizer(engine_parameters, {})
        optimizer.start(project)
        # Get models performances
        self.__performance = optimizer.best_loss_estimator
        self.__config_estimator = optimizer.best_config_estimator
        self.__best_time_train = optimizer.best_time_estimator
        self.__best_config = optimizer.best_config
        self.__supported_metrics = optimizer.supported_metrics
        self.__metrics_for_best_config = optimizer.metrics_for_best_config
        self.__best_loss = optimizer.best_loss
        # Get the best model
        best_model = optimizer.best_model_
        self.__model = best_model
        self.X = X
        self.y = y

    @property
    def best_loss(self):
        return self.__best_loss

    @property
    def supported_metrics(self):
        return self.__supported_metrics

    @property
    def metrics_for_best_config(self):
        return self.__metrics_for_best_config

    @property
    def best_config(self):
        if isinstance(self.model, keras.Sequential):
            return self.model.summary()
        else:
            return self.__best_config

    @property
    def best_time_train_estimator(self):
        return self.__best_time_train

    @property
    def best_config_estimators(self):
        return self.__config_estimator

    @property
    def estimators_performances(self):
        """Function that returns the performances of trained estimator"""
        return self.__performance

    @property
    def model(self):
        """Function that returns the best model selected"""
        return self.__model

    def save(self, folder_name: str, file_name: str):
        """Function aims to save the model, the data and prediction in hdf5 file
        Parameters
        ----------
        folder_name:str
            The folder name where to save the hfd5 file
        file_name: str
            The file name where to save the model, the data and the prediction
        """
        srgt_model = LiteModel(self.X_train, self.y_train, self.model)
        srgt_model.score = self.best_loss
        srgt_model.dump(folder_name, file_name)

    def predict(self, X_new):
        """Function aims to predict targets from new values
        Parameters
        ----------
        X_new :
        Dataframe or array to predict
        """
        srgt_model = LiteModel(self.X_train, self.y_train, self.model)
        self.prediction = srgt_model.predict(X_new)
        return self.prediction
