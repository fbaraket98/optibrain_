from typing import Optional, List, Dict

import keras
import pandas as pd
from palma.base.splitting_strategy import ValidationStrategy
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

    def get_best_model(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        learners: Optional[Dict[str, BaseEstimator]] = None,
    ):
        """Function that aims to select the best model, the user can also add learner to flaml
        :param X: data for training
        :param y: data for training
        :param learners dict, with new learner to add
        """
        engine_parameters = {
            "time_budget": 50,
            "metric": "r2",
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
        self.X = X
        self.y = y

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

    @property
    def get_best_loss(self):
        return self.__best_loss

    @property
    def get_supported_metrics(self):
        return self.__supported_metrics

    @property
    def get_metrics_for_best_config(self):
        return self.__metrics_for_best_config

    @property
    def get_best_config(self):
        if isinstance(self.model, keras.Sequential):
            return self.model.summary()
        else:
            return self.__best_config

    @property
    def get_best_time_train_estimator(self):
        return self.__best_time_train

    @property
    def get_best_config_estimators(self):
        return self.__config_estimator

    @property
    def get_estimators_performances(self):
        """Function that returns the performances of trained estimator"""
        return self.__performance

    @property
    def model(self):
        """Function that returns the best model selected"""
        return self.__model

    def save(self, folder_name: str, file_name: str):
        """Function aims to save the model, the data and prediction in hdf5 file
        :param folder_name: The folder name where to save the hfd5 file
        :param file_name: The file name where to save the model, the data and the prediction
        """
        srgt_model = LiteModel()
        srgt_model.set(self.X, self.y, self.model)
        srgt_model.dump(folder_name, file_name)
