import logging
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import Dict

import pandas as pd
from flaml import AutoML
from palma.base.splitting_strategy import ValidationStrategy
from sklearn import base
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

from optibrain.utils.NN_model import FullNeuralNetwork
from optibrain.utils.kriging_model import KRGModel
from optibrain.utils.project import Project
from optibrain.utils.utils import get_hash


class BaseOptimizer(metaclass=ABCMeta):

    def __init__(self, engine_parameters: dict) -> None:
        self.__engine_parameters = engine_parameters
        self.__date = datetime.now()
        self.__run_id = get_hash(date=self.__date)
        self._problem = "unknown"

    @abstractmethod
    def optimize(
        self, X: pd.DataFrame, y: pd.Series, splitter: "ValidationStrategy" = None
    ) -> None: ...

    @property
    @abstractmethod
    def best_model_(self) -> None: ...

    @property
    @abstractmethod
    def transformer_(self) -> None: ...

    @property
    def engine_parameters(self) -> Dict:
        return self.__engine_parameters

    @property
    def allow_splitter(self):
        return False

    def allowing_splitter(self, splitter):
        if not self.allow_splitter and splitter is not None:
            logging.warning(f"Optimizer does not support splitter {splitter}")

    def start(self, project: "Project"):
        from palma import logger

        self._problem = project.problem
        self.optimize(
            project.X.iloc[project.validation_strategy.train_index],
            project.y.iloc[project.validation_strategy.train_index],
            splitter=project.validation_strategy,
        )

        logger.logger.log_artifact(self.best_model_, self.__run_id)
        try:
            logger.logger.log_metrics(
                {"best_estimator": str(self.best_model_)}, "optimizer"
            )
        except:
            pass

    @property
    def run_id(self) -> str:
        return self.__run_id

    @property
    def problem(self):
        return self._problem


class FlamlOptimizer(BaseOptimizer):
    def __init__(self, engine_parameters: dict, learner_dict: dict) -> None:
        super().__init__(engine_parameters)
        self.learner = learner_dict

    def optimize(
        self, X: pd.DataFrame, y: pd.DataFrame, splitter: ValidationStrategy = None
    ) -> None:
        split_type = None if splitter is None else splitter.splitter
        groups = None if splitter is None else splitter.groups
        groups = groups if groups is None else groups[splitter.train_index]

        logging.disable()
        self.engine_parameters["task"] = self.problem
        self.allowing_splitter(splitter)
        self.__optimizer = AutoML()

        # add NeuralNetwork and KRG models to the flaml optimizer
        self.__optimizer.add_learner("NN", FullNeuralNetwork)
        self.__optimizer.add_learner("KRG", KRGModel)

        is_multi_output = isinstance(y, pd.DataFrame) and y.shape[1] > 1
        # Add new learners
        if self.learner is not None:
            for key, value in self.learner.items():
                self.__optimizer.add_learner(key, learner_class=value)
        y_train = (
            pd.Series(y.values.flatten(), index=range(len(X)))
            if not is_multi_output
            else y.iloc[:, 0].to_numpy()
        )
        self.__optimizer.fit(
            X_train=pd.DataFrame(X.values, index=range(len(X))),
            y_train=y_train,
            split_type=split_type,
            groups=groups,
            mlflow_logging=False,
            **self.engine_parameters,
        )
        if is_multi_output:
            base_model = self.__optimizer.model
        else:
            base_model = self.__optimizer.model.model

        if is_multi_output:
            if self.problem == "regression":
                self._model = MultiOutputRegressor(base_model)
            elif self.problem == "classification":
                self._model = MultiOutputClassifier(base_model)
            else:
                raise ValueError("Unknown problem type")
            self._model.fit(X, y)
        else:
            self._model = base_model
        logging.basicConfig(level=logging.DEBUG)

    @property
    def best_model_(self) -> base.BaseEstimator:
        return self._model

    @property
    def transformer_(self):
        return self.__optimizer._transformer

    @property
    def allow_splitter(self):
        return True

    @property
    def best_loss_estimator(self):
        return (1 - pd.Series(self.__optimizer.best_loss_per_estimator)).sort_values(
            ascending=False
        )

    @property
    def best_config_estimator(self):
        return self.__optimizer.best_config_per_estimator

    @property
    def best_time_estimator(self):
        return self.__optimizer.best_config_train_time

    @property
    def best_config(self):
        return self.__optimizer.best_config

    @property
    def supported_metrics(self):
        return self.__optimizer.supported_metrics

    @property
    def metrics_for_best_config(self):
        return self.__optimizer.metrics_for_best_config

    @property
    def best_loss(self):
        return 1 - pd.Series(self.__optimizer.best_loss)
