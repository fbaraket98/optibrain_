import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from optibrain import SurrogateModeling
from optibrain.utils.NN_model import FullNeuralNetwork
from optibrain.utils.kriging_model import KRGModel


def test_save():
    X, y = make_classification()
    X = pd.DataFrame(X)
    y = pd.Series(y)
    # Optimizer Flaml parameters
    estimator_list = ["catboost", "xgboost", "lgbm", "KRG", "RN"]
    learners = {"KRG": KRGModel, "RN": FullNeuralNetwork}
    # Instanciate the metamodel
    srgt = SurrogateModeling(
        estimator_list=estimator_list, problem="regression", project_name="default"
    )
    # Get the best model, wihth adding new learner to flaml
    srgt.get_best_model(X, y, learners=learners)
    # Save the model
    # Asserts
    assert np.allclose(srgt.X, X), "The data X are not matching"
    assert np.allclose(srgt.y, y), "The data y are not matching"
