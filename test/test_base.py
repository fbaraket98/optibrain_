import tempfile

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from optibrain import SurrogateModeling


def test_save_classification_prob():
    from revival import load_model

    X, y = make_classification()
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    # Optimizer Flaml parameters
    estimator_list = ["catboost", "xgboost", "lgbm"]
    # Instantiate the metamodel
    srgt = SurrogateModeling(
        estimator_list=estimator_list, problem="classification", project_name="default"
    )
    srgt.find_best_model(X, y)
    # save the model
    with tempfile.TemporaryDirectory() as tmpdir:
        srgt.save(tmpdir, "file_test")
        loadmodel = load_model(tmpdir, "file_test")

    # Asserts
    assert np.allclose(srgt.X, X), "The data X are not matching"
    assert np.allclose(srgt.y, y), "The data y are not matching"

    # Asserts
    assert np.allclose(loadmodel.X_train, srgt.X_train), "The data X are not matching"
    assert np.allclose(loadmodel.y_train, srgt.y_train), "The data y are not matching"


def test_predictions():
    from revival import load_model

    X, y = make_regression()
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    # split train and test datasets

    surrogate_model = SurrogateModeling(["catboost", "lgbm", "KRG"], "regression")
    surrogate_model.find_best_model(X, y)

    # prediction with surrogate_model
    surrogate_model_prediction = surrogate_model.predict(surrogate_model.X_test)
    # save the trained model
    with tempfile.TemporaryDirectory() as tmpdir:

        surrogate_model.save(tmpdir, "test_prediction")
        # load the trained model
        loaded_model = load_model(tmpdir, "test_prediction")

    loaded_model_prediction = loaded_model.predict(surrogate_model.X_test)
    # assert predictions
    assert np.allclose(
        loaded_model_prediction, surrogate_model_prediction
    ), "the predictions are not matching"
