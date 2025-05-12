import tempfile
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from optibrain import SurrogateModeling


def test_save_classification_prob():
    from revival import LiteModel

    X, y = make_classification()
    X = pd.DataFrame(X)
    y = pd.Series(y)
    # Optimizer Flaml parameters
    estimator_list = ["catboost", "xgboost", "lgbm"]
    # Instantiate the metamodel
    srgt = SurrogateModeling(
        estimator_list=estimator_list, problem="classification", project_name="default"
    )
    srgt.get_best_model(X, y)
    # save the model
    with tempfile.TemporaryDirectory() as tmpdir:
        srgt.save(tmpdir, "file_test")
        loadmodel = LiteModel()
        loadmodel.load(tmpdir, "file_test")

    # Asserts
    assert np.allclose(srgt.X, X), "The data X are not matching"
    assert np.allclose(srgt.y, y), "The data y are not matching"

    # Asserts
    assert np.allclose(loadmodel.X_train, X), "The data X are not matching"
    assert np.allclose(loadmodel.y_train, y), "The data y are not matching"


def test_predictions():
    from revival import LiteModel
    from sklearn.model_selection import train_test_split

    X, y = make_regression()
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    # split train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    surrogate_model = SurrogateModeling(["catboost", "lgbm", "KRG"], "regression")
    surrogate_model.get_best_model(X_train, y_train)

    # prediction with surrogate_model
    surrogate_model_prediction = surrogate_model.model.predict(X_test)
    # save the trained model
    with tempfile.TemporaryDirectory() as tmpdir:

        surrogate_model.save(tmpdir, "test_prediction")

        loaded_model = LiteModel()
        # Load the saved model
        loaded_model.load(tmpdir, "test_prediction")

    loaded_model.predict(X_test)
    loaded_model_prediction = loaded_model.prediction
    # assert predictions
    assert np.allclose(
        loaded_model_prediction, surrogate_model_prediction
    ), "the predictions are not matching"
