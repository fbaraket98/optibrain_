from optibrain.utils.NN_model import FullNeuralNetwork

# OptiBrain

A python package that aims to select automatically the best model for your data and save the trained model.
The result of this this example is a HDF5 file where the information about the selected model is saved.

## Install
```shell
pip install optibrain@git+https://github.com/eurobios-mews-labs/optibrain
```

## Simple usage

* Auto-select and save model.

First you need to install the packages of the test :
```shell
 pip install optibrain[test]
 ```

```python
import pandas as pd

from sklearn.datasets import make_classification
from optibrain import SurrogateModeling

X, y = make_classification()
X = pd.DataFrame(X)
y = pd.Series(y)
estimator_list = ["catboost", 'xgboost', 'lgbm', 'KRG']
# instanciate the metamodel
srgt = SurrogateModeling(estimator_list=estimator_list, problem='classification')
# select the best model
srgt.find_best_model(X, y)
# print the performances of the estimators
print(srgt.estimators_performances)
# save the model
srgt.save("./metamodel_folder", "file_name")
```

In the method get_best_model, the user can add new learners, by adding learner
dictionary with the names of the learners and their classes.
```python
from optibrain.utils.NN_model import FullNeuralNetwork
#ADD the Neural Network personalized learner 
srgt.find_best_model(X,y,learners={"NN":FullNeuralNetwork})
```
The result of this example is a HDF5 file where the information about the selected model is saved.

The saved model, can be loaded with the package revival.   
See instructions in the below link : https://github.com/eurobios-mews-labs/revivAl



