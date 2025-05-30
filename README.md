# OptiBrain

A python package that aims to automatically select the best model for your data and save the trained model.
The information selected model and information is saved to an efficient and portable HDF5 file.

## Install
To install the basic version
```shell
pip install optibrain@git+https://github.com/eurobios-mews-labs/optibrain
```

## Simple usage
For this example, you need to download the repository and install the test packages :
```shell
 pip install optibrain[test]
 ```

Then run

```python
import pandas as pd

from sklearn.datasets import make_regression
from optibrain import SurrogateModeling

X, y = make_regression()
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
dictionary with the names of the learners and their classes, for example
```python
from optibrain.utils.NN_model import FullNeuralNetwork
#ADD the Neural Network personalized learner 
srgt.find_best_model(X,y,learners={"NN":FullNeuralNetwork})
```
The result of this example is a HDF5 file where information on the selected model is saved. The saved model can be loaded with the revival package. See instructions [here](https://github.com/eurobios-mews-labs/revivAl).



