# OptiBrain

A python package that aims to select automatically the best model for your tasks and save the trained model..

## Install

`pip install optibrain@git+https://github.com/eurobios-mews-labs/optibrain

## Simple usage

* Auto-select and save model.

```python
import pandas as pd

from sklearn.datasets import make_classification
from optibrain import SurrogateModeling
from optibrain.utils.kriging_model import KRGModel
import numpy as np

X, y = make_classification()
X = pd.DataFrame(X)
y = pd.Series(y)
estimator_list = ["catboost", 'xgboost', 'lgbm', 'KRG', 'keras']
# instanciate the metamodel
srgt = SurrogateModeling(estimator_list=estimator_list, problem='classification')
# select the best model
srgt.get_best_model(X, y, learners={"KRG": KRGModel})
# print the performances of the estimators
print(srgt.get_estimators_performances)
# save the model
srgt.save("./metamodel_folder", "file_name")
```

In the method get_best_model, the user can add new learners, by setting True to add_learner and adding learner
dictionary with
the names of the learners and their classes.  
The result of this example is a HDF5 file where the information about the selected model are saved.

The saved model, can be loaded with the package revival.



