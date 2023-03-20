import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from common.read_dataset import read_dataset_for_regression

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

x_data, y_data = read_dataset_for_regression("../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

reg = make_pipeline(StandardScaler(),
                     SGDRegressor(max_iter=100000, tol=1e-3, penalty=None, learning_rate="constant", eta0=0.000001))
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print(mean_squared_error(y_test, y_pred, squared=False))

# 2642428720.846702
# 2518821736.208726 # with penalty=None
# 5450479.277792359 # with penalty=None, learning_rate="constant", eta0=0.0001
# 3389459.071302906 # 10x
# 4566850.796792029 # 100x
# 8265.948216106564 # 100x 0.1x
# 8265.335532821098 # 100x 0.01x
