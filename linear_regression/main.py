from common.read_dataset import read_dataset_for_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from linear_regression.my_linear_regression import MyLinearRegressor, MyLinearClassifier
import time

def sk_predict(x_train, y_train, x_test, y_test):
    start = time.time()
    sk_regressor = SGDRegressor(penalty=None, max_iter=1000, shuffle=False, learning_rate="constant", eta0=0.01,
                                early_stopping=False)
    sk_regressor = sk_regressor.fit(x_train, y_train)
    sk_y_pred = sk_regressor.predict(x_test)
    end = time.time()

    print(f"sklearn RMSE: {metrics.mean_squared_error(y_test, sk_y_pred, squared=False)}")
    print(f"in {end - start} seconds")

    return sk_y_pred


def sk_predict_l1(x_train, y_train, x_test, y_test):
    start = time.time()
    sk_regressor = SGDRegressor(penalty="l1", max_iter=1000, shuffle=False, learning_rate="constant", eta0=0.01,
                                early_stopping=False, l1_ratio=1)
    sk_regressor = sk_regressor.fit(x_train, y_train)
    sk_y_pred = sk_regressor.predict(x_test)
    end = time.time()

    print(f"sklearn L1 penalty RMSE: {metrics.mean_squared_error(y_test, sk_y_pred, squared=False)}")
    print(f"in {end - start} seconds")

    return sk_y_pred


def sk_predict_l2(x_train, y_train, x_test, y_test):
    start = time.time()
    sk_regressor = SGDRegressor(penalty="l2", max_iter=1000, shuffle=False, learning_rate="constant", eta0=0.01,
                                early_stopping=False, alpha=1)
    sk_regressor = sk_regressor.fit(x_train, y_train)
    sk_y_pred = sk_regressor.predict(x_test)
    end = time.time()

    print(f"sklearn L2 penalty RMSE: {metrics.mean_squared_error(y_test, sk_y_pred, squared=False)}")
    print(f"in {end - start} seconds")

    return sk_y_pred


def sk_predict_elasticnet(x_train, y_train, x_test, y_test):
    start = time.time()
    sk_regressor = SGDRegressor(penalty="elasticnet", max_iter=1000, shuffle=False, learning_rate="constant", eta0=0.01,
                                early_stopping=False, alpha=0.5, l1_ratio=0.5)
    sk_regressor = sk_regressor.fit(x_train, y_train)
    sk_y_pred = sk_regressor.predict(x_test)
    end = time.time()

    print(f"sklearn elasticnet penalty RMSE: {metrics.mean_squared_error(y_test, sk_y_pred, squared=False)}")
    print(f"in {end - start} seconds")

    return sk_y_pred


def my_predict(x_train, y_train, x_test, y_test):
    start = time.time()
    my_regressor = MyLinearRegressor(learning_rate=0.01, iterations=1000)
    my_regressor.fit(x_train, y_train)
    my_y_pred = my_regressor.predict(x_test)
    end = time.time()

    print(f"my RMSE: {metrics.mean_squared_error(y_test, my_y_pred, squared=False)}")
    print(f"in {end - start} seconds")

    return my_y_pred


def my_predict_l1(x_train, y_train, x_test, y_test):
    start = time.time()
    my_regressor = MyLinearRegressor(learning_rate=0.01, iterations=1000, l1_penalty=1)
    my_regressor.fit(x_train, y_train)
    my_y_pred = my_regressor.predict(x_test)
    end = time.time()

    print(f"my L1 RMSE: {metrics.mean_squared_error(y_test, my_y_pred, squared=False)}")
    print(f"in {end - start} seconds")

    return my_y_pred


def my_predict_l2(x_train, y_train, x_test, y_test):
    start = time.time()
    my_regressor = MyLinearRegressor(learning_rate=0.01, iterations=1000, l2_penalty=1)
    my_regressor.fit(x_train, y_train)
    my_y_pred = my_regressor.predict(x_test)
    end = time.time()

    print(f"my L2 RMSE: {metrics.mean_squared_error(y_test, my_y_pred, squared=False)}")
    print(f"in {end - start} seconds")

    return my_y_pred


def my_predict_elasticnet(x_train, y_train, x_test, y_test):
    start = time.time()
    my_regressor = MyLinearRegressor(learning_rate=0.01, iterations=1000, l2_penalty=0.5, l1_penalty=0.5)
    my_regressor.fit(x_train, y_train)
    my_y_pred = my_regressor.predict(x_test)
    end = time.time()

    print(f"my elasticnet RMSE: {metrics.mean_squared_error(y_test, my_y_pred, squared=False)}")
    print(f"in {end - start} seconds")

    return my_y_pred


print("REGRESSION")

x_data, y_data = read_dataset_for_regression("../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")
x_data = StandardScaler().fit(x_data).transform(x_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

print("------------------------------------------------------------------")

sk_predict(x_train, y_train, x_test, y_test)

print("------------------------------------------------------------------")

my_predict(x_train, y_train, x_test, y_test)

print("------------------------------------------------------------------")

sk_predict_l1(x_train, y_train, x_test, y_test)

print("------------------------------------------------------------------")

my_predict_l1(x_train, y_train, x_test, y_test)

print("------------------------------------------------------------------")

sk_predict_l2(x_train, y_train, x_test, y_test)

print("------------------------------------------------------------------")

my_predict_l2(x_train, y_train, x_test, y_test)

print("------------------------------------------------------------------")

sk_pred = sk_predict_elasticnet(x_train, y_train, x_test, y_test)

print("------------------------------------------------------------------")

my_pred = my_predict_elasticnet(x_train, y_train, x_test, y_test)

print("------------------------------------------------------------------")

print("\n\n\n")
print("CLASSIFICATION")

y_test = np.array([True if x >= 1400 else False for x in y_test])
y_train = np.array([True if x >= 1400 else False for x in y_train])

print("------------------------------------------------------------------")

start = time.time()
sk_classifier = SGDClassifier(penalty=None, max_iter=1000, shuffle=False, learning_rate="constant", eta0=0.01, early_stopping=False)
sk_classifier.fit(x_train, y_train)
sk_y_pred = sk_classifier.predict(x_test)
end = time.time()

print(f"sklearn accuracy: {metrics.accuracy_score(y_test, sk_y_pred)}")
print(f"in {end - start} seconds")

print("------------------------------------------------------------------")

start = time.time()
classifier = MyLinearClassifier(learning_rate=0.01, iterations=1000)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
end = time.time()

print(f"my accuracy: {metrics.accuracy_score(y_test, y_pred)}")
print(f"in {end - start} seconds")

print("------------------------------------------------------------------")
