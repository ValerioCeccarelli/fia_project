from sklearn.model_selection import train_test_split
from common.read_dataset import read_dataset_for_regression, read_dataset_for_classification
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import time
from nearest_neighbors.my_nearest_neighbors import MyNearestNeighborsClassifier, MyNearestNeighborsRegressor


def predict_classifier_with_sklearn(x_train, y_train, x_test, y_test) -> float:
    classifier = KNeighborsClassifier(n_neighbors=5, p=2)
    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)
    return accuracy_score(y_test, y_predict)


def predict_classifier_with_my_nearest_neighbors(x_train, y_train, x_test, y_test) -> float:
    classifier = MyNearestNeighborsClassifier(distance=2, k_nearest=5)
    classifier.fit(x_train, y_train)

    y_predict = classifier.predict(x_test)
    return accuracy_score(y_test, y_predict)


def predict_regressor_with_sklearn(x_train, y_train, x_test, y_test) -> float:
    regressor = KNeighborsRegressor(n_neighbors=5, p=2)
    regressor.fit(x_train, y_train)
    y_predict = regressor.predict(x_test)
    return mean_squared_error(y_test, y_predict, squared=False)


def predict_regressor_with_my_nearest_neighbors(x_train, y_train, x_test, y_test) -> float:
    regressor = MyNearestNeighborsRegressor(distance=2, k_nearest=5)
    regressor.fit(x_train, y_train)

    y_predict = regressor.predict(x_test)
    return mean_squared_error(y_test, y_predict, squared=False)



print("CLASSIFICATION\n")
x_data, y_data = read_dataset_for_classification("../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

print("------------------------------------------------------------------")

start = time.time()
sklearn_accuracy = predict_classifier_with_sklearn(x_train, y_train, x_test, y_test)
end = time.time()
print(f"sklearn accuracy: {sklearn_accuracy}\nin {end - start} seconds")

print("------------------------------------------------------------------")

start = time.time()
my_accuracy = predict_classifier_with_my_nearest_neighbors(x_train, y_train, x_test, y_test)
end = time.time()
print(f"my accuracy: {my_accuracy}\nin {end - start} seconds")

print("------------------------------------------------------------------")

print("\n\nREGRESSION\n")
x_data, y_data = read_dataset_for_regression("../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

print("------------------------------------------------------------------")

start = time.time()
sklearn_accuracy = predict_regressor_with_sklearn(x_train, y_train, x_test, y_test)
end = time.time()
print(f"sklearn accuracy: {sklearn_accuracy}\nin {end - start} seconds")

print("------------------------------------------------------------------")

start = time.time()
my_accuracy = predict_regressor_with_my_nearest_neighbors(x_train, y_train, x_test, y_test)
end = time.time()
print(f"my accuracy: {my_accuracy}\nin {end - start} seconds")

print("------------------------------------------------------------------")
