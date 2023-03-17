from sklearn.model_selection import train_test_split
from common.read_dataset import read_dataset_with_numpy
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import time
from nearest_neighbors.my_nearest_neighbors import MyNearestNeighborsClassifier
import numpy as np

def predict_with_sklearn(x_train, y_train, x_test, y_test) -> float:
    # classifier = NearestNeighbors(n_neighbors=1, p=2)
    # classifier = classifier.fit(x_train, y_train)
    # y_predict = classifier.predict(x_test)
    # return accuracy_score(y_test, y_predict)
    return 0

def predict_with_my_nearest_neighbors(x_train, y_train, x_test, y_test) -> float:
    classifier = MyNearestNeighborsClassifier(distance=2, k_nearest=1)
    classifier.fit(x_train, y_train)

    y_predict = classifier.predict(x_test)
    return accuracy_score(y_test, y_predict)


x_data, y_data = read_dataset_with_numpy("../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

print("------------------------------------------------------------------")
start = time.time()
sklearn_accuracy = predict_with_sklearn(x_train, y_train, x_test, y_test)
end = time.time()
print(f"sklearn accuracy: {sklearn_accuracy}\nin {end - start} seconds")

print("------------------------------------------------------------------")
start = time.time()
my_accuracy = predict_with_my_nearest_neighbors(x_train, y_train, x_test, y_test)
end = time.time()
print(f"my accuracy: {my_accuracy}\nin {end - start} seconds")

print("------------------------------------------------------------------")

# start = time.time()
# x_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# y_train = np.array([10, 20, 30, 40])
# x_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# y_test = np.array([10, 20, 30])
# my_accuracy = predict_with_my_nearest_neighbors(x_train, y_train, x_test, y_test)
# end = time.time()
# print(f"{my_accuracy} in {end - start} seconds")