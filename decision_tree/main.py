from common.read_dataset import read_dataset_for_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from decision_tree.my_decision_tree import MyDecisionTree
import time


def predict_with_sklearn(x_train, y_train, x_test, y_test) -> float:
    clf = tree.DecisionTreeClassifier(criterion="entropy")  # max_depth=15
    clf.fit(x_train, y_train)

    y_predict = clf.predict(x_test)
    return accuracy_score(y_test, y_predict)


def predict_with_my_decision_tree(x_train, y_train, x_test, y_test) -> float:
    my_tree = MyDecisionTree()
    my_tree.fit(x_train, y_train)

    y_predict = my_tree.predict(x_test)
    return accuracy_score(y_test, y_predict)


x_data, y_data = read_dataset_for_classification("../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

print("------------------------------------------------------------------")

start = time.time()
sklearn_accuracy = predict_with_sklearn(x_train, y_train, x_test, y_test)
end = time.time()
print(f"sklearn accuracy: {sklearn_accuracy}\nin {end - start} seconds")

print("------------------------------------------------------------------")

start = time.time()
my_accuracy = predict_with_my_decision_tree(x_train, y_train, x_test, y_test)
end = time.time()

print(f"my accuracy: {my_accuracy}\nin {end - start} seconds")

print("------------------------------------------------------------------")
