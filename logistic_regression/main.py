from common.read_dataset import read_dataset_for_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time
from sklearn import metrics
from my_logistic_regression import MyLogisticClassifier

x_data, y_data = read_dataset_for_classification("../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")
x_data = StandardScaler().fit(x_data).transform(x_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

print("------------------------------------------------------------------")

start = time.time()
sk_classifier = LogisticRegression(penalty=None, max_iter=1000)
sk_classifier = sk_classifier.fit(x_train, y_train)
sk_y_pred = sk_classifier.predict(x_test)
end = time.time()


print(f"sklearn accuracy: {metrics.accuracy_score(sk_y_pred, y_test)}")
print(f"in {end - start} seconds")

print("------------------------------------------------------------------")

start = time.time()
my_classifier = MyLogisticClassifier(iterations=1000, learning_rate=0.01)
my_classifier.fit(x_train, y_train)
my_y_pred = my_classifier.predict(x_test)
end = time.time()

print(f"my accuracy: {metrics.accuracy_score(my_y_pred, y_test)}")
print(f"in {end - start} seconds")

print("------------------------------------------------------------------")
