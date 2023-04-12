import time
from my_neural_network import MyNeuralNetwork

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from common.read_dataset import read_dataset_for_regression, read_dataset_for_classification

x_data, y_data = read_dataset_for_classification("../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")
x_data = StandardScaler().fit(x_data).transform(x_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

start = time.time()
my_model = MyNeuralNetwork(hidden_neurons=400, iterations=400)
my_model.fit(x_train, y_train)
y_pred = my_model.predict(x_test)
end = time.time()

y_pred = [1 if y > 0.5 else 0 for y in y_pred]

print(f"my accuracy: {metrics.accuracy_score(y_test, y_pred)} in {end - start} seconds")

# print("------------------------------------------------------------------")
#
# start = time.time()
# sk_model = MLPRegressor(hidden_layer_sizes=(128,), max_iter=100).fit(x_train, y_train)
# y_pred = sk_model.predict(x_test)
# end = time.time()
#
# print(f"sklearn accuracy: {metrics.mean_squared_error(y_test, y_pred, squared=False)} in {end - start} seconds")
#
# print("------------------------------------------------------------------")
#
# start = time.time()
# my_model = MyNeuralNetwork()
# my_model.fit(x_train, y_train)
# y_pred = my_model.predict(x_test)
# end = time.time()
#
# print(f"my accuracy: {metrics.mean_squared_error(y_test, y_pred, squared=False)} in {end - start} seconds")
