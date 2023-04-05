import time

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from common.read_dataset import read_dataset_for_regression, read_dataset_for_classification

x_data, y_data = read_dataset_for_regression("../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")
x_data = StandardScaler().fit(x_data).transform(x_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

for i in range(1, 10):
    hidden_size = 128*i
    max_iter = 4000

    start = time.time()
    sk_model = MLPRegressor(
        hidden_layer_sizes=(hidden_size,),
        # verbose=True,
        max_iter=max_iter,
    ).fit(x_train, y_train)
    y_pred = sk_model.predict(x_test)
    end = time.time()

    print(
        f"Scikit Learn: {metrics.mean_squared_error(y_test, y_pred, squared=False)} in {end - start} seconds, hidden size: {hidden_size}, max iter: {max_iter}"
    )

# Scikit Learn: 8454.501093790243 in 479.51882433891296 seconds, hidden size: 128, max iter: 4000
# E:\Progetti\Python\fia_project\venv\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:686: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (4000) reached and the optimization hasn't converged yet.
#   warnings.warn(
# E:\Progetti\Python\fia_project\venv\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:686: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (4000) reached and the optimization hasn't converged yet.
#   warnings.warn(
# Scikit Learn: 8597.035787561581 in 681.9018676280975 seconds, hidden size: 256, max iter: 4000
# Scikit Learn: 8675.469420838072 in 849.1892602443695 seconds, hidden size: 384, max iter: 4000
# E:\Progetti\Python\fia_project\venv\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:686: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (4000) reached and the optimization hasn't converged yet.
#   warnings.warn(
# Scikit Learn: 8768.74804516016 in 1090.3696205615997 seconds, hidden size: 512, max iter: 4000
# E:\Progetti\Python\fia_project\venv\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:693: UserWarning: Training interrupted by user.
#   warnings.warn("Training interrupted by user.")
# Scikit Learn: 8318.074871631601 in 241.170889377594 seconds, hidden size: 640, max iter: 4000
# E:\Progetti\Python\fia_project\venv\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:686: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (4000) reached and the optimization hasn't converged yet.
#   warnings.warn(
# Scikit Learn: 9050.114008010472 in 1846.6594069004059 seconds, hidden size: 768, max iter: 4000