
import random

import numpy as np
from sklearn.linear_model import LinearRegression

def f(x):
    return 50 * x + 20

ins = [x-500 for x in range(1000)]
random.shuffle(ins)

X = np.array([[x] for x in ins])

y_pp = [f(x) for x in ins]
y = np.array(y_pp)
reg = LinearRegression().fit(X, y)
reg.score(X, y)
print(reg.coef_)
# print(array([1., 2.]))
print(reg.intercept_)

p = reg.predict(np.array([[4]]))
print(p)
print(f(4))
#array([16.])


# # ins = [
# #     (1,2),
# #     (2,1),
# #     (2,2),
# #     (2,3),
# #     (3,2),
# #     (4,3),
# #
# #     (6,5),
# #     (7,4),
# #     (7,5),
# #     (7,6),
# #     (8,4),
# #     (8,5),
# #     (9,5)
# # ]