











import numpy as np
import time

x = [x for x in range(999999)]
xn = np.array(x)

print("--------------------")

start = time.time()
for i in range(100):
    for j in range(len(x)):
        x[j] = 2*x[j]
end = time.time()
print(f"list time: {end - start}")

start = time.time()
for i in range(100):
    xn = 2*xn
end = time.time()
print(f"list time: {end - start}")