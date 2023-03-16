import random


m_vera = 20
b_vera = 50

def f(x):
    return m_vera * x + b_vera

def generate_data(n):
    data = []
    for i in range(n):
        x = random.random() * 100
        y = f(x) + random.random() * 10 - 5
        data.append((x, y))
    return data

def linear_regression(data):
    x_sum = 0
    y_sum = 0
    x2_sum = 0
    xy_sum = 0
    n = len(data)
    for x, y in data:
        x_sum += x
        y_sum += y
        x2_sum += x * x
        xy_sum += x * y
    m = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
    b = (y_sum - m * x_sum) / n
    return m, b

w = [50, 20, 65]

def fw(x):
    return w[2] * x[1] + w[1] * x[0] + w[0]

def generate_data2(n):
    data = []
    for i in range(n):
        x = random.random() * 100
        y = random.random() * 100
        z = fw((x, y)) + random.random() * 10 - 5
        data.append((x, y, z))
    return data

def multivariate_linear_regression(data):
    x1_sum = 0
    x2_sum = 0
    y_sum = 0
    x1x2_sum = 0
    x1y_sum = 0
    x2y_sum = 0
    x1x1_sum = 0
    x2x2_sum = 0
    n = len(data)
    for x1, x2, y in data:
        x1_sum += x1
        x2_sum += x2
        y_sum += y
        x1x2_sum += x1 * x2
        x1y_sum += x1 * y
        x2y_sum += x2 * y
        x1x1_sum += x1 * x1
        x2x2_sum += x2 * x2
    a = (n * x1x2_sum - x1_sum * x2_sum) / (n * x1x1_sum - x1_sum * x1_sum)
    b = (x2_sum - a * x1_sum) / n
    c = (n * x1y_sum - x1_sum * y_sum) / (n * x1x1_sum - x1_sum * x1_sum)
    d = (n * x2y_sum - x2_sum * y_sum) / (n * x2x2_sum - x2_sum * x2_sum)
    e = (y_sum - c * x1_sum - d * x2_sum) / n
    return a, b, c, d, e


def main():
    data = generate_data(1000)
    m, b = linear_regression(data)
    print(m, b)
    print(f(4))

    data2 = generate_data2(1000)
    a, b, c, d, e = multivariate_linear_regression(data2)
    print(a, b, c, d, e)
    print(fw((4, 5)))

if __name__ == '__main__':
    main()
