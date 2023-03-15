import random


# ins = [
#     (1,2),
#     (2,1),
#     (2,2),
#     (2,3),
#     (3,2),
#     (4,3),
#
#     (6,5),
#     (7,4),
#     (7,5),
#     (7,6),
#     (8,4),
#     (8,5),
#     (9,5)
# ]


def f(x):
    return 50 * x + 20


ins = []
for i in range(1000):
    my_rand = random.randint(-100, 100) % 100 / 10
    ins.append((i, f(i) + my_rand))

random.shuffle(ins)

m_stimata = 1
q_stimata = 1

eps = 0.0001
learning_rate = 0.00001

error_sum = float("inf")

while abs(error_sum) > eps:
    error_sum = 0
    error_sum_by_x = 0

    for i in range(10):
        for i in range(i*100, (i+1)*100):
            x, y = ins[i]
            error = y - (m_stimata * x + q_stimata)
            error_sum += error
            error_sum_by_x += error * x
    # for x, y in ins:
    #     error = y - (m_stimata * x + q_stimata)
    #     error_sum += error
    #     error_sum_by_x += error * x

        m_stimata += learning_rate * error_sum_by_x / len(ins)
        q_stimata += learning_rate * error_sum

    print(f"m_stimata: {m_stimata}, q_stimata: {q_stimata}, error_sum: {error_sum}")
    learning_rate *= 0.99

