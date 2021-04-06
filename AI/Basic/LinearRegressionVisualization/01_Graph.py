import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

x_train = np.array([0, 1, 2, 3, 4])
y_train = np.array([0, 1, 2, 3, 4])

def print_graph():
    x_predict = np.array(list(range(-3, 7)))
    y_predict = x_predict * W + b

    hypothesis = x_train * W + b
    cost = np.sum((hypothesis - y_train) ** 2) / 5

    print(cost)
    plt.plot(x_train, y_train, 'or', label='origin data')
    plt.plot(x_predict, y_predict, 'b', label='predict')
    plt.legend(['origin', 'predict'])
    plt.xlabel('Temperature °C')
    plt.ylabel('Number of beverage bottles sold')
    plt.show()


W = 1.0
b = 0.0
print_graph()

W = 1.0
b = 5.0
print_graph()

W = 1.0
b = -5.0
print_graph()