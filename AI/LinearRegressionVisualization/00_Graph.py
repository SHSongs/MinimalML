import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

x_train = np.array([5, 6, 7, 8, 9])
y_train = np.array([0, 0, 1, 1, 3])

def print_graph():

    plt.plot(x_train, y_train, 'or', label='origin data')
    plt.legend(['origin'])
    plt.xlabel('Temperature °C')
    plt.ylabel('Number of beverage bottles sold')
    plt.show()

print_graph()