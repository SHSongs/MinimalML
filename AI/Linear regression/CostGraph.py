import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

x_train = np.array([0, 1, 2, 3, 4])
y_train = np.array([0, 1, 2, 3, 4])

W = 1.0

n_data = len(x_train)

blist = np.arange(-10, 10, 0.1)

costs = []
for b in blist:
    hypothesis = x_train * W + b
    cost = np.sum((hypothesis - y_train) ** 2) / n_data
    costs.append(cost)


x_predict = np.array(list(range(-3, 7)))
y_predict = x_predict * W + b

plt.plot(x_train, y_train, 'or', label='origin data')
plt.plot(x_predict, y_predict, 'b', label='predict')
plt.legend(['origin', 'predict'])
plt.show()

plt.plot(blist, costs, 'r', label='cost')
plt.show()