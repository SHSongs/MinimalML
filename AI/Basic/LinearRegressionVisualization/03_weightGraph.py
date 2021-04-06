import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

x_train = np.array([0, 1, 2, 3, 4])
y_train = np.array([0, 1, 2, 3, 4])

b = 0.0

n_data = len(x_train)

weightList = np.arange(-10, 10, 0.1)

costs = []
for W in weightList:
    hypothesis = x_train * W + b
    cost = np.sum((hypothesis - y_train) ** 2) / n_data
    costs.append(cost)

plt.plot(weightList, costs, 'r', label='cost')
plt.xlabel('weight')
plt.ylabel('loss')
plt.show()