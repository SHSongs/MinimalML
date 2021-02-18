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


plt.plot(blist, costs, 'r', label='cost')
plt.xlabel('bias')
plt.ylabel('loss')
plt.show()