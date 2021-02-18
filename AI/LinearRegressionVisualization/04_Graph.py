import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

matplotlib.use('Qt5Agg')

x_train = np.array([0, 1, 2, 3, 4])
y_train = np.array([0, 1, 2, 3, 4])

b = 0.0

n_data = len(x_train)

# weightList = np.linspace(-6, 6, 30)
# biasList = np.linspace(-6, 6, 30)
weightList = np.arange(-6, 6, 0.1)
biasList = np.arange(-6, 6, 0.1)

costs = []
for i, W in enumerate(weightList):
    costs.append([])
    for b in biasList:
        hypothesis = x_train * W + b
        cost = np.sum((hypothesis - y_train) ** 2) / n_data
        costs[i].append(cost)

X, Y = np.meshgrid(weightList, biasList)
Z = np.array(costs)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('weight')
ax.set_ylabel('bias')
ax.set_zlabel('loss')

plt.show()
