import numpy as np

x_train = np.array([0, 1, 2, 3, 4])
y_train = np.array([0, 1, 2, 3, 4])

W = 1.0
b = 5

n_data = len(x_train)

epochs = 1000
learning_rate = 0.001

for i in range(epochs):
    h = 0.001

    hypothesis_b_up = x_train * W + (b + h)
    hypothesis_b_down = x_train * W + (b - h)

    cost_b_up = np.sum((hypothesis_b_up - y_train)**2) / n_data
    cost_b_down = np.sum((hypothesis_b_down - y_train)**2) / n_data

    numerical_grad_b = (cost_b_up - cost_b_down) * 2 / h

    b -= learning_rate * numerical_grad_b


print(b)


