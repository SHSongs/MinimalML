import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1., 2., 3., 4., 5., 6.])
y_train = np.array([5., 14., 20., 34., 35., 40.])

W = 0.0
b = 0.0

n_data = len(x_train)

epochs = 1000
learning_rate = 0.01

for i in range(epochs):
    h = 0.001

    hypothesis = x_train * W + b
    cost = np.sum((hypothesis - y_train) ** 2) / n_data

    hypothesis_w_up = x_train * (W + h) + b
    hypothesis_b_up = x_train * W + (b + h)

    hypothesis_w_down = x_train * (W - h) + b
    hypothesis_b_down = x_train * W + (b - h)

    cost_w_up = np.sum((hypothesis_w_up - y_train) ** 2) / n_data
    cost_b_up = np.sum((hypothesis_b_up - y_train) ** 2) / n_data

    cost_w_down = np.sum((hypothesis_w_down - y_train) ** 2) / n_data
    cost_b_down = np.sum((hypothesis_b_down - y_train) ** 2) / n_data

    numerical_grad_w = (cost_w_up - cost_w_down) * 2 / h
    numerical_grad_b = (cost_b_up - cost_b_down) * 2 / h

    W -= learning_rate * numerical_grad_w
    b -= learning_rate * numerical_grad_b

    # 편미분
    gradient_w = np.sum((W * x_train - y_train + b) * 2 * x_train) / n_data
    gradient_b = np.sum((W * x_train - y_train + b) * 2) / n_data

    if i % 100 == 0:
        print('Epoch ({:10d}/{:10d}) cost: {:10f}, W: {:10f}, b:{:10f}'.format(i, epochs, cost, W, b))

print('W: {:10f}'.format(W))
print('b: {:10f}'.format(b))
print('result : ')
print(x_train * W + b)

x_predict = np.array(list(range(1, 7)))
y_predict = x_predict * W + b

plt.plot(x_train, y_train, 'or', label='origin data')
plt.plot(x_predict, y_predict, 'b', label='predict')
plt.legend(['origin', 'predict'])
plt.show()
