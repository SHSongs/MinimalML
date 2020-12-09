import numpy as np

x_train = np.array([1., 2., 3., 4., 5., 6.])
y_train = np.array([9., 16., 23., 30., 37., 44.])


W = 0.0
b = 0.0

n_data = len(x_train)

epochs = 5000
learning_rate = 0.01

for i in range(epochs):
    h = 0.001

    hypothesis = x_train * W + b

    hypothesis2_w_up = x_train * (W + h) + b
    hypothesis2_b_up = x_train * W + (b + h)

    hypothesis2_w_down = x_train * (W - h) + b
    hypothesis2_b_down = x_train * W + (b - h)

    cost = np.sum((hypothesis - y_train) ** 2) / n_data

    cost2_w_up = np.sum((hypothesis2_w_up - y_train) ** 2) / n_data
    cost2_b_up = np.sum((hypothesis2_b_up - y_train) ** 2) / n_data

    cost2_w_down = np.sum((hypothesis2_w_down - y_train) ** 2) / n_data
    cost2_b_down = np.sum((hypothesis2_b_down - y_train) ** 2) / n_data

    numerical_grad_w = (cost2_w_up - cost2_w_down)*2 / h
    numerical_grad_b = (cost2_b_up - cost2_b_down)*2 / h

    gradient_w = np.sum((W * x_train - y_train + b) * 2 * x_train) / n_data
    gradient_b = np.sum((W * x_train - y_train + b) * 2) / n_data

    W -= learning_rate * numerical_grad_w
    b -= learning_rate * numerical_grad_b

    if i % 100 == 0:
        print('Epoch ({:10d}/{:10d}) cost: {:10f}, W: {:10f}, b:{:10f}'.format(i, epochs, cost, W, b))

print('W: {:10f}'.format(W))
print('b: {:10f}'.format(b))
print('result : ')
print(x_train * W + b)