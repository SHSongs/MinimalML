import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib

matplotlib.use('Qt5Agg')

fig = plt.figure()
ax = plt.axes(xlim=(-2, 7), ylim=(-2, 7))
line, = ax.plot([], [], lw=2)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    x = np.linspace(-2, 10, 1000)
    y = x * 1 + ((i - 5) / 2)

    line.set_data(x, y)
    return line,


x_train = np.array([0, 1, 2, 3, 4])
y_train = np.array([0, 1, 2, 3, 4])

plt.plot(x_train, y_train, 'or', label='origin data')

W = 1.0
b = 0.0

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=20, interval=1000, blit=True)

plt.show()