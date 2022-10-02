# TODO double DQN
import numpy as np
import matplotlib.pyplot as plt


# от Ox'y' -> Oxy
t = np.linspace(0, 2 * np.pi, 25)
x_ = [np.sin(t_) for t_ in t]
y_ = [- np.cos(t_) + 1 for t_ in t]


PC = [x_[-1] - x_[-2], y_[-1] - y_[-2]]

len_PC = np.linalg.norm(PC)


a = (len_PC / 2) + 0.05
b = 0.05
vertical = np.linspace(-b, b, 10)
horizontal = np.linspace(-a, a, 10)

one = np.array(np.meshgrid(np.linspace(a, a, 10), vertical)).T.reshape(-1, 2)
two = np.array(np.meshgrid(np.linspace(-a, -a, 10), vertical)).T.reshape(-1, 2)
three = np.array(np.meshgrid(horizontal, np.linspace(b, b, 10))).T.reshape(-1, 2)
four = np.array(np.meshgrid(horizontal, np.linspace(-b, -b, 10))).T.reshape(-1, 2)

points = np.concatenate((np.concatenate((np.concatenate((one, two), axis=0), three), axis=0), four), axis=0)

for i in range(1, 25):
    j = i - 1

    PC = np.array([x_[i] - x_[j], y_[i] - y_[j]])

    cos_a = np.round(np.dot(PC, [1, 0]) / len_PC, 6)

    a = np.arccos(cos_a)
    if str(a) == 'nan':
        print()

    cos_on_y = np.dot(PC, [0, 1])
    if cos_on_y < 0:
        a = 2 * np.pi - a

    print(a)
    M = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    x_center = x_[i-1] + (x_[i] - x_[i-1]) / 2
    y_center = y_[i-1] + (y_[i] - y_[i-1]) / 2

    x, y = np.dot(M, points.T)
    x += x_center
    y += y_center

    plt.plot(x, y)


plt.plot(x_, y_)
# plt.scatter(x_center + 0.04, y_center + 0.04, color='red', lw=0.01)
# plt.scatter(x, y, color='blue', lw=0.01)
# plt.scatter(2, 2, color='yellow', lw=0.01)

# plt.plot(np.linspace(a, a, 10), vertical)
# plt.plot(np.linspace(-a, -a, 10), vertical)
#
# plt.plot(horizontal, np.linspace(b, b, 10))
# plt.plot(horizontal, np.linspace(-b, -b, 10))

plt.grid()

print("x = ", x, "y = ", y)
plt.show()