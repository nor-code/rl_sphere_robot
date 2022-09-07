# TODO double DQN
import numpy as np
import matplotlib.pyplot as plt


# от Ox'y' -> Oxy
t = np.linspace(0, 2 * np.pi, 30)
x_ = [np.sin(t_) for t_ in t]
y_ = [- np.cos(t_) + 1 for t_ in t]


PC = [x_[-1] - x_[-2], y_[-1] - y_[-2]]

len_PC = np.linalg.norm(PC)

x_center = x_[-2] + (x_[-1] - x_[-2])/2
y_center = y_[-2] + (y_[-1] - y_[-2])/2
print("x_center = ", x_center)
print("y_center = ", y_center)

x = x_center + 0.04
y = y_center + 0.04

cos_a = np.dot(PC, [1, 0]) / len_PC
a = np.arccos(cos_a)

cos_on_y = np.dot(PC, [0, 1])
if cos_on_y < 0:
    a = 2 * np.pi - a


print(a)
M = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])


x -= x_center
y -= y_center
x, y = np.dot(M, [x, y])


a = (len_PC / 2) + 0.05
b = 0.05


vertical = np.linspace(-b, b, 20)
horizontal = np.linspace(-a, a, 20)

plt.plot(x_, y_)
plt.scatter(x_center + 0.04, y_center + 0.04, color='red', lw=0.01)
plt.scatter(x, y, color='blue', lw=0.01)
plt.scatter(2, 2, color='yellow', lw=0.01)

plt.plot(np.linspace(a, a, 20), vertical)
plt.plot(np.linspace(-a, -a, 20), vertical)

plt.plot(horizontal, np.linspace(b, b, 20))
plt.plot(horizontal, np.linspace(-b, -b, 20))

plt.grid()

print("x = ", x, "y = ", y)
plt.show()