# TODO double DQN
import numpy as np
import matplotlib.pyplot as plt


# от Ox'y' -> Oxy
# t = np.linspace(0, 2 * np.pi, 25)
# x_ = [np.sin(t_) for t_ in t]
# y_ = [- np.cos(t_) + 1 for t_ in t]
#
#
# PC = [x_[-1] - x_[-2], y_[-1] - y_[-2]]
#
# len_PC = np.linalg.norm(PC)
#
#
# a = (len_PC / 2) + 0.05
# b = 0.05
# vertical = np.linspace(-b, b, 10)
# horizontal = np.linspace(-a, a, 10)
#
# one = np.array(np.meshgrid(np.linspace(a, a, 10), vertical)).T.reshape(-1, 2)
# two = np.array(np.meshgrid(np.linspace(-a, -a, 10), vertical)).T.reshape(-1, 2)
# three = np.array(np.meshgrid(horizontal, np.linspace(b, b, 10))).T.reshape(-1, 2)
# four = np.array(np.meshgrid(horizontal, np.linspace(-b, -b, 10))).T.reshape(-1, 2)
#
# points = np.concatenate((np.concatenate((np.concatenate((one, two), axis=0), three), axis=0), four), axis=0)
#
# for i in range(1, 25):
#     j = i - 1
#
#     PC = np.array([x_[i] - x_[j], y_[i] - y_[j]])
#
#     cos_a = np.round(np.dot(PC, [1, 0]) / len_PC, 6)
#
#     a = np.arccos(cos_a)
#     if str(a) == 'nan':
#         print()
#
#     cos_on_y = np.dot(PC, [0, 1])
#     if cos_on_y < 0:
#         a = 2 * np.pi - a
#
#     print(a)
#     M = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
#
#     x_center = x_[i-1] + (x_[i] - x_[i-1]) / 2
#     y_center = y_[i-1] + (y_[i] - y_[i-1]) / 2
#
#     x, y = np.dot(M, points.T)
#     x += x_center
#     y += y_center
#
#     plt.plot(x, y)
#
#
# plt.plot(x_, y_)
# plt.scatter(x_center + 0.04, y_center + 0.04, color='red', lw=0.01)
# plt.scatter(x, y, color='blue', lw=0.01)
# plt.scatter(2, 2, color='yellow', lw=0.01)

# plt.plot(np.linspace(a, a, 10), vertical)
# plt.plot(np.linspace(-a, -a, 10), vertical)
#
# plt.plot(horizontal, np.linspace(b, b, 10))
# plt.plot(horizontal, np.linspace(-b, -b, 10))

# plt.grid()
#
# t = np.linspace(0, 5, 100)
#
#
# for i in range(30):
#     p = []
#     w = []
#
#     phase_platform = np.random.uniform(-np.pi, np.pi, size=1)[0]
#     phase_wheel = np.random.uniform(-np.pi, np.pi, size=1)[0]
#     sigma, amp, omega = np.random.randn(3)
#     sigma = abs(sigma)
#
#     for t_ in t:
#         mu_p = 0.001 * amp * np.sin(omega * t_ + phase_platform)
#         mu_w = 0.001 * amp * np.sin(omega * t_ + phase_wheel)
#         p.append(sigma * np.random.randn(1) + mu_p)
#         w.append(sigma * np.random.randn(1) + mu_w)
#
#     plt.plot(t, p)
#     plt.plot(t, w)
#     plt.show()

import numpy as np
import shapely.geometry as geom
import matplotlib.pyplot as plt

from robot.enviroment import random_trajectory


class NearestPoint(object):
    def __init__(self, line, ax, points):
        self.points = points
        self.line = line
        self.ax = ax
        ax.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        x, y = event.xdata, event.ydata
        point = geom.Point(x, y)
        distance = self.line.distance(point)
        self.draw_segment(point)

        h_error = self.line.interpolate(self.line.project(point))

        dist = np.array([np.sqrt((h_error.x - point[0]) ** 2 + (h_error.y - point[1]) ** 2) for point in self.points])

        arr = dist.argsort()[:1]

        print("\n index = ", arr)
        print('Distance to line:', distance)

    def draw_segment(self, point):
        point_on_line = line.interpolate(line.project(point))
        self.ax.plot([point.x, point_on_line.x], [point.y, point_on_line.y],
                     color='red', marker='o', scalex=False, scaley=False)
        fig.canvas.draw()

xy = random_trajectory()

arr = np.array(xy).T

line = geom.LineString(arr)

fig, ax = plt.subplots()
ax.plot(arr[:, 0][0:], arr[:, 1][0:])
ax.scatter(arr[:, 0][0:], arr[:, 1][0:])

ax.axis('equal')
NearestPoint(line, ax, arr)

plt.show()
