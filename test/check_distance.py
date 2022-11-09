import numpy as np
import shapely.geometry as geom
import matplotlib.pyplot as plt

from robot.enviroment import random_trajectory


class NearestPoint(object):
    def __init__(self, line, ax, points):
        self.points = points
        self.line = line
        self.ax = ax
        self.arr = 0
        ax.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        x, y = event.xdata, event.ydata
        point = geom.Point(x, y)
        distance = self.line.distance(point)
        self.draw_segment(point)

        h_error = self.line.interpolate(self.line.project(point))

        dist = np.array([np.sqrt((h_error.x - point[0]) ** 2 + (h_error.y - point[1]) ** 2) for point in self.points])

        self.arr = dist.argsort()[:1][0]
        self.draw_nearest_line(point, -5)
        self.draw_nearest_line(point, -4)
        self.draw_nearest_line(point, -3)
        self.draw_nearest_line(point, -2)
        self.draw_nearest_line(point, -1)
        self.draw_nearest_line(point, 0)
        self.draw_nearest_line(point, 1)
        self.draw_nearest_line(point, 2)
        self.draw_nearest_line(point, 3)
        self.draw_nearest_line(point, 4)
        self.draw_nearest_line(point, 5)

        print("\n index = ", self.arr)
        print('Distance to line:', distance)

    def draw_segment(self, point):
        point_on_line = line.interpolate(line.project(point))
        self.ax.plot([point.x, point_on_line.x], [point.y, point_on_line.y],
                     color='red', marker='o', scalex=False, scaley=False)
        self.ax.annotate("O", xy=(point.x, point.y), fontsize=12)
        fig.canvas.draw()

    def draw_nearest_line(self, point, i):
        point_on_line = self.points[self.get_index(self.arr + i)]
        dist = np.sqrt((point_on_line[0] - point.x) ** 2 + (point_on_line[1] - point.y) ** 2)
        print(i, " distanse = ", round(dist, 3))
        self.ax.plot([point_on_line[0], point.x], [point_on_line[1], point.y],
                     color='green', scalex=False, scaley=False)
        self.ax.annotate(str(i), xy=(point_on_line[0], point_on_line[1]), fontsize=10)
        fig.canvas.draw()

    def get_index(self, index):
        if index < 0:
            return len(self.points) + index
        if index > len(self.points) - 1:
            return index % len(self.points)
        return index

xy = random_trajectory()

arr = np.array(xy).T

# for i in range(1, 75):
#     l = np.sqrt((arr[i][1] - arr[i-1][1])**2 + (arr[i][0] - arr[i-1][0])**2)
#     print(l)

line = geom.LineString(arr)

fig, ax = plt.subplots()
ax.plot(arr[:, 0][0:], arr[:, 1][0:])
ax.scatter(arr[:, 0][0:], arr[:, 1][0:], s=2.1)

ax.axis('equal')
NearestPoint(line, ax, arr)

plt.show()
