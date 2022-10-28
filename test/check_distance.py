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
