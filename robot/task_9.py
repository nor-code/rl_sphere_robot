import numpy as np
from numpy.linalg import norm
from dm_control.suite import base
import shapely.geometry as geom


def vector(pointA, pointB):
    return [pointB[0] - pointA[0], pointB[1] - pointA[1]]


class TrakingTrajectoryTask9(base.Task):

    def __init__(self, trajectory_x_y, begin_index, timeout, R=0.242, random=None):
        """ тайм-аут одного эпизода """
        self.timeout = timeout
        """ количество точек, которые мы достигли в рамках текущего эпищода """
        self.achievedPoints = 0
        """ радиус окрестности для 4х ближайших точек """
        self.radius = R

        """ целевая траектория и начальная точка на ней """
        self.point_x_y = trajectory_x_y
        self.points = np.copy(self.point_x_y, order='K')
        self.line = geom.LineString(self.points)

        self.begin_index = begin_index
        self.no_return_index = self.get_index(begin_index)

        self.index1 = self.get_index(begin_index - 4)
        self.index2 = self.get_index(begin_index - 3)
        self.index3 = self.get_index(begin_index - 2)
        self.index4 = self.get_index(begin_index - 1)
        self.index5 = begin_index
        self.index6 = self.get_index(begin_index + 1)
        self.index7 = self.get_index(begin_index + 2)
        self.index8 = self.get_index(begin_index + 3)
        self.index9 = self.get_index(begin_index + 4)

        self.prev_index1 = self.get_index(begin_index-4)

        self.dist = 0.0
        self.robot_position = [self.points[begin_index][0], self.points[begin_index][1]]

        """ общее количество точек на пути """
        self.totalPoint = len(self.points)

        """ количество неправильных состояний за эпизод """
        self.count_invalid_states = 0
        self.count_hard_invalid_state = 0

        self.state = []

        super().__init__(random=random)

    def get_index(self, index):
        if index < 0:
            return len(self.points) + index
        if index > len(self.points) - 1:
            return index % len(self.points)
        return index

    def initialize_episode(self, physics):
        self.points = np.copy(self.point_x_y, order='K')
        x, y = self.robot_position

        index = self.begin_index
        print("begin index = ", index)

        self.get_nearest_10_points_index()
        self.no_return_index = self.index1
        self.dist = np.linalg.norm([x - self.points[self.no_return_index][0],
                                    y - self.points[self.no_return_index][1]])
        print("init no return point = ", self.no_return_index)

        physics.named.data.qpos[0:3] = [self.points[index][0], self.points[index][1], 0.2]
        physics.named.data.qvel[:] = 0

        self.count_invalid_states = 0
        self.count_hard_invalid_state = 0

        v_r1 = vector(self.robot_position, self.points[self.index1])
        v_r2 = vector(self.robot_position, self.points[self.index2])
        v_r3 = vector(self.robot_position, self.points[self.index3])
        v_r4 = vector(self.robot_position, self.points[self.index4])
        v_r5 = vector(self.robot_position, self.points[self.index5])
        v_r6 = vector(self.robot_position, self.points[self.index6])
        v_r7 = vector(self.robot_position, self.points[self.index7])
        v_r8 = vector(self.robot_position, self.points[self.index8])
        v_r9 = vector(self.robot_position, self.points[self.index9])

        self.state = [v_r1[0], v_r1[1],
                      v_r2[0], v_r2[1],
                      v_r3[0], v_r3[1],
                      v_r4[0], v_r4[1],
                      v_r5[0], v_r5[1],
                      v_r6[0], v_r6[1],
                      v_r7[0], v_r7[1],
                      v_r8[0], v_r8[1],
                      v_r9[0], v_r9[1],
                      0.0
        ]

        super().initialize_episode(physics)

    def get_nearest_10_points_index(self):
        x, y = self.robot_position

        point = geom.Point(x, y)
        h_error = self.line.interpolate(self.line.project(point))

        dist = np.array([np.sqrt((h_error.x - point[0]) ** 2 + (h_error.y - point[1]) ** 2) for point in self.points])
        arr = dist.argsort()[:1]

        self.prev_index1 = self.index1

        self.index1 = self.get_index(arr[0] - 4)
        self.index2 = self.get_index(arr[0] - 3)
        self.index3 = self.get_index(arr[0] - 2)
        self.index4 = self.get_index(arr[0] - 1)
        self.index5 = arr[0]
        self.index6 = self.get_index(arr[0] + 1)
        self.index7 = self.get_index(arr[0] + 2)
        self.index8 = self.get_index(arr[0] + 3)
        self.index9 = self.get_index(arr[0] + 4)

        if (self.no_return_index == len(self.points) - 1 and self.prev_index1 == 0)\
                or self.no_return_index < self.prev_index1:
            self.no_return_index = self.prev_index1
            self.dist = np.linalg.norm([x - self.points[self.no_return_index][0],
                                        y - self.points[self.no_return_index][1]])
            print("new no return point = ", self.no_return_index)

    def get_observation(self, physics):
        # координаты центра колеса
        x, y, z = physics.named.data.geom_xpos['wheel_']
        # вектор скорости в абс системе координат
        v_x, v_y, v_z = physics.named.data.sensordata['wheel_vel']

        self.robot_position = [x, y]

        self.get_nearest_10_points_index()

        point = geom.Point(x, y)
        h_error_dist = self.line.distance(point)

        v_r1 = vector(self.robot_position, self.points[self.index1])
        v_r2 = vector(self.robot_position, self.points[self.index2])
        v_r3 = vector(self.robot_position, self.points[self.index3])
        v_r4 = vector(self.robot_position, self.points[self.index4])
        v_r5 = vector(self.robot_position, self.points[self.index5])
        v_r6 = vector(self.robot_position, self.points[self.index6])
        v_r7 = vector(self.robot_position, self.points[self.index7])
        v_r8 = vector(self.robot_position, self.points[self.index8])
        v_r9 = vector(self.robot_position, self.points[self.index9])

        self.state = [v_r1[0], v_r1[1],
                      v_r2[0], v_r2[1],
                      v_r3[0], v_r3[1],
                      v_r4[0], v_r4[1],
                      v_r5[0], v_r5[1],
                      v_r6[0], v_r6[1],
                      v_r7[0], v_r7[1],
                      v_r8[0], v_r8[1],
                      v_r9[0], v_r9[1],
                      h_error_dist
        ]
        return self.state  # np.concatenate((xy, acc_gyro), axis=0)

    def get_termination(self, physics):
        if len(self.points) == 0 or physics.data.time > self.timeout or self.count_invalid_states >= 1\
                or len(self.points) == self.achievedPoints or self.count_hard_invalid_state >= 1:
            print("end episode at t = ", np.round(physics.data.time, 2), "\n")
            return 0.0

    def get_reward(self, physics):

        h_error_dist = self.state[-1]

        if h_error_dist >= 0.1:
            print("soft invalid state")
            self.count_invalid_states += 1
            return -10

        if self.is_invalid_state_hard():
            print("hard invalid state")
            self.count_hard_invalid_state += 1
            return -10

        if self.count_invalid_states > 0:
            print("вернулись на траекторию")
            self.count_invalid_states = 0

        if h_error_dist <= 0.01:
            return 10

        reward = -17.951 + (1 / (0.02667 + h_error_dist))

        return reward

    # TODO check. если робот повернул назад, обычно бывает из-за резкого поворота назад
    def is_invalid_state_hard(self):
        x, y = self.robot_position
        new_dist = np.linalg.norm([x - self.points[self.no_return_index][0], y - self.points[self.no_return_index][1]])
        # if new_dist <= self.dist:
        #     print("new dist = ", round(new_dist, 4), " no return dist = ", round(self.dist, 4))
        #     return True
        # поскольку индекс невозврата неуменьшается (всегда),
        # то эта проверка - еще один способ проверить не свернул ли робот назад
        if (self.no_return_index == len(self.points) - 1 and self.index1 == 0) or (self.no_return_index - self.index1 >= 72):
            return False
        if self.no_return_index > self.index1: # or self.no_return_index == self.index2 or self.no_return_index == self.index3 or self.no_return_index == self.index4:
            print("index1 = ", self.index1)
            return True
        return False
