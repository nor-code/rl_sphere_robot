import numpy as np
from numpy.linalg import norm
from dm_control.suite import base
import shapely.geometry as geom


def vector(pointA, pointB):
    return [pointB[0] - pointA[0], pointB[1] - pointA[1]]


class TrakingTrajectoryTask10(base.Task):

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

        self.index1 = self.get_index(begin_index)
        self.index2 = self.get_index(begin_index + 1)
        self.index3 = self.get_index(begin_index + 2)
        self.index4 = self.get_index(begin_index + 3)
        self.index5 = self.get_index(begin_index + 4)
        self.index6 = self.get_index(begin_index + 5)
        self.index7 = self.get_index(begin_index + 6)
        self.index8 = self.get_index(begin_index + 7)
        self.index9 = self.get_index(begin_index + 8)
        self.index10 = self.get_index(begin_index + 9)
        self.index11 = self.get_index(begin_index + 10)
        self.index12 = self.get_index(begin_index + 11)
        self.index13 = self.get_index(begin_index + 12)
        self.index14 = self.get_index(begin_index + 13)
        self.index15 = self.get_index(begin_index + 14)
        self.index16 = self.get_index(begin_index + 15)
        self.index17 = self.get_index(begin_index + 16)
        self.index18 = self.get_index(begin_index + 17)
        self.index19 = self.get_index(begin_index + 18)
        self.index20 = self.get_index(begin_index + 19)

        self.prev_index1 = self.get_index(begin_index)

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

        shell_matrix_orientation = physics.named.data.xmat["shell"]
        shell_matrix_orientation.resize(3, 3)
        matrix = shell_matrix_orientation[0:2, 0:2]
        matrix = matrix.T

        v_r1 = np.dot(matrix, vector(self.robot_position, self.points[self.index1]))
        v_r2 = np.dot(matrix, vector(self.robot_position, self.points[self.index2]))
        v_r3 = np.dot(matrix, vector(self.robot_position, self.points[self.index3]))
        v_r4 = np.dot(matrix, vector(self.robot_position, self.points[self.index4]))
        v_r5 = np.dot(matrix, vector(self.robot_position, self.points[self.index5]))
        v_r6 = np.dot(matrix, vector(self.robot_position, self.points[self.index6]))
        v_r7 = np.dot(matrix, vector(self.robot_position, self.points[self.index7]))
        v_r8 = np.dot(matrix, vector(self.robot_position, self.points[self.index8]))
        v_r9 = np.dot(matrix, vector(self.robot_position, self.points[self.index9]))
        v_r10 = np.dot(matrix, vector(self.robot_position, self.points[self.index10]))
        v_r11 = np.dot(matrix, vector(self.robot_position, self.points[self.index11]))
        v_r12 = np.dot(matrix, vector(self.robot_position, self.points[self.index12]))
        v_r13 = np.dot(matrix, vector(self.robot_position, self.points[self.index13]))
        v_r14 = np.dot(matrix, vector(self.robot_position, self.points[self.index14]))
        v_r15 = np.dot(matrix, vector(self.robot_position, self.points[self.index15]))
        v_r16 = np.dot(matrix, vector(self.robot_position, self.points[self.index16]))
        v_r17 = np.dot(matrix, vector(self.robot_position, self.points[self.index17]))
        v_r18 = np.dot(matrix, vector(self.robot_position, self.points[self.index18]))
        v_r19 = np.dot(matrix, vector(self.robot_position, self.points[self.index19]))
        v_r20 = np.dot(matrix, vector(self.robot_position, self.points[self.index20]))

        self.state = [v_r1[0], v_r1[1], # norm(v_r1),
                      v_r2[0], v_r2[1], # norm(v_r2),
                      v_r3[0], v_r3[1], # norm(v_r3),
                      v_r4[0], v_r4[1], # norm(v_r4),
                      v_r5[0], v_r5[1], # norm(v_r5),
                      v_r6[0], v_r6[1], # norm(v_r6),
                      v_r7[0], v_r7[1], # norm(v_r7),
                      v_r8[0], v_r8[1], # norm(v_r8),
                      v_r9[0], v_r9[1], # norm(v_r9),
                      v_r10[0], v_r10[1], #norm(v_r10),
                      v_r11[0], v_r11[1], #norm(v_r11),
                      v_r12[0], v_r12[1], #norm(v_r12),
                      v_r13[0], v_r13[1], #norm(v_r13),
                      v_r14[0], v_r14[1], #norm(v_r14),
                      v_r15[0], v_r15[1], #norm(v_r15),
                      v_r16[0], v_r16[1], #norm(v_r16),
                      v_r17[0], v_r17[1], #norm(v_r17),
                      v_r18[0], v_r18[1], #norm(v_r18),
                      v_r19[0], v_r19[1], #norm(v_r19),
                      v_r20[0], v_r20[1]] #norm(v_r20)]

        super().initialize_episode(physics)

    def get_nearest_10_points_index(self):
        x, y = self.robot_position

        point = geom.Point(x, y)
        h_error = self.line.interpolate(self.line.project(point))

        dist = np.array([np.sqrt((h_error.x - point[0]) ** 2 + (h_error.y - point[1]) ** 2) for point in self.points])
        arr = dist.argsort()[:1]

        self.prev_index1 = self.index1

        self.index1 = arr[0]
        self.index2 = self.get_index(arr[0] + 1)
        self.index3 = self.get_index(arr[0] + 2)
        self.index4 = self.get_index(arr[0] + 3)
        self.index5 = self.get_index(arr[0] + 4)
        self.index6 = self.get_index(arr[0] + 5)
        self.index7 = self.get_index(arr[0] + 6)
        self.index8 = self.get_index(arr[0] + 7)
        self.index9 = self.get_index(arr[0] + 8)
        self.index10 = self.get_index(arr[0] + 9)
        self.index11 = self.get_index(arr[0] + 10)
        self.index12 = self.get_index(arr[0] + 11)
        self.index13 = self.get_index(arr[0] + 12)
        self.index14 = self.get_index(arr[0] + 13)
        self.index15 = self.get_index(arr[0] + 14)
        self.index16 = self.get_index(arr[0] + 15)
        self.index17 = self.get_index(arr[0] + 16)
        self.index18 = self.get_index(arr[0] + 17)
        self.index19 = self.get_index(arr[0] + 18)
        self.index20 = self.get_index(arr[0] + 19)

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

        shell_matrix_orientation = physics.named.data.xmat["shell"]
        shell_matrix_orientation.resize(3, 3)
        matrix = shell_matrix_orientation[0:2, 0:2]
        matrix = matrix.T

        self.robot_position = [x, y]

        self.get_nearest_10_points_index()

        v_r1 = np.dot(matrix, vector(self.robot_position, self.points[self.index1]))
        v_r2 = np.dot(matrix, vector(self.robot_position, self.points[self.index2]))
        v_r3 = np.dot(matrix, vector(self.robot_position, self.points[self.index3]))
        v_r4 = np.dot(matrix, vector(self.robot_position, self.points[self.index4]))
        v_r5 = np.dot(matrix, vector(self.robot_position, self.points[self.index5]))
        v_r6 = np.dot(matrix, vector(self.robot_position, self.points[self.index6]))
        v_r7 = np.dot(matrix, vector(self.robot_position, self.points[self.index7]))
        v_r8 = np.dot(matrix, vector(self.robot_position, self.points[self.index8]))
        v_r9 = np.dot(matrix, vector(self.robot_position, self.points[self.index9]))
        v_r10 = np.dot(matrix, vector(self.robot_position, self.points[self.index10]))
        v_r11 = np.dot(matrix, vector(self.robot_position, self.points[self.index11]))
        v_r12 = np.dot(matrix, vector(self.robot_position, self.points[self.index12]))
        v_r13 = np.dot(matrix, vector(self.robot_position, self.points[self.index13]))
        v_r14 = np.dot(matrix, vector(self.robot_position, self.points[self.index14]))
        v_r15 = np.dot(matrix, vector(self.robot_position, self.points[self.index15]))
        v_r16 = np.dot(matrix, vector(self.robot_position, self.points[self.index16]))
        v_r17 = np.dot(matrix, vector(self.robot_position, self.points[self.index17]))
        v_r18 = np.dot(matrix, vector(self.robot_position, self.points[self.index18]))
        v_r19 = np.dot(matrix, vector(self.robot_position, self.points[self.index19]))
        v_r20 = np.dot(matrix, vector(self.robot_position, self.points[self.index20]))

        self.state = [v_r1[0], v_r1[1], #norm(v_r1),
                      v_r2[0], v_r2[1], #norm(v_r2),
                      v_r3[0], v_r3[1], #norm(v_r3),
                      v_r4[0], v_r4[1], #norm(v_r4),
                      v_r5[0], v_r5[1], #norm(v_r5),
                      v_r6[0], v_r6[1], #norm(v_r6),
                      v_r7[0], v_r7[1], #norm(v_r7),
                      v_r8[0], v_r8[1], #norm(v_r8),
                      v_r9[0], v_r9[1], #norm(v_r9),
                      v_r10[0], v_r10[1], #norm(v_r10),
                      v_r11[0], v_r11[1], #norm(v_r11),
                      v_r12[0], v_r12[1], #norm(v_r12),
                      v_r13[0], v_r13[1], #norm(v_r13),
                      v_r14[0], v_r14[1], #norm(v_r14),
                      v_r15[0], v_r15[1], #norm(v_r15),
                      v_r16[0], v_r16[1], #norm(v_r16),
                      v_r17[0], v_r17[1], #norm(v_r17),
                      v_r18[0], v_r18[1], #norm(v_r18),
                      v_r19[0], v_r19[1], #norm(v_r19),
                      v_r20[0], v_r20[1]] #norm(v_r20)]
        return self.state  # np.concatenate((xy, acc_gyro), axis=0)

    def get_termination(self, physics):
        if len(self.points) == 0 or physics.data.time > self.timeout or self.count_invalid_states >= 7\
                or len(self.points) == self.achievedPoints or self.count_hard_invalid_state >= 1:
            print("end episode at t = ", np.round(physics.data.time, 2), "\n")
            return 0.0

    def get_reward(self, physics):
        state = self.state

        x, y = self.robot_position
        point = geom.Point(x, y)
        h_error_dist = self.line.distance(point)

        if h_error_dist > 0.08:
            print("soft invalid state")
            self.count_invalid_states += 1
            return -1.0

        if self.is_invalid_state_hard():
            print("hard invalid state")
            self.count_hard_invalid_state += 1
            return -1.0

        if self.count_invalid_states > 0:
            print("вернулись на траекторию")
            self.count_invalid_states = 0

        reward = 1 - h_error_dist * 20

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
