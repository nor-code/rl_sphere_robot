import numpy as np
from numpy.linalg import norm
from dm_control.suite import base


def get_angle_between_2_vector(v1, v2, len_v1, len_v2):
    if len_v2 == 0.0 or len_v1 == 0.0:
        return 0.0
    cos = np.round(np.dot(v1, v2) / (len_v1 * len_v2), 4)
    return np.arccos(cos)


def vector(pointA, pointB):
    return [pointB[0] - pointA[0], pointB[1] - pointA[1]]

class TrakingTrajectoryTask7(base.Task):

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
        self.begin_index = begin_index
        self.no_return_index = self.get_index(begin_index - 2)

        self.index1 = self.get_index(begin_index - 2)
        self.index2 = self.get_index(begin_index - 1)
        self.index3 = self.get_index(begin_index)
        self.index4 = self.get_index(begin_index + 1)

        self.prev_index1 = self.get_index(begin_index - 2)
        self.prev_index2 = self.get_index(begin_index - 1)
        self.prev_index3 = self.get_index(begin_index)
        self.prev_index4 = self.get_index(begin_index + 1)

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

        self.get_nearest_4_points_index()
        self.no_return_index = self.index1
        self.dist = np.linalg.norm([x - self.points[self.no_return_index][0],
                                    y - self.points[self.no_return_index][1]])
        print("init no return point = ", self.no_return_index)

        physics.named.data.qpos[0:3] = [self.points[index][0], self.points[index][1], 0.2]
        physics.named.data.qvel[:] = 0

        self.count_invalid_states = 0
        self.count_hard_invalid_state = 0

        self.dist = 0.0

        r1 = norm([x - self.points[self.index1][0], y - self.points[self.index1][1]])
        r2 = norm([x - self.points[self.index2][0], y - self.points[self.index2][1]])
        r3 = norm([x - self.points[self.index3][0], y - self.points[self.index3][1]])
        r4 = norm([x - self.points[self.index4][0], y - self.points[self.index4][1]])

        prev_r1 = norm([x - self.points[self.prev_index1][0], y - self.points[self.prev_index1][1]])
        prev_r2 = norm([x - self.points[self.prev_index2][0], y - self.points[self.prev_index2][1]])
        prev_r3 = norm([x - self.points[self.prev_index3][0], y - self.points[self.prev_index3][1]])
        prev_r4 = norm([x - self.points[self.prev_index4][0], y - self.points[self.prev_index4][1]])

        self.state = [r1, r2, r3, r4, 0.0, 0.0, prev_r1, prev_r2, prev_r3, prev_r4, 0.0, 0.0]
        super().initialize_episode(physics)

    def get_nearest_4_points_index(self):
        x, y = self.robot_position
        dist = np.array([np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2) for point in self.points])

        arr = dist.argsort()[:4]
        brr = arr - arr.max()
        indexes = np.where(brr <= -5)

        if indexes[0].shape[0] != 0:
            index = np.where(arr == np.max(arr[indexes]))[0]
            left = arr[np.where(arr > arr[index])]
            left.sort()
            right = arr[np.where(arr <= arr[index])]
            right.sort()
            arr = np.concatenate((left, right))
        else:
            arr.sort()

        self.prev_index1 = self.index1
        self.prev_index2 = self.index2
        self.prev_index3 = self.index3
        self.prev_index4 = self.index4

        self.index1 = arr[0]
        self.index2 = arr[1]
        self.index3 = arr[2]
        self.index4 = arr[3]

        if (self.no_return_index == len(
                self.points) - 1 and self.prev_index1 == 0) or self.no_return_index < self.prev_index1:
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

        self.get_nearest_4_points_index()

        v_r1 = vector(self.points[self.index1], self.robot_position)
        v_r2 = vector(self.points[self.index2], self.robot_position)
        v_r3 = vector(self.points[self.index3], self.robot_position)
        v_r4 = vector(self.points[self.index4], self.robot_position)

        v_prev_r1 = vector(self.points[self.prev_index1], self.robot_position)
        v_prev_r2 = vector(self.points[self.prev_index2], self.robot_position)
        v_prev_r3 = vector(self.points[self.prev_index3], self.robot_position)
        v_prev_r4 = vector(self.points[self.prev_index4], self.robot_position)

        r1, r2, r3, r4 = norm(v_r1), norm(v_r2), norm(v_r3), norm(v_r4)
        prev_r1, prev_r2, prev_r3, prev_r4 = norm(v_prev_r1), norm(v_prev_r2), norm(v_prev_r3), norm(v_prev_r4)

        v_i1_i2 = vector(self.points[self.index2], self.points[self.index1])
        v_i2_i3 = vector(self.points[self.index3], self.points[self.index2])
        sin_alpha1 = np.sin(round(get_angle_between_2_vector(v_i1_i2, v_r1, norm(v_i1_i2), r1), 4))
        sin_alpha2 = np.sin(round(get_angle_between_2_vector(v_i2_i3, v_r2, norm(v_i2_i3), r2), 4))

        v_prev_i1_i2 = vector(self.points[self.prev_index2], self.points[self.prev_index1])
        v_prev_i2_i3 = vector(self.points[self.prev_index3], self.points[self.prev_index2])
        sin_prev_alpha1 = np.sin(round(get_angle_between_2_vector(v_prev_i1_i2, v_prev_r1, norm(v_prev_i1_i2), prev_r1), 4))
        sin_prev_alpha2 = np.sin(round(get_angle_between_2_vector(v_prev_i2_i3, v_prev_r2, norm(v_prev_i2_i3), prev_r2), 4))
                     #0   1   2   3   4           5           6        7        8        9        10               11
        self.state = [r1, r2, r3, r4, sin_alpha1, sin_alpha2, prev_r1, prev_r2, prev_r3, prev_r4, sin_prev_alpha1, sin_prev_alpha2]
        return self.state  # np.concatenate((xy, acc_gyro), axis=0)

    def get_termination(self, physics):
        if len(self.points) == 0 or physics.data.time > self.timeout or self.count_invalid_states >= 7 \
                or len(self.points) == self.achievedPoints or self.count_hard_invalid_state >= 1:
            print("end episode at t = ", np.round(physics.data.time, 2), "\n")
            return 0.0

    def get_reward_for_distance(self, a_i, r_i):
        if r_i > self.radius:
            return -10
        else:
            if r_i < 0.01:
                return a_i * 100
            else:
                return a_i * (1 / r_i)

    def get_reward(self, physics):
        state = self.state
        x, y = self.robot_position

        if self.is_invalid_state_soft():
            self.count_invalid_states += 1
            return -50

        if self.is_invalid_state_hard():
            self.count_hard_invalid_state += 1
            return -50

        if self.count_invalid_states > 0:
            print("вернулись на траекторию")
            self.count_invalid_states = 0

        r1, r2, r3, r4 = state[0], state[1], state[2], state[3]
        prev_r1, prev_r2 = state[6], state[7]

        distance_reward1 = self.get_reward_for_distance(0.10, r1)
        distance_reward2 = self.get_reward_for_distance(0.15, r2)
        distance_reward3 = self.get_reward_for_distance(0.55, r3)
        distance_reward4 = self.get_reward_for_distance(0.85, r4)

        reward = distance_reward1 + distance_reward2 + distance_reward3 + distance_reward4 - 20 * state[4] * r1 - 30 * state[5]
        return reward

    # если количество точек в окретности робота не осталось
    def is_invalid_state_soft(self):
        state = self.state
        x = state[0]
        y = state[1]
        for point in self.points:
            if np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2) <= self.radius:
                return False
        return True

    # TODO check. если робот повернул назад, обычно бывает из-за резкого поворота назад
    def is_invalid_state_hard(self):
        x, y = self.robot_position
        new_dist = np.linalg.norm([x - self.points[self.no_return_index][0], y - self.points[self.no_return_index][1]])
        # if new_dist <= self.dist:
        #     print("new dist = ", round(new_dist, 4), " no return dist = ", round(self.dist, 4))
        #     return True
        # поскольку индекс невозврата неуменьшается (всегда),
        # то эта проверка - еще один способ проверить не свернул ли робот назад
        if self.no_return_index == len(self.points) - 1 and self.index1 == 0:
            return False
        if self.no_return_index > self.index1 or self.no_return_index == self.index2 \
                or self.no_return_index == self.index3 or self.no_return_index == self.index4:
            return True
        return False
