import numpy as np
from numpy.linalg import norm
from dm_control.suite import base


def get_angle_between_2_vector(v1, v2, len_v1, len_v2):
    if len_v2 == 0.0 or len_v1 == 0.0:
        return 0.0
    cos = np.round(np.dot(v1, v2) / (len_v1 * len_v2), 4)
    if cos < 0:
        return np.pi - np.arccos(cos)
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
        self.index5 = self.get_index(begin_index + 2)

        self.prev_index1 = self.get_index(begin_index - 2)
        self.prev_index2 = self.get_index(begin_index - 1)
        self.prev_index3 = self.get_index(begin_index)
        self.prev_index4 = self.get_index(begin_index + 1)
        self.prev_index4 = self.get_index(begin_index + 2)

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

        self.get_nearest_5_points_index()
        self.no_return_index = self.index1
        self.dist = np.linalg.norm([x - self.points[self.no_return_index][0],
                                    y - self.points[self.no_return_index][1]])
        print("init no return point = ", self.no_return_index)

        physics.named.data.qpos[0:3] = [self.points[index][0], self.points[index][1], 0.2]
        physics.named.data.qvel[:] = 0

        self.count_invalid_states = 0
        self.count_hard_invalid_state = 0

        self.dist = 0.0

        v_r1 = vector(self.points[self.index1], self.robot_position)
        v_r2 = vector(self.points[self.index2], self.robot_position)
        v_r3 = vector(self.points[self.index3], self.robot_position)
        v_r4 = vector(self.points[self.index4], self.robot_position)
        v_r5 = vector(self.points[self.index5], self.robot_position)

        v_prev_r1 = vector(self.points[self.prev_index1], self.robot_position)
        v_prev_r2 = vector(self.points[self.prev_index2], self.robot_position)
        v_prev_r3 = vector(self.points[self.prev_index3], self.robot_position)
        v_prev_r4 = vector(self.points[self.prev_index4], self.robot_position)
        v_prev_r5 = vector(self.points[self.prev_index5], self.robot_position)

        v_i1_i2 = vector(self.points[self.index1], self.points[self.index2])
        v_i2_i3 = vector(self.points[self.index2], self.points[self.index3])
        v_i3_i4 = vector(self.points[self.index3], self.points[self.index4])
        v_i4_i5 = vector(self.points[self.index4], self.points[self.index5])

        self.state = [v_r1[0], v_r1[1], norm(v_r1),
                      v_r2[0], v_r2[1], norm(v_r2),
                      v_r3[0], v_r3[1], norm(v_r3),
                      v_r4[0], v_r4[1], norm(v_r4),
                      v_r5[0], v_r5[1], norm(v_r5),
                      v_i1_i2[0], v_i1_i2[1], v_i2_i3[0], v_i2_i3[1],
                      v_i3_i4[0], v_i3_i4[1], v_i4_i5[0], v_i4_i5[1]]
        super().initialize_episode(physics)

    def get_nearest_5_points_index(self):
        x, y = self.robot_position
        dist = np.array([np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2) for point in self.points])

        arr = dist.argsort()[:5]
        brr = arr - arr.max()
        indexes = np.where(brr <= -6)

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
        self.prev_index5 = self.index5

        self.index1 = arr[0]
        self.index2 = arr[1]
        self.index3 = arr[2]
        self.index4 = arr[3]
        self.index5 = arr[4]

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

        self.get_nearest_5_points_index()

        v_r1 = vector(self.points[self.index1], self.robot_position)
        v_r2 = vector(self.points[self.index2], self.robot_position)
        v_r3 = vector(self.points[self.index3], self.robot_position)
        v_r4 = vector(self.points[self.index4], self.robot_position)
        v_r5 = vector(self.points[self.index5], self.robot_position)

        v_prev_r1 = vector(self.points[self.prev_index1], self.robot_position)
        v_prev_r2 = vector(self.points[self.prev_index2], self.robot_position)
        v_prev_r3 = vector(self.points[self.prev_index3], self.robot_position)
        v_prev_r4 = vector(self.points[self.prev_index4], self.robot_position)
        v_prev_r5 = vector(self.points[self.prev_index5], self.robot_position)

        v_i1_i2 = vector(self.points[self.index1], self.points[self.index2])
        v_i2_i3 = vector(self.points[self.index2], self.points[self.index3])
        v_i3_i4 = vector(self.points[self.index3], self.points[self.index4])
        v_i4_i5 = vector(self.points[self.index4], self.points[self.index5])

        v_i2_i4 = vector(self.points[self.index2], self.points[self.index4])
        # 10            11            12            13            14            15            16            17            18            19
        #   v_prev_r1[0], v_prev_r1[1], v_prev_r2[0], v_prev_r2[1], v_prev_r3[0], v_prev_r3[1], v_prev_r4[0], v_prev_r4[1], v_prev_r5[0], v_prev_r5[1],

                      #0       1        2
        self.state = [v_r1[0], v_r1[1], norm(v_r1),
                      #3       4        5
                      v_r2[0], v_r2[1], norm(v_r2),
                      #6       7        8
                      v_r3[0], v_r3[1], norm(v_r3),
                      #9       10       11
                      v_r4[0], v_r4[1], norm(v_r4),
                      #12      13       14
                      v_r5[0], v_r5[1], norm(v_r5),
                      #15         16
                      v_i1_i2[0], v_i1_i2[1],
                      #17         18
                      v_i2_i3[0], v_i2_i3[1],
                      #19         20
                      v_i3_i4[0], v_i3_i4[1],
                      #21         22
                      v_i4_i5[0], v_i4_i5[1]]
        return self.state  # np.concatenate((xy, acc_gyro), axis=0)

    def get_termination(self, physics):
        if len(self.points) == 0 or physics.data.time > self.timeout or self.count_invalid_states >= 10 \
                or len(self.points) == self.achievedPoints or self.count_hard_invalid_state >= 1:
            print("end episode at t = ", np.round(physics.data.time, 2), "\n")
            return 0.0

    def get_reward_for_distance(self, a_i, r_i, sin_alpha_i):
        # if r_i > self.radius:
        #     return -15
        # else:
        if r_i < 0.01:
            return a_i * 100
        else:
            return a_i * (1 / r_i) * (1 - abs(sin_alpha_i))

    def get_reward(self, physics):
        state = self.state

        if self.is_invalid_state_soft():
            print("soft invalid state")
            self.count_invalid_states += 1
            return -85

        if self.is_invalid_state_hard():
            print("hard invalid state")
            self.count_hard_invalid_state += 1
            return -85

        if self.count_invalid_states > 0:
            print("вернулись на траекторию")
            self.count_invalid_states = 0

        v_r1 = [state[0], state[1]]
        v_r2 = [state[3], state[4]]
        v_r3 = [state[6], state[7]]
        v_r4 = [state[9], state[10]]
        v_r5 = [state[12], state[13]]

        r1, r2, r3, r4, r5 = state[2], state[5], state[8], state[11], state[14]
        i1_i2, i2_i3, i3_i4, i4_i5 = [state[15], state[16]], [state[17], state[18]], [state[19], state[20]], [state[21], state[22]]

        sin_a1 = np.sin(round(get_angle_between_2_vector(i1_i2, v_r1, norm(i1_i2), r1), 4))
        sin_a2 = np.sin(round(get_angle_between_2_vector(i2_i3, v_r2, norm(i2_i3), r2), 4))
        sin_a3 = np.sin(round(get_angle_between_2_vector(i3_i4, v_r3, norm(i3_i4), r3), 4))
        sin_a4 = np.sin(round(get_angle_between_2_vector(i4_i5, v_r4, norm(i4_i5), r4), 4))
        sin_a5 = np.sin(round(get_angle_between_2_vector([-state[21], -state[22]], v_r5, norm(i4_i5), r5), 4))

        reward1 = self.get_reward_for_distance(0.10, r1, 0.0)
        reward2 = self.get_reward_for_distance(0.25, r2, 0.0)
        reward3 = self.get_reward_for_distance(0.45, r3, 0.0)
        reward4 = self.get_reward_for_distance(0.85, r4, 0.0)
        reward5 = self.get_reward_for_distance(1.2, r5, 0.0)

        # h = np.min([sin_a1 * r1, sin_a2 * r2, sin_a3 * r3, sin_a4 * r4, sin_a5 * r5])

        # i2_i4 = [state[28], state[29]]
        # velocity = [state[30], state[31]]
        # cos_velocity_i2_i4 = get_angle_between_2_vector(velocity, i2_i4, norm(velocity), norm(i2_i4))

        # print("reward1 = ", round(reward1, 2), " r1 = ", r1)
        # print("reward2 = ", round(reward2, 2), " r2 = ", r2)
        # print("reward3 = ", round(reward3, 2), " r3 = ", r3)
        # print("reward4 = ", round(reward4, 2), " r4 = ", r4)
        # print("reward5 = ", round(reward5, 2), " r5 = ", r5)
        #
        # print(" discount1 = ", sin_a1)
        # print(" discount2 = ", sin_a2)
        # print(" discount3 = ", sin_a3)
        # print(" discount4 = ", sin_a4)
        # print(" discount5 = ", sin_a5)

        # reward = reward1 + reward2 + reward3 + reward4 + reward5
        # print("reward = ", reward, " discount = ", -150 * h, "distance = ", h)
        # reward -= 150 * h

        reward = reward1 + reward2 + reward3 + reward4 + reward5 \
                 - 10 * sin_a1 * r1 - 15 * sin_a2 * r2 - 35 * sin_a3 * r3 - 25 * sin_a4 * r4 - 25 * sin_a5 * r5

        return reward

    # если количество точек в окретности робота не осталось
    def is_invalid_state_soft(self):
        x, y = self.robot_position
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
        if self.no_return_index > self.index1: # or self.no_return_index == self.index2 or self.no_return_index == self.index3 or self.no_return_index == self.index4:
            return True
        return False
