import numpy as np
from dm_control.suite import base


class TrakingTrajectoryTask5(base.Task):

    def __init__(self, trajectory_x_y, begin_index, timeout, R=0.238, random=None):
        """тайм-аут одного эпизода"""
        self.timeout = timeout
        """ количество точек, которые мы достигли в рамках текущего эпищода """
        self.achievedPoints = 0

        """ целевая траектория и начальная точка на ней """
        self.point_x_y = trajectory_x_y
        self.points = np.copy(self.point_x_y, order='K')
        self.begin_index = begin_index

        self.curr_indexes = []
        self.prev_indexes = []
        self.curr_dist = []
        self.prev_dist = []
        self.curr_v = []
        self.prev_v = []
        self.radius = R

        """текущая и предыдущая целевая точка ( координаты ) """
        self.robot_position = [self.points[begin_index][0], self.points[begin_index][1]]

        """ количество неправильных состояний за эпизод """
        self.count_invalid_states = 0

        self.state = []

        super().__init__(random=random)

    def get_index(self, index):
        if index < 0:
            return len(self.points) + index
        if index > len(self.points) - 1:
            return index % len(self.points)
        return index

    def init_8_indexes(self, begin_index):
        return [
            self.get_index(begin_index - 2), self.get_index(begin_index - 1), self.get_index(begin_index), self.get_index(begin_index + 1),

            self.get_index(begin_index + 2), self.get_index(begin_index + 3), self.get_index(begin_index + 4), self.get_index(begin_index + 5)
        ]

    def set_8_prev_indexes(self):
        self.prev_indexes = self.curr_indexes.copy()

    def set_8_curr_indexes(self, i1, i2, i3, i4):
        self.curr_indexes = [
            self.get_index(i1), self.get_index(i2), self.get_index(i3), self.get_index(i4),

            self.get_index(i4 + 1), self.get_index(i4 + 2), self.get_index(i4 + 3), self.get_index(i4 + 4)
        ]

    def calculate_8_curr_dist(self, x, y):
        r1 = np.linalg.norm([x - self.points[self.curr_indexes[0]][0], y - self.points[self.curr_indexes[0]][1]])
        r2 = np.linalg.norm([x - self.points[self.curr_indexes[1]][0], y - self.points[self.curr_indexes[1]][1]])
        r3 = np.linalg.norm([x - self.points[self.curr_indexes[2]][0], y - self.points[self.curr_indexes[2]][1]])
        r4 = np.linalg.norm([x - self.points[self.curr_indexes[3]][0], y - self.points[self.curr_indexes[3]][1]])
        r5 = np.linalg.norm([x - self.points[self.curr_indexes[4]][0], y - self.points[self.curr_indexes[4]][1]])
        r6 = np.linalg.norm([x - self.points[self.curr_indexes[5]][0], y - self.points[self.curr_indexes[5]][1]])
        r7 = np.linalg.norm([x - self.points[self.curr_indexes[6]][0], y - self.points[self.curr_indexes[6]][1]])
        r8 = np.linalg.norm([x - self.points[self.curr_indexes[7]][0], y - self.points[self.curr_indexes[7]][1]])
        return [r1, r2, r3, r4, r5, r6, r7, r8]

    def get_missed_indexes(self, prev_i, curr_i):
        a = np.setdiff1d(range(prev_i[0], prev_i[-1]), prev_i).tolist()
        diff = curr_i[0] - prev_i[-1]
        b = list(range(prev_i[-1] + 1, curr_i[0])) if 1 < diff < 20 else []
        c = np.setdiff1d(range(curr_i[0], curr_i[-1]), curr_i).tolist()
        return a + b + c

    def initialize_episode(self, physics):
        self.points = np.copy(self.point_x_y, order='K')

        print("init index on trajectory = ", self.begin_index)
        index = self.begin_index

        x = self.robot_position[0]
        y = self.robot_position[1]

        self.curr_indexes = self.init_8_indexes(index)
        self.prev_indexes = self.curr_indexes.copy()
        self.curr_dist = self.calculate_8_curr_dist(x, y)
        self.prev_dist = self.curr_dist.copy()

        physics.named.data.qpos[0:3] = [self.points[index][0], self.points[index][1], 0.2]
        physics.named.data.qvel[:] = 0
        self.curr_v = [0.0, 0.0]
        self.prev_v = [0.0, 0.0]

        self.count_invalid_states = 0

        self.state = self.curr_v + self.curr_dist + self.prev_v + self.prev_dist
        # v_x, v_y,                                # 0 1
        # r1, r2, r3, r4,                          # 2 3 4 5
        # r5, r6, r7, r8,                          # 6 7 8 9
        # prev_v_x, prev_v_y,                      # 10 11
        # prev_r1, prev_r2, prev_r3, prev_r4,      # 12 13 14 15
        # prev_r5, prev_r6, prev_r7, prev_r8       # 16 17 18 19

        super().initialize_episode(physics)

    def current_first_index_less_than_prev_first_index(self):
        if self.curr_indexes[0] >= 0 and len(self.points) - 1 == self.prev_indexes[0]:
            return False
        return self.curr_indexes[0] < self.prev_indexes[0]  # получается что повернули обратно

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

        self.set_8_prev_indexes()
        self.set_8_curr_indexes(arr[0], arr[1], arr[2], arr[3])

    def get_observation(self, physics):
        x, y, z = physics.named.data.geom_xpos['wheel_']         # координаты центра колеса

        self.prev_v = self.curr_v.copy()
        v_x, v_y, _ = physics.named.data.sensordata['wheel_vel']  # вектор скорости в абс системе координат
        self.curr_v = [v_x, v_y]

        self.robot_position = [x, y]
        self.get_nearest_4_points_index()

        self.prev_dist = self.curr_dist.copy()
        self.curr_dist = self.calculate_8_curr_dist(x, y)

        self.state = self.curr_v + self.curr_dist + self.prev_v + self.prev_dist

        return self.state

    def get_termination(self, physics):
        if len(self.points) == 0 or physics.data.time > self.timeout or self.count_invalid_states >= 15:
            print("end episode at t = ", np.round(physics.data.time, 2))
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

        if self.is_invalid_state():
            self.count_invalid_states += 1
            return -50

        if self.count_invalid_states > 0:
            print("вернулись на траекторию")
            self.count_invalid_states = 0

        prev_i = [self.prev_indexes[0], self.prev_indexes[1], self.prev_indexes[2], self.prev_indexes[3]]
        curr_i = [self.curr_indexes[0], self.curr_indexes[1], self.curr_indexes[2], self.curr_indexes[3]]
        missed = set(self.get_missed_indexes(prev_i, curr_i))

        print("\nprev_i: ", prev_i)
        print("curr_i: ", curr_i)
        print("missed: ", missed)

        r1 = state[2]
        r2 = state[3]
        r3 = state[4]
        r4 = state[5]

        distance_reward1 = self.get_reward_for_distance(0.05, r1)
        distance_reward2 = self.get_reward_for_distance(0.15, r2)
        distance_reward3 = self.get_reward_for_distance(0.45, r3)
        distance_reward4 = self.get_reward_for_distance(1.2, r4)

        reward = distance_reward1 + distance_reward2 + distance_reward3 + distance_reward4 - 10 * len(missed)

        return reward

    # если количество точек в окретности робота не осталось, т.е. робот ушел с траектории, либо робот повернул назад
    def is_invalid_state(self):
        x, y = self.robot_position
        for point in self.points:
            if np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2) <= self.radius:
                return False
        if self.current_first_index_less_than_prev_first_index():
            return True
        return True
