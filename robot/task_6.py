import numpy as np
from dm_control.suite import base


class TrakingTrajectoryTask6(base.Task):

    def __init__(self, trajectory_x_y, begin_index, timeout, R=0.238, random=None):
        """тайм-аут одного эпизода"""
        self.timeout = timeout

        """ целевая траектория и начальная точка на ней """
        self.point_x_y = trajectory_x_y
        self.points = np.copy(self.point_x_y, order='K')
        self.begin_index = begin_index

        self.curr_indexes = []
        self.prev_index = None
        self.curr_dist = []

        self.radius = R

        """ координаты робота """
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

    def init_16_indexes(self, begin_index):
        return [
            self.get_index(begin_index - 2), self.get_index(begin_index - 1), self.get_index(begin_index), self.get_index(begin_index + 1),
            self.get_index(begin_index + 2), self.get_index(begin_index + 3), self.get_index(begin_index + 4), self.get_index(begin_index + 5),
            self.get_index(begin_index + 6), self.get_index(begin_index + 7), self.get_index(begin_index + 8), self.get_index(begin_index + 9),
            self.get_index(begin_index + 10), self.get_index(begin_index + 11), self.get_index(begin_index + 12), self.get_index(begin_index + 13)
        ]

    def set_16_curr_indexes(self, i1, i2, i3, i4):
        self.curr_indexes = [
            self.get_index(i1),     self.get_index(i2),      self.get_index(i3),      self.get_index(i4),
            self.get_index(i4 + 1), self.get_index(i4 + 2),  self.get_index(i4 + 3),  self.get_index(i4 + 4),
            self.get_index(i4 + 5), self.get_index(i4 + 6),  self.get_index(i4 + 7),  self.get_index(i4 + 8),
            self.get_index(i4 + 9), self.get_index(i4 + 10), self.get_index(i4 + 11), self.get_index(i4 + 12)
        ]

    def calculate_16_curr_dist(self, x, y):
        r = np.zeros(16)
        for i in range(16-1):
            index = self.curr_indexes[i]
            r[i] = np.linalg.norm([x - self.points[index][0], y - self.points[index][1]])
        return r.tolist()

    def initialize_episode(self, physics):
        self.points = np.copy(self.point_x_y, order='K')

        print("init index on trajectory = ", self.begin_index)
        index = self.begin_index

        x = self.robot_position[0]
        y = self.robot_position[1]

        self.curr_indexes = self.init_16_indexes(index)
        self.prev_index = self.curr_indexes[0].copy()
        self.curr_dist = self.calculate_16_curr_dist(x, y)

        physics.named.data.qpos[0:3] = [self.points[index][0], self.points[index][1], 0.2]
        physics.named.data.qvel[:] = 0

        self.count_invalid_states = 0

        self.state = self.curr_dist  # 0, 1, 2, 3

        super().initialize_episode(physics)

    def current_first_index_less_than_prev_first_index(self):
        if self.curr_indexes[0] >= 0 and len(self.points) - 1 == self.prev_index:
            return False
        return self.curr_indexes[0] < self.prev_index  # получается что повернули обратно

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

        self.prev_index = self.curr_indexes[0]
        self.set_16_curr_indexes(arr[0], arr[1], arr[2], arr[3])

    def get_observation(self, physics):
        x, y, z = physics.named.data.geom_xpos['wheel_']         # координаты центра колеса

        self.robot_position = [x, y]
        self.get_nearest_4_points_index()

        self.curr_dist = self.calculate_16_curr_dist(x, y)

        self.state = self.curr_dist.copy()

        return self.state

    def get_termination(self, physics):
        if len(self.points) == 0 or physics.data.time > self.timeout or self.count_invalid_states >= 10:
            print("end episode at t = ", np.round(physics.data.time, 2))
            return 0.0

    def get_reward(self, physics):

        if self.is_invalid_state():
            self.count_invalid_states += 1
            return -1

        if self.count_invalid_states > 0:
            print("вернулись на траекторию")
            self.count_invalid_states = 0

        x, y = self.robot_position[0], self.robot_position[1]

        h1 = self.get_h(x, y, self.curr_indexes[0], self.curr_indexes[1])
        h2 = self.get_h(x, y, self.curr_indexes[1], self.curr_indexes[2])
        h3 = self.get_h(x, y, self.curr_indexes[2], self.curr_indexes[3])

        distance_to_path = min([h1, h2, h3])

        if distance_to_path > self.radius:
            return -1

        return self.radius - distance_to_path

    def get_h(self, x, y, i1, i2):
        P1P2 = TrakingTrajectoryTask6.vector(self.points[i1], self.points[i2])
        P1O = TrakingTrajectoryTask6.vector(self.points[i1], [x, y])

        len_P1P2 = np.linalg.norm(P1P2)
        len_P1O = np.linalg.norm(P1O)

        cos = np.round(np.dot(P1P2, P1O) / (len_P1P2 * len_P1O), 3)
        alpha = np.arccos(cos)
        return len_P1O * np.sin(alpha)

    @staticmethod
    def vector(pointA, pointB):
        """
        вектор AB = |pointA, pointB|
        """
        return [pointB[0] - pointA[0], pointB[1] - pointA[1]]

    # если количество точек в окретности робота не осталось, т.е. робот ушел с траектории, либо робот повернул назад
    def is_invalid_state(self):
        x, y = self.robot_position
        for point in self.points:
            if np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2) <= self.radius:
                return False
        if self.current_first_index_less_than_prev_first_index():
            return True
        return True
