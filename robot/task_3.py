import numpy as np
from dm_control.suite import base


def is_belong_rectangle(x, y, a, b):
    return (-a - 0.1 <= x) and (x <= a) and (-b <= y) and (y <= b)


class TrakingTrajectoryTask3(base.Task):

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
        self.prev_index = None
        self.curr_dist = []

        self.radius = R

        """ координаты робота """
        self.robot_position = [self.points[begin_index][0], self.points[begin_index][1]]

        """текущая и предыдущая целевая точка на траектории ( координаты ) """
        self.current_point = self.points[begin_index]
        self.current_index = begin_index
        self.prev_point = None

        """ общее количество точек на пути """
        self.totalPoint = len(self.points)

        """ количество неправильных состояний за эпизод """
        self.count_invalid_states = 0

        """предыдущее расстояние и текущее до целевой точки на кривой"""
        self.prev_dist = self.current_dist = 0

        self.state = []

        super().__init__(random=random)

    def initialize_episode(self, physics):
        self.points = np.copy(self.point_x_y, order='K')

        index = self.begin_index

        x, y = self.robot_position[0], self.robot_position[1]

        self.set_16_curr_indexes(index)
        self.prev_index = self.curr_indexes[0]
        self.curr_dist = self.calculate_16_curr_dist(x, y)

        self.current_index = index
        self.current_point = self.points[index]
        self.prev_point = None

        physics.named.data.qpos[0:3] = [self.current_point[0], self.current_point[1], 0.2]
        physics.named.data.qvel[:] = 0

        self.achievedPoints = 0

        self.count_invalid_states = 0

        self.prev_dist = self.current_dist = 0

        self.state = self.curr_dist.copy()

        super().initialize_episode(physics)

    def get_observation(self, physics):
        # координаты центра колеса
        x, y, z = physics.named.data.geom_xpos['wheel_']
        # вектор скорости в абс системе координат
        # v_x, v_y, _ = physics.named.data.sensordata['wheel_vel']
        # угловая скорость сферической оболочки
        # w_x, w_y, w_z = physics.named.data.sensordata['sphere_angular_vel']
        # отклонение от траектории
        # h_error = 0.0 if self.begin_index == self.current_index else self.get_h_error(x, y)

        self.robot_position = [x, y]

        self.state = self.calculate_16_curr_dist(x, y)
        return self.state

    def get_termination(self, physics):
        if len(self.points) == 0 or physics.data.time > self.timeout or self.count_invalid_states >= 1 \
                or len(self.points) == self.achievedPoints:
            print("count achieved points = ", self.achievedPoints,
                  " time = ", round(physics.data.time, 2),
                  " init index of point = ", self.begin_index)
            return 0.0

    def __distance_to_current_point(self, x, y):
        return np.sqrt((x - self.current_point[0]) ** 2 + (y - self.current_point[1]) ** 2)

    @staticmethod
    def vector(pointA, pointB):
        """
        вектор AB = |pointA, pointB|
        """
        return [pointB[0] - pointA[0], pointB[1] - pointA[1]]

    def get_angle_y(self, PC, len_PC):
        cos_a = np.round(np.dot(PC, [1, 0]) / len_PC, 6)
        a = np.arccos(cos_a)
        if np.dot(PC, [0, 1]) < 0:
            a = 2 * np.pi - a
        return a

    def get_angle_between_2_vector(self, v1, v2, len_v1, len_v2):
        cos = np.round(np.dot(v1, v2) / (len_v1 * len_v2), 5)
        return np.arccos(cos)

    def get_h_error(self, x, y):
        PO = TrakingTrajectoryTask3.vector(self.prev_point, np.array([x, y]))
        PC = TrakingTrajectoryTask3.vector(self.prev_point, self.current_point)
        len_PO = np.linalg.norm(PO)
        len_PC = np.linalg.norm(PC)

        angle_PO_PC = self.get_angle_between_2_vector(PO, PC, len_PC, len_PO)
        return len_PO * np.sin(angle_PO_PC)

    def is_invalid_state(self):
        x, y = self.robot_position[0], self.robot_position[1]
        PC = TrakingTrajectoryTask3.vector(self.prev_point, self.current_point)

        x_center = self.prev_point[0] + (self.current_point[0] - self.prev_point[0]) / 2
        y_center = self.prev_point[1] + (self.current_point[1] - self.prev_point[1]) / 2

        len_PC = np.linalg.norm(PC)
        a = self.get_angle_y(PC, len_PC)

        M = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])  # от Ox'y' -> Oxy

        x -= x_center
        y -= y_center

        x, y = np.dot(M, [x, y])

        if (not is_belong_rectangle(x, y, (len_PC / 2), 0.1)) or self.current_dist > self.prev_dist:
            return True

        return False

    def get_reward(self, physics):
        x, y = self.robot_position

        self.current_dist = self.__distance_to_current_point(x, y)

        if self.current_dist < 0.1:
            self.prev_point = self.current_point
            self.current_point = self.get_next_point(self.points)

            self.set_16_curr_indexes(self.current_index)

            self.achievedPoints += 1
            if len(self.points) == self.achievedPoints:
                print("FINAL. init index of point = ", self.begin_index)
                return 1

            self.prev_dist = self.current_dist = self.__distance_to_current_point(x, y)
            return 1

        if self.is_invalid_state():
            self.count_invalid_states += 1
            return -1.5

        h_error = self.get_h_error(x, y)
        h_error_norm = (0.1 - h_error) / 0.1

        reward = 1 - h_error_norm
        self.prev_dist = self.current_dist
        return reward

    """
    следующая точка и дошли ли мы до конца или нет
    """

    def get_next_point(self, points):
        self.current_index += 1
        self.current_index %= len(points)
        if self.current_index == self.begin_index:
            return None
        return points[self.current_index]

    def get_index(self, index):
        if index < 0:
            return len(self.points) + index
        if index > len(self.points) - 1:
            return index % len(self.points)
        return index

    def set_16_curr_indexes(self, i1):
        self.curr_indexes = [
            self.get_index(i1), self.get_index(i1 + 1), self.get_index(i1 + 2), self.get_index(i1 + 3),
            self.get_index(i1 + 4), self.get_index(i1 + 5), self.get_index(i1 + 6), self.get_index(i1 + 7),
            self.get_index(i1 + 8), self.get_index(i1 + 9), self.get_index(i1 + 10), self.get_index(i1 + 11),
            self.get_index(i1 + 12), self.get_index(i1 + 13), self.get_index(i1 + 14), self.get_index(i1 + 15)
        ]

    def calculate_16_curr_dist(self, x, y):
        r = np.zeros(16)
        for i in range(16):
            index = self.curr_indexes[i]
            r[i] = np.linalg.norm([x - self.points[index][0], y - self.points[index][1]])
        return r.tolist()
