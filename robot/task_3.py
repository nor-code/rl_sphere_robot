import numpy as np
from dm_control.suite import base


def is_belong_rectangle(x, y, a, b):
    return (-a <= x) and (x <= a) and (-b <= y) and (y <= b)


class TrakingTrajectoryTask3(base.Task):

    def __init__(self, points_function, type_curve, timeout, random=None):
        self.type_curve = type_curve

        """тайм-аут одного эпизода"""
        self.timeout = timeout
        """ количество точек, которые мы достигли в рамках текущего эпищода """
        self.achievedPoints = 0

        """ функции для целевой траектории"""
        self.p_fun = points_function
        self.points = points_function(type_curve)

        """текущая и предыдущая целевая точка ( координаты ) """
        self.current_point = self.points.popitem(last=False)

        self.prev_point = None

        """ общее количество точек на пути """
        self.totalPoint = len(self.points)

        """ количество неправильных состояний за эпизод """
        self.count_invalid_states = 0

        """предыдущее расстояние и текущее до целевой точки на кривой"""
        self.prev_dist = self.current_dist = 0

        """ точка невозврата """
        self.point_no_return = [0, 0]

        self.state = [0, 0, 0, 0]

        super().__init__(random=random)

    def initialize_episode(self, physics):
        physics.named.data.qpos[0:3] = [0, 0, 0.2]
        physics.named.data.qvel[:] = 0
        self.points = self.p_fun(self.type_curve)

        self.current_point = self.points.popitem(last=False)
        self.prev_point = None

        self.achievedPoints = 0

        self.count_invalid_states = 0

        self.prev_dist = self.current_dist = 0

        self.state = [0, 0, 0, 0]

        super().initialize_episode(physics)

    def get_observation(self, physics):
        # координаты центра колеса
        x, y, z = physics.named.data.geom_xpos['wheel_']
        # вектор скорости в абс системе координат
        v_x, v_y, v_z = physics.named.data.sensordata['sphere_vel']
        self.state = [x, y, v_x, v_y]
        return self.state  # np.concatenate((xy, acc_gyro), axis=0)

    def get_termination(self, physics):
        if len(self.points) == 0 or physics.data.time > self.timeout or self.count_invalid_states >= 1:
            return 0.0

    def __distance_to_current_point(self, x, y):
        return np.sqrt((x - self.current_point[0][0]) ** 2 + (y - self.current_point[0][1]) ** 2)

    @staticmethod
    def vector(pointA, pointB):
        """
        вектор AB = |pointA, pointB|
        """
        return [pointB[0][0] - pointA[0][0], pointB[0][1] - pointA[0][1]]

    def is_invalid_state(self):
        x, y, _, _ = self.state
        PC = TrakingTrajectoryTask3.vector(self.prev_point, self.current_point)

        x_center = self.prev_point[0][0] + (self.current_point[0][0] - self.prev_point[0][0]) / 2
        y_center = self.prev_point[0][1] + (self.current_point[0][1] - self.prev_point[0][1]) / 2

        len_PC = np.linalg.norm(PC)

        cos_a = np.dot(PC, [1, 0]) / len_PC
        a = np.arccos(cos_a)
        if np.dot(PC, [0, 1]) < 0:
            a = 2 * np.pi - a

        M = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])  # от Ox'y' -> Oxy

        x -= x_center
        y -= y_center

        x, y = np.dot(M, [x, y])

        if (not is_belong_rectangle(x, y, (len_PC / 2) + 0.05, 0.05)) or self.current_dist > self.prev_dist:
            return True

        return False

    def get_reward(self, physics):
        x, y, _, _ = self.state

        self.current_dist = self.__distance_to_current_point(x, y)

        if self.current_dist < 0.05:
            self.prev_point = self.current_point
            self.current_point = self.points.popitem(last=False)

            self.achievedPoints += 1

            self.prev_dist = self.current_dist = self.__distance_to_current_point(x, y)

            if len(self.points) == 0:
                print("FINAL")
                return 100
            return 10

        if self.is_invalid_state():
            self.count_invalid_states += 1
            return -40

        if self.achievedPoints > 1:
            print("count achieved points = ", self.achievedPoints, " time = ", physics.data.time)

        PC = TrakingTrajectoryTask3.vector(self.prev_point, self.current_point)

        reward = self.achievedPoints + np.linalg.norm(PC) / self.current_dist
        self.prev_dist = self.current_dist
        return reward
