import numpy as np
from dm_control.suite import base

e_x = [1, 0]
e_y = [0, 1]


class TrakingTrajectoryTask2(base.Task):

    def __init__(self, trajectory_function, begin_index, timeout, random=None):

        """тайм-аут одного эпизода"""
        self.timeout = timeout
        """ количество точек, которые мы достигли в рамках текущего эпищода """
        self.achievedPoints = 0

        """ функции для целевой траектории"""
        self.p_fun = trajectory_function
        self.points = np.array(self.p_fun()).T
        self.begin_index = begin_index
        self.current_index = begin_index

        """текущая и предыдущая целевая точка ( координаты ) """
        self.current_point = self.points[begin_index]
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
        self.points = np.array(self.p_fun()).T

        index = self.begin_index
        self.current_index = index
        self.current_point = self.points[index]
        self.prev_point = None

        physics.named.data.qpos[0:3] = [self.current_point[0], self.current_point[1], 0.2]
        physics.named.data.qvel[:] = 0

        self.achievedPoints = 0

        self.count_invalid_states = 0

        self.prev_dist = self.current_dist = 0

        self.state = [0, 0]

        super().initialize_episode(physics)

    def get_observation(self, physics):
        # координаты центра колеса
        x, y, z = physics.named.data.geom_xpos['wheel_']

        self.state = [x, y]
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

    # знак для прямой, брать > 0 или < 0
    @staticmethod
    def get_sign(vector):
        on_x = np.dot(vector, e_x)
        on_y = np.dot(vector, e_y)

        if on_x * on_y > 0:
            return 1
        return -1

    # робот лежит в пределах окружности
    @staticmethod
    def is_belong_ellipse(P, C, radius, x, y):
        x_center = P[0] + (C[0] - P[0]) / 2
        y_center = P[1] + (C[1] - P[1]) / 2
        return (x - x_center)**2 + (y - y_center)**2 <= radius**2

    def is_invalid_state(self):
        x, y = self.state
        PC = TrakingTrajectoryTask2.vector(self.prev_point, self.current_point)
        radius = np.linalg.norm(PC) / 2

        if not TrakingTrajectoryTask2.is_belong_ellipse(self.prev_point[0], self.current_point[0], radius + 0.056, x, y):
            return True

        if self.current_dist > self.prev_dist:
            return True
        return False

    def get_reward(self, physics):
        x, y = self.state

        self.current_dist = self.__distance_to_current_point(x, y)

        if self.current_dist < 0.056:
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
            return -50 * self.achievedPoints

        if self.achievedPoints > 1:
            print("count achived points = ", self.achievedPoints, " time = ", physics.data.time)

        PC = TrakingTrajectoryTask2.vector(self.prev_point, self.current_point)

        self.point_no_return = [x, y]

        reward = self.achievedPoints + np.linalg.norm(PC) / self.current_dist
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
