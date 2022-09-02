import numpy as np
from dm_control.suite import base


class TrakingTrajectoryTask1(base.Task):

    def __init__(self, points_function, timeout, random=None):
        """тайм-аут одного эпизода"""
        self.timeout = timeout
        """ количество точек, которые мы достигли в рамках текущего эпищода """
        self.achievedPoints = 0

        """ функции для целевой траектории"""
        self.p_fun = points_function
        self.points = points_function()

        """текущая и предыдущая целевая точка ( координаты ) """
        self.current_point = self.points.popitem(last=False)
        self.prev_point = None

        """ общее количество точек на пути """
        self.totalPoint = len(self.points)

        """текущее и предыдущее состояние ( координаты ) """
        self.current_xy = [0, 0]
        self.prev_xy = [0, 0]
        self.current_direction = [0, 0]
        self.prev_direction = [0, 0]

        self.state = [0, 0, 0, 0]

        super().__init__(random=random)

    def initialize_episode(self, physics):
        physics.named.data.qpos[0:3] = [0, 0, 0.2]
        physics.named.data.qvel[:] = 0
        self.points = self.p_fun()
        self.current_point = self.points.popitem(last=False)
        self.achievedPoints = 0
        self.prev_point = None

        self.current_xy = [0, 0]
        self.prev_xy = [0, 0]

        self.state = [0, 0, 0, 0]

        super().initialize_episode(physics)

    def get_observation(self, physics):
        if physics.data.time > 0:
            self.prev_xy = self.current_xy

        # координаты центра сферической оболочки
        x, y, z = physics.named.data.geom_xpos['sphere_shell']
        self.current_xy = [x, y]

        # угол поворота вилки колеса
        angle = physics.named.data.sensordata['fork_wheel_angle'] % 360 + 90

        # координаты напрвляющего единичного вектора курса робота
        sign_x = np.sign(x - self.prev_xy[0])
        sign_x = sign_x == 1 if sign_x == 0 else sign_x
        cos = sign_x * np.cos(np.deg2rad(angle))

        sign_y = np.sign(y - self.prev_xy[1])
        sign_y = sign_y == 1 if sign_y == 0 else sign_y
        sin = sign_y * np.sqrt(1 - cos ** 2)

        self.state = [x, y, cos[0], sin[0]]
        return self.state  # np.concatenate((xy, acc_gyro), axis=0)

    def get_termination(self, physics):
        x, y, z = physics.named.data.geom_xpos['sphere_shell']
        if len(self.points) == 0 or physics.data.time > self.timeout or self.__distance_to_current_point(x, y) > 1.2:
            return 0.0

    def __distance_to_current_point(self, x, y):
        return np.sqrt((x - self.current_point[0][0]) ** 2 + (y - self.current_point[0][1]) ** 2)

    @staticmethod
    def vector(pointA, pointB):
        """
        вектор AB = |pointA, pointB|
        """
        return [pointB[0][0] - pointA[0][0], pointB[0][1] - pointA[0][1]]

    def get_reward(self, physics):
        x, y, cos, sin = self.state
        distance = self.__distance_to_current_point(x, y)

        if distance < 0.1:
            self.prev_point = self.current_point
            self.achievedPoints += 1
            self.current_point = self.points.popitem(last=False)
            return 1

        if self.achievedPoints == 0:
            return -distance

        PC = TrakingTrajectoryTask1.vector(self.prev_point, self.current_point)
        PO = [x - self.prev_point[0][0], y - self.prev_point[0][1]]
        norm_PC = np.linalg.norm(PC)
        norm_PO = np.linalg.norm(PO)

        if norm_PO > norm_PC or np.linalg.norm(distance) > norm_PC \
                or self.__distance_to_current_point(self.prev_xy[0], self.prev_xy[1]) < distance:
            return -1

        cos_a = np.dot(PC, PO) / (norm_PO * norm_PC)
        return (norm_PC / (norm_PC - norm_PO * cos_a)) - np.sqrt(norm_PO ** 2 - (norm_PO * cos_a) ** 2) \
               - np.arccos(np.dot([PC[0] / norm_PC, PC[1] / norm_PC], [cos, sin]))
