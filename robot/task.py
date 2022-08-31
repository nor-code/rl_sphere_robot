import numpy as np
from dm_control.suite import base


class TrakingTrajectoryTask(base.Task):

    def __init__(self, points_function, random=None):
        self.timeout = 12
        self.p_fun = points_function
        self.points = points_function()
        self.current_point = self.points.popitem(last=False)
        """ количество точек, которые мы достигли в рамках текущего эпищода """
        self.achievedPoints = 0
        """ общее количество точек на пути """
        self.totalPoint = len(self.points)

        self.prev_point = None
        super().__init__(random=random)

    def initialize_episode(self, physics):
        physics.named.data.qpos[0:3] = [0, 0, 0.2]
        physics.named.data.qvel[:] = 0
        self.points = self.p_fun()
        self.current_point = self.points.popitem(last=False)
        self.achievedPoints = 0
        self.prev_point = None
        super().initialize_episode(physics)

    def get_observation(self, physics):
        xy = physics.named.data.geom_xpos['sphere_shell'][0:2]
        # acc_gyro = physics.data.sensordata
        return xy  # np.concatenate((xy, acc_gyro), axis=0)

    def get_termination(self, physics):
        x, y, z = physics.named.data.geom_xpos['sphere_shell']
        if len(self.points) == 0 or physics.data.time > self.timeout or self.__distance_to_current_point(x, y) > 1.2 :
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
        x, y, z = physics.named.data.geom_xpos['sphere_shell']
        distance = self.__distance_to_current_point(x, y)

        if distance < 0.1:
            self.prev_point = self.current_point
            self.achievedPoints += 1
            self.current_point = self.points.popitem(last=False)
            return 1

        if self.achievedPoints == 0:
            return -distance

        PC = TrakingTrajectoryTask.vector(self.prev_point, self.current_point)
        PO = [x - self.prev_point[0][0], y - self.prev_point[0][1]]
        norm_PC = np.linalg.norm(PC)
        norm_PO = np.linalg.norm(PO)

        if norm_PC < norm_PO:
            return -1

        cos_a = np.dot(PC, PO) / (norm_PO * norm_PC)
        return cos_a * distance
