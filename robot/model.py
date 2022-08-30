from dm_control import mujoco

class RobotPhysics(mujoco.Physics):
    def get_sphere_position(self):
        return self.named.data.geom_xpos['sphere_shell']