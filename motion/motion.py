from dm_control import mujoco

# # Access to enums and MuJoCo library functions.
# from dm_control.mujoco.wrapper.mjbindings import enums
# from dm_control.mujoco.wrapper.mjbindings import mjlib
#
# # PyMJCF
# from dm_control import mjcf
#
# # Composer high level imports
# from dm_control import composer
# from dm_control.composer.observation import observable
# from dm_control.composer import variation
#
# # Imports for Composer tutorial example
# from dm_control.composer.variation import distributions
# from dm_control.composer.variation import noises
# from dm_control.locomotion.arenas import floors
#
# # Control Suite
from dm_control import suite
#
# # Run through corridor example
# from dm_control.locomotion.walkers import cmu_humanoid
# from dm_control.locomotion.arenas import corridors as corridor_arenas
# from dm_control.locomotion.tasks import corridors as corridor_tasks
#
# # Soccer
# from dm_control.locomotion import soccer
#
# # Manipulation
# from dm_control import manipulation

tippe_top = """
<mujoco model="tippe top">
  <option integrator="RK4"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" 
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>
  <worldbody>
    <geom size=".2 .2 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <body name="top" pos="0 0 .02">
      <freejoint/>
      <geom name="ball" type="sphere" size=".02" />
      <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008"/>
      <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015" 
       contype="0" conaffinity="0" group="3"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 200" />
  </keyframe>
</mujoco>
"""
physics = mujoco.Physics.from_xml_string(tippe_top)

duration = 7    # (seconds)
framerate = 60  # (Hz)

# Simulate and display video.
frames = []
physics.reset(0)  # Reset to keyframe 0 (load a saved state).
while physics.data.time < duration:
  physics.step()
  if len(frames) < (physics.data.time) * framerate:
    pixels = physics.render(camera_id='closeup')
    frames.append(pixels)

