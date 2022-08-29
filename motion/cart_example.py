"""
Example of how bodies interact with each other. For a body to be able to
move it needs to have joints. In this example, the "robot" is a red ball
with X and Y slide joints (and a Z slide joint that isn't controlled).
On the floor, there's a cylinder with X and Y slide joints, so it can
be pushed around with the robot. There's also a box without joints. Since
the box doesn't have joints, it's fixed and can't be pushed around.
"""
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os
import numpy as np

# Environment parameters
gravity = -9.81

#Model Car Dimensional Parameters (in units m, sec, kg,)
body_x = 0.28
body_y = 0.20
body_z = 0.003
upright_thickness = 0.01
upright_spacing = 0.01
upright_edge_length = 0.04
wheel_diameter = 0.08
wheel_thickness = 0.01
wheel_base = 0.28

# fake coordinate systems
axis_diam = 0.01
axis_length = 0.1
# Helpful pre-written Quaternion values for rotataions



MODEL_XML = f"""
<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.005" gravity="0 0 {gravity}">
        <flag  gravity="enable"/>
    </option>
    <worldbody>
        <body name="xaxis" pos = "1 1 1" quat="1 1 0 0">
            <geom mass="1.0" pos="0 0 0" rgba="1 0 0 0.5" size="0.1 1" type="cylinder"/>
        </body>
        <body name="yaxis" pos = "1 1 1" quat="1 0 1 0">
            <geom mass="1.0" pos="0 0 0" rgba="0 1 0 0.5" size="0.1 1" type="cylinder"/>
        </body>
        <body name="zaxis" pos = "1 1 1" quat = "1 0 0 1">
            <geom mass="1.0" pos="0 0 0" rgba="0 0 1 0.5" size="0.1 1" type="cylinder"/>
        </body>

        <body name="robot" pos="0 0 0.2" quat="1 0 0 0">
            <joint name="slide0" pos="0 0 0" type="free"/>
            <geom mass="10.0" pos="0 0 0" rgba="1 0 0 1" size="{body_x} {body_y} {body_z}" type="box"/>
			<camera euler="0 0 0" fovy="40" name="rgb" pos="0 0 1.5"></camera>

            <body name="upright0" pos = "{body_x} {(body_y) + upright_spacing} 0" quat="1 -1 0 0">
                <geom mass="1.0" pos="0 0 0" rgba="0 1 0 1" size="{upright_edge_length} {upright_edge_length} {upright_thickness}" type="box"/>
                <joint axis="0 1 0" damping="0.0" name="turn0" pos="0 0 0" type="hinge"/>

                <body name="wheel0" pos="0 0 {(wheel_thickness) + upright_thickness}" quat="1 0 0 0">
                    <joint axis="0 0 1" damping="0.0" name="freespin0" pos="0 0 0" type="hinge"/>
                    <geom mass="1.0" pos="0 0 0" rgba="1 0 0 0.5" size="{wheel_diameter} {wheel_thickness}" type="cylinder" solref="0.1"/>
                </body>
            </body>

            <body name="upright1" pos = "{body_x} {-1*(body_y + upright_spacing)} 0" quat="1 1 0 0">
                <geom mass="1.0" pos="0 0 0" rgba="0 1 0 1" size="{upright_edge_length} {upright_edge_length} {upright_thickness}" type="box"/>
                <joint axis="0 1 0" damping="0.0" name="turn1" pos="0 0 0" type="hinge"/>

                <body name="wheel1" pos="0 0 {(wheel_thickness) + upright_thickness}" quat="1 0 0 0">
                    <joint axis="0 0 1" damping="0.0" name="freespin1" pos="0 0 0" type="hinge"/>
                    <geom mass="1.0" pos="0 0 0" rgba="1 0 0 0.5" size="{wheel_diameter} {wheel_thickness}" type="cylinder" solref="0.1"/>
                </body>
            </body>

            <body name="upright2" pos = "{-1*body_x} {(body_y + upright_spacing)} 0" quat="1 -1 0 0">
                <geom mass="1.0" pos="0 0 0" rgba="0 1 0 1" size="{upright_edge_length} {upright_edge_length} {upright_thickness}" type="box"/>
                <body name="wheel2" pos="0 0 {(wheel_thickness) + upright_thickness}" quat="1 0 0 0">
                    <joint axis="0 0 1" damping="0.0" name="power0" pos="0 0 0" type="hinge"/>
                    <geom mass="1.0" pos="0 0 0" rgba="1 0 1 0.5" size="{wheel_diameter} {wheel_thickness}" type="cylinder" friction = "1 1" solref="0.1"/>
                </body>
            </body>

            <body name="upright3" pos = "{-1*body_x} {-1*(body_y + upright_spacing)} 0" quat="1 1 0 0">
                <geom mass="1.0" pos="0 0 0" rgba="0 1 0 1" size="{upright_edge_length} {upright_edge_length} {upright_thickness}" type="box"/>
                <body name="wheel3" pos="0 0 {(wheel_thickness) + upright_thickness}" quat="1 0 0 0">
                    <joint axis="0 0 1" damping="0.0" name="power1" pos="0 0 0" type="hinge"/>
                    <geom mass="1.0" pos="0 0 0" rgba="1 0 1 0.5" size="{wheel_diameter} {wheel_thickness}" type="cylinder" friction = "1 1" solref="0.1"/>
                </body>
            </body>

            <body name="origin" pos="0 0 0">
                <geom mass="1.0" pos="0 0 0" rgba="1 1 1 0.5" size="{axis_diam}" type="sphere"/>
            </body>
            <body name="localx" pos = "0 0 0">
                <geom mass="1.0" pos="{axis_length/2} 0 0" rgba="1 0 0 0.5" size="{axis_diam}" type="sphere"/>
            </body>
            <body name="localy" pos = "0 0 0" >
                <geom mass="1.0" pos="0 {axis_length/2} 0" rgba="0 1 0 0.5" size="{axis_diam}" type="sphere"/>
            </body>
            <body name="localz" pos = "0 0 0">
                <geom mass="1.0" pos="0 0 {axis_length/2}" rgba="0 0 1 0.5" size="{axis_diam}" type="sphere"/>
            </body>
        </body>

        <body mocap="true" name="mocap" pos="0.5 0.5 0.5">
			<geom conaffinity="0" contype="0" pos="0 0 0" rgba="1.0 1.0 1.0 0.5" size="0.1 0.1 0.1" type="box"></geom>
			<geom conaffinity="0" contype="0" pos="0 0 0" rgba="1.0 1.0 1.0 0.5" size="0.2 0.2 0.05" type="box"></geom>
		</body>

        <body name="floor" pos="0 0 0.025">
            <geom condim="3" size="10.0 10.0 0.02" rgba="0 1 0 1" type="box"/>
        </body>
    </worldbody>
    <actuator>
        <motor joint="turn0"/>
        <motor joint="power0"/>
        <motor joint="power1"/>
    </actuator>
</mujoco>
"""

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0
Ts=0.005
f = 1.0 # Frequency

while True:
    # sim.data.ctrl[0] = math.cos(2*np.pi*t*Ts*f)*0.1
    # sim.data.ctrl[1] = math.sin(2*np.pi*t*Ts)*100
    # sim.data.ctrl[0] = 0.1
    # sim.data.ctrl[1] = -1
    # sim.data.ctrl[2] = -1
    t += 1
    sim.step()
    viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break