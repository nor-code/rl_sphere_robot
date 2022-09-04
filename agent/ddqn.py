# TODO double DQN
import numpy as np

PC = [1, 1]
len_PC = np.linalg.norm(PC)
x_center = 3.5
y_center = 2.5
x = 3
y = 3

cos_a = np.dot(PC, [1, 0]) / len_PC
a = np.arccos(cos_a)

if np.dot(PC, [0, 1]) < 0:
    a = 2 * np.pi - a

M = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])  # от Ox'y' -> Oxy


x -= x_center
y -= y_center
x, y = np.dot(M, [x, y])


print("x = ", x, "y = ", y)