import numpy as np


def trajectory():
    t = np.linspace(0, 2 * np.pi, 120)
    x_ = [2 * np.sin(t_) for t_ in t]
    y_ = [2 * np.cos(t_) - 2 for t_ in t]
    return x_, y_

x, y = trajectory()

print()