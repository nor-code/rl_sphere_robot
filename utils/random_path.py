import matplotlib.pyplot as plt

from robot.enviroment import random_trajectory,circle

for i in range(10):
    x, y = random_trajectory()
    plt.plot(x, y)
    plt.scatter(x, y, lw=0.1)

plt.grid()
plt.show()
