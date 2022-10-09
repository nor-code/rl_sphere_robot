import matplotlib.pyplot as plt

from robot.enviroment import random_trajectory

for i in range(5):
    x, y = random_trajectory()
    plt.plot(x, y)
    plt.scatter(x, y, lw=0.1)

plt.grid()
plt.show()
