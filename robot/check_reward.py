import matplotlib.pyplot as plt
import numpy as np

from robot.enviroment import circle

x_, y_ = circle()

len = np.sqrt((x_[1] - x_[0]) ** 2 + (y_[1] - y_[0]) ** 2)

t = np.linspace(0, len, 20)

x = np.linspace(0, len, 20)
y = x * 1/np.sqrt(x + 0.0001) #np.linspace(1, 1, 20)

def discountH(x, y):
    # if x <= 3/5 * len:
    return np.exp(y + 1) / 2
    # else:
    #     return np.exp(-((y+1) - len / 2) **2)


DiscountH = [discountH(x[i], y[i]) for i in range(20)]  # 2 * np.exp(-(y - len/2)**2)
RewardDistance = len / (len - x -0.001)# 2 * np.tanh(len / (len - x))
plt.plot(x, DiscountH, label="discountH")
plt.plot(np.linspace(0, len-0.001, 20), RewardDistance, label="reward")
# plt.plot(np.linspace(0, len-0.001, 20), RewardDistance - DiscountH, label="diff dis-rew")
plt.plot(x, y, label='change trajectory')
plt.legend(loc='upper right')
plt.show()

