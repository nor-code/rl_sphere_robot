import json
import matplotlib.pyplot as plt

j = open('./top_10.json')

data = json.load(j)

fig, ax = plt.subplots(2, 5)

for i in range(5):
    info = data[str(i)]
    ax[0][i].plot(info["res_x"][1:], info["res_y"][1:])
    ax[0][i].plot(info["x_req"], info["y_req"])
    ax[0][i].grid()

for i in range(5):
    info = data[str(i + 5)]
    ax[1][i].plot(info["res_x"][1:], info["res_y"][1:])
    ax[1][i].plot(info["x_req"], info["y_req"])
    ax[1][i].grid()

plt.show()
