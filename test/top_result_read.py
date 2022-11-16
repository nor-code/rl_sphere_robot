import json
import matplotlib.pyplot as plt

fig, ax = plt.subplots(6, 10)

i = 0
for m in range(1, 11):
    j = open('./top_10_' + str(m) + '.json')
    data = json.load(j)

    info = data[str(0)]
    ax[0][i].plot(info["res_x"][1:], info["res_y"][1:])
    ax[0][i].plot(info["x_req"], info["y_req"])
    ax[0][i].grid()

    info = data[str(1)]
    ax[1][i].plot(info["res_x"][1:], info["res_y"][1:])
    ax[1][i].plot(info["x_req"], info["y_req"])
    ax[1][i].grid()

    info = data[str(2)]
    ax[2][i].plot(info["res_x"][1:], info["res_y"][1:])
    ax[2][i].plot(info["x_req"], info["y_req"])
    ax[2][i].grid()

    info = data[str(3)]
    ax[3][i].plot(info["res_x"][1:], info["res_y"][1:])
    ax[3][i].plot(info["x_req"], info["y_req"])
    ax[3][i].grid()

    info = data[str(4)]
    ax[4][i].plot(info["res_x"][1:], info["res_y"][1:])
    ax[4][i].plot(info["x_req"], info["y_req"])
    ax[4][i].grid()

    info = data[str(5)]
    ax[5][i].plot(info["res_x"][1:], info["res_y"][1:])
    ax[5][i].plot(info["x_req"], info["y_req"])
    ax[5][i].grid()

    i += 1

plt.show()
