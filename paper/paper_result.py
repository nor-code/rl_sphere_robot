import json
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom

pic3 = [8, 1, 4]
names = {0: 'а', 1: 'б', 2: 'в'}

j = 0
fig, ax = plt.subplots(1, 3)
plt.xlim([-1.3, 1.3])
plt.ylim([-0.4, 2.2])

top_file = open('results_paper.json')
load = json.load(top_file)

# 3 графика из статьи
for key in pic3:
    data = load[str(key)]

    x_res = data["res_x"][1:]
    y_res = data["res_y"][1:]
    res = np.array([x_res, y_res]).T

    x_req = data["x_req"]
    y_req = data["y_req"]

    ax[j].plot(x_res, y_res, 'r', linestyle='dashed', linewidth=3, label='random curve')
    ax[j].plot(x_req, y_req, 'b', linewidth=1.1, label='robot')
    # ax[j].legend(loc='best')
    ax[j].set_ylabel('y [m]',
                     labelpad=0.1,
                     loc="center",
                     rotation="horizontal")
    ax[j].set_xlabel('x [m]', loc='center')
    ax[j].set_title(names[j])
    ax[j].grid()

    j += 1


# значения для таблицы в статье
L2_arr = []
mu_arr = []
for key in range(10):
    data = load[str(key)]

    x_res = data["res_x"][1:]
    y_res = data["res_y"][1:]
    res = np.array([x_res, y_res]).T

    x_req = data["x_req"]
    y_req = data["y_req"]

    line = geom.LineString(np.array([x_req, y_req]).T)

    robot_line = geom.LineString(np.array([x_res, y_res]).T)

    L2 = 0.0
    mean_h = 0.0
    for i_ in range(len(x_req)):
        point = geom.Point(x_res[i_], y_res[i_])
        point_on_line = line.interpolate(line.project(point))
        dist = np.sqrt((point_on_line.x - point.x) ** 2 + (point_on_line.y - point.y) ** 2)
        L2 += dist ** 2
        mean_h += dist

    L2 = np.sqrt(L2)
    mean_h = mean_h / len(x_req)

    mu_arr.append(mean_h)
    L2_arr.append(L2)

    j += 1
    print("&  ", round(mean_h, 6), "& ", round(L2, 6), " &", round(line.length, 3), " &", round(robot_line.length, 3), "& 100.0 \\\\")

print("L2 mean = ", np.mean(mu_arr))
print("mu mean = ", np.mean(L2_arr))

plt.show()