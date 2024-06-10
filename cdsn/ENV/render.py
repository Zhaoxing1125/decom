Cam_Pose = [[-742, 706, -62.842588558231455], [-843, 69, -26.590794532324466], [510, 703, -135.84636503921902],
            [466, -609, 153.13035548432399]]
Target_Pose = [[407.90650859, -716.624028],
               [-64.83188835, -233.64760113],
               [-980.29575616, 201.18355808],
               [-493.24174167, 655.69319226],
               [-571.57383471, -673.35637078]]
Target_camera_dict = {0: [], 1: [3], 2: [], 3: [], 4: [1]}
Camera_target_dict = {0: [], 1: [4], 2: [], 3: [1]}
Distance = [[1829.24686786, 1158.22893495, 558.23338079, 253.79410157, 1389.84498251],
            [1477.15002847, 834.94980715, 190.58493563, 683.03714475, 790.42086539],
            [1423.29036457, 1098.97244214, 1572.51428681, 1004.35647371, 1750.47388421],
            [122.3020243, 650.13223042, 1657.7601793, 1587.3227742, 1039.56779718]]
reward = [0.4]

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
visual_distance = 100


def render(camera_pos, target_pos, reward=None, save=False):
    camera_pos = np.array(camera_pos)
    target_pos = np.array(target_pos)

    camera_pos[:, :2] /= 1000.0
    target_pos[:, :2] /= 1000.0

    length = 600
    area_length = 1  # for random cam loc
    target_pos[:, :2] = (target_pos[:, :2] + 1) / 2
    camera_pos[:, :2] = (camera_pos[:, :2] + 1) / 2

    img = np.zeros((length + 1, length + 1, 3)) + 255
    num_cam = len(camera_pos)
    num_target = len(target_pos)
    camera_position = [camera_pos[i][:2] for i in range(num_cam)]
    target_position = [target_pos[i][:2] for i in range(num_target)]

    camera_position = length * (1 - np.array(camera_position) / area_length) / 2
    target_position = length * (1 - np.array(target_position) / area_length) / 2
    abs_angles = [camera_pos[i][2] * -1 for i in range(num_cam)]

    fig = plt.figure(0)
    plt.cla()
    plt.imshow(img.astype(np.uint8))

    # get camera's view space positions
    visua_len = 100  # length of arrow
    L = 140  # length of arrow
    ax = plt.gca()
    for i in range(num_cam):

        # dash-circle
        r = L
        a, b = np.array(camera_position[i]) + visua_len
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = a + r * np.cos(theta)
        y = b + r * np.sin(theta)
        plt.plot(x, y, linestyle=' ',
                 linewidth=1,
                 color='steelblue',
                 dashes=(6, 5.),
                 dash_capstyle='round',
                 alpha=0.9)

        # fill circle
        disk1 = plt.Circle((a, b), r, color='steelblue', fill=True, alpha=0.05)
        ax.add_artist(disk1)
        #

    for i in range(num_cam):
        theta = abs_angles[i]  # -90
        theta -= 90
        the1 = theta - 45
        the2 = theta + 45

        a = camera_position[i][0] + visua_len
        b = camera_position[i][1] + visua_len
        wedge = mpatches.Wedge((a, b), L, the1*-1, the2*-1+180, color='green', alpha=0.2)
        # print(i, the1*-1, the2*-1)
        ax.add_artist(wedge)

        disk1 = plt.Circle((camera_position[i][0] + visua_len, camera_position[i][1] + visua_len), 4, color='slategray', fill=True)
        ax.add_artist(disk1)
        plt.annotate(str(i + 1), xy=(camera_position[i][0] + visua_len, camera_position[i][1] + visua_len),
                     xytext=(camera_position[i][0] + visua_len, camera_position[i][1] + visua_len), fontsize=10,
                     color='black')

    plt.text(5, 5, '{} sensors & {} targets'.format(num_cam, num_target), color="black")

    for i in range(num_target):
        plt.plot(target_position[i][0] + visua_len, target_position[i][1] + visua_len, color='firebrick',
                 marker="o")
        plt.annotate(str(i + 1), xy=(target_position[i][0] + visua_len, target_position[i][1] + visua_len),
                     xytext=(target_position[i][0] + visua_len, target_position[i][1] + visua_len), fontsize=10,
                     color='maroon')

    plt.axis('off')
    # plt.show()
    if save:
        file_path = '../demo/img'
        file_name = '{}.jpg'.format(datetime.now())
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        plt.savefig(os.path.join(file_path, file_name))
    plt.pause(0.01)


if __name__ == '__main__':
    render(Cam_Pose, Target_Pose, reward)