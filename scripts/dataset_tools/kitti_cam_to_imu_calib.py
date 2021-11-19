"""compute camera to IMU transformation for the KITTI dataset"""
import numpy as np
import os

folder = 'kitti_drive/09-raw/2011_09_30_drive_0033_sync/2011_09_30'
imu_to_vel_txt = os.path.join(folder, 'calib_imu_to_velo.txt')

vel_to_cam_txt = os.path.join(folder, 'calib_velo_to_cam.txt')

cam_to_imu_txt = os.path.join(folder, "calib_cam_to_imu.txt")

def loadkittiRT(fn):
    with open(fn, 'r') as stream:
        lid = 0
        for line in stream:
            if lid == 1:
                R = np.zeros((3, 3))
                rags = line.split(": ")
                vals = rags[1].split()
                for i in range(3):
                    for j in range(3):
                        R[i, j] = float(vals[i * 3 + j])
            if lid == 2:
                t = np.zeros((3,))
                rags = line.split(": ")
                vals = rags[1].split()
                for i in range(3):
                    t[i] = float(vals[i])
                break
            lid += 1
    print("Loaded R\n", R)
    print("Loaded t\n", t)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


T_CV = loadkittiRT(vel_to_cam_txt)
T_VI = loadkittiRT(imu_to_vel_txt)

T_CI = np.matmul(T_CV, T_VI)
T_IC = np.eye(4)
T_IC[:3, :3] = np.transpose(T_CI[:3, :3])
T_IC[:3, 3] = - np.matmul(T_IC[:3, :3], T_CI[:3, 3])


with open(cam_to_imu_txt, 'w') as stream:
    for i in range(4):
        for j in range(3):
            stream.write("{:.9f},".format(T_IC[i, j]))
        stream.write("{:.9f}".format(T_IC[i, 3]))

        stream.write('\n')


print("Cam to imu transformation saved to {0}.\n".format(cam_to_imu_txt))
