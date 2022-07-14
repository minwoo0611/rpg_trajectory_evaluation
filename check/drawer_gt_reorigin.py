import numpy as np
from dataclasses import dataclass

import gtsam

import matplotlib.pyplot as plt
from gtsam.utils import plot

import open3d as o3d

@dataclass
class TrajGT:
    t: np.array
    xyz: np.array
    qxyzw: np.array
'''
class TrajGT:
    t: np.array
    xyz: np.array
    rph: np.array
'''

downsampling = 100
traj_gt_file = '/home/jmw0611/MyResearch/rpg_trajectory_evaluation_open3d/results/example/stamped_groundtruth_orig.txt'
traj_gt = np.loadtxt(traj_gt_file, delimiter=' ')

traj_gt = TrajGT(t=traj_gt[:, 0], xyz=traj_gt[:, 1:4], qxyzw=traj_gt[:, 4:])
#traj_gt = TrajGT(t=traj_gt[:, 0], xyz=traj_gt[:, 1:4], rph=np.deg2rad(traj_gt[:, 4:]))

# parse pose (imu coord)
traj_gt_poses = []
traj_gt_poses_mesh = []
for pose_idx in range( traj_gt.t.shape[0] ):
    trans = gtsam.Point3(traj_gt.xyz[pose_idx, :])
    #rph = traj_gt.rph[pose_idx, :]
    #R = gtsam.Rot3.RzRyRx(rph[0], rph[1], rph[2]) # gtsam uses the order of w,x,y,z
    qxyzw = traj_gt.qxyzw[pose_idx, :]
    R = gtsam.Rot3.Quaternion(qxyzw[-1], qxyzw[0], qxyzw[1], qxyzw[2])
    pose = gtsam.Pose3(R, trans)
    traj_gt_poses.append(pose)

# parse reorigined pose (init is origin)

traj_gt_poses_reorigin = []
for ii, curr_pose_gt in enumerate(traj_gt_poses):
    if(ii == 0):
        init_pose = gtsam.Pose3()
        prev_pose_gt = curr_pose_gt
        traj_gt_poses_reorigin.append(init_pose)
        continue 

    rel_pose = prev_pose_gt.between(curr_pose_gt)
    curr_pose_recalc = traj_gt_poses_reorigin[-1].compose(rel_pose)
    traj_gt_poses_reorigin.append(curr_pose_recalc)
    prev_pose_gt = curr_pose_gt


traj_gt_poses_mesh = []
for ii, pose in enumerate(traj_gt_poses_reorigin):
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(5)
    mesh_t = mesh.transform(pose.matrix())
    if(ii % downsampling == 0):
        traj_gt_poses_mesh.append(mesh_t)
    
o3d.visualization.draw_geometries(traj_gt_poses_mesh)

# rewrite text file 
poses_lines = []
for ii, pose in enumerate(traj_gt_poses_reorigin):
    timestamp = traj_gt.t[ii]
    xyz = pose.translation()
    xyz = np.array([xyz.x(), xyz.y(), xyz.z()])
    qwxyz = pose.rotation().quaternion()
    qxyzw = np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]])
    line = np.concatenate( (np.expand_dims(timestamp, axis=0), xyz, qxyzw) )

    poses_lines.append(line)

poses_lines = np.array(poses_lines)
print(poses_lines.shape)
poses_lines_filename = traj_gt_file[:-9] + ".txt"
np.savetxt(poses_lines_filename, poses_lines, delimiter=' ')
print('reorigined pose file is saved. see the file', poses_lines_filename)    
