import numpy as np
from dataclasses import dataclass

import gtsam

from gtsam.utils import plot

import open3d as o3d
'''
@dataclass
class Traj:
    t: np.array
    xyz: np.array
    qxyzw: np.array

'''
@dataclass
class Traj:
    t: np.array
    xyz: np.array
    rph: np.array

downsampling = 1
traj_est_file = '/home/jmw0611/MyResearch/rpg_trajectory_evaluation_open3d/results/example/stamped_traj_estimate_orig.txt'
traj_est = np.loadtxt(traj_est_file, delimiter=',')

#traj_est = Traj(t=traj_est[:, 0], xyz=traj_est[:, 1:4], qxyzw=traj_est[:, 4:])
traj_est = Traj(t=traj_est[:, 0], xyz=traj_est[:, 1:4], rph=np.deg2rad(traj_est[:, 4:]))

traj_est_poses = []
traj_est_poses_mesh = []
for pose_idx in range( traj_est.t.shape[0] ):
    trans = gtsam.Point3(traj_est.xyz[pose_idx, :])
    rph = traj_est.rph[pose_idx, :]
    R = gtsam.Rot3.RzRyRx(rph[0], rph[1], rph[2]) # gtsam uses the order of w,x,y,z
    #qxyzw = traj_est.qxyzw[pose_idx, :]
    #R = gtsam.Rot3.Quaternion(qxyzw[-1], qxyzw[0], qxyzw[1], qxyzw[2]) # gtsam uses the order of w,x,y,z
    pose = gtsam.Pose3(R, trans)
    traj_est_poses.append(pose)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh_t = mesh.transform(pose.matrix())
    traj_est_poses_mesh.append(mesh_t)

#_l2c = [ 1, 0, 0, 0, 0,1,0,0,0,0,1,0]
_l2c = [0.6728451712377544, -0.7397833300947684, 7.47873302166832e-06, 0, 0.7397833296516727, 0.6728451711305004, 2.925494093534264e-05, 0, -2.667434702667728e-05, -1.4151403726845309e-05, 0.9999999995441085, 0]
l2c_rot = gtsam.Rot3(_l2c[0], _l2c[1],_l2c[2], _l2c[4], _l2c[5], _l2c[6], _l2c[8], _l2c[9], _l2c[10])

c2l = gtsam.Pose3(l2c_rot, gtsam.Point3(_l2c[3], _l2c[7], _l2c[11]))
l2c = c2l.inverse()

###
# parse reorigined pose (init is origin)

traj_est_poses_reorigin = []
for ii, curr_pose_est in enumerate(traj_est_poses):
    if(ii == 0):
        init_pose = gtsam.Pose3()
        prev_pose_est = curr_pose_est

        traj_est_poses_reorigin.append(init_pose)
        continue 

    rel_pose = prev_pose_est.between(curr_pose_est)

    curr_pose_recalc = traj_est_poses_reorigin[-1].compose(l2c).compose(rel_pose).compose(c2l)
    print(curr_pose_recalc)
    traj_est_poses_reorigin.append(curr_pose_recalc)
    prev_pose_est = curr_pose_est


traj_gt_poses_mesh = []
for ii, pose in enumerate(traj_est_poses_reorigin):
    # print(pose)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(5)
    mesh_t = mesh.transform(pose.matrix())
    if(ii%downsampling == 0):
        traj_gt_poses_mesh.append(mesh_t)

o3d.visualization.draw_geometries(traj_gt_poses_mesh)

# rewrite text file 
poses_lines = []
for ii, pose in enumerate(traj_est_poses_reorigin):
    timestamp = traj_est.t[ii]
    #xyz = traj_est_poses[ii].translation()
    xyz = pose.translation()

    xyz = np.array([xyz.x(), xyz.y(), xyz.z()])
    qwxyz = pose.rotation().quaternion()
    qxyzw = np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]])
    
    line = np.concatenate( (np.expand_dims(timestamp, axis=0), xyz, qxyzw) )

    poses_lines.append(line)

poses_lines = np.array(poses_lines)
print(poses_lines.shape)
poses_lines_filename = traj_est_file[:-9] + ".txt"
np.savetxt(poses_lines_filename, poses_lines, delimiter=' ')
print('reorigined pose file is saved. see the file', poses_lines_filename)    
