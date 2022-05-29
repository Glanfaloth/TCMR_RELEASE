from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
from mpl_toolkits.mplot3d import Axes3D
import json
import time
import argparse
import time
from tqdm import tqdm
import open3d as o3d
import torch
import math
from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)


# # target path
# #  '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/non_reg_kinect/you2me_output
# target_path ='/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/kinect/final/repr_table6_you2me_kinect_model_output'
# gt_path = osp.join(target_path,'gt.npy') # catch55_gt.npy 2-catch2_
# pred_path = osp.join(target_path,'pred.npy') # 2-catch2_


# # /home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/cmu/final/repr_table6_you2me_cmu_model_output/gt.npy
# gt_np= np.load(gt_path)
# pred_np= np.load(pred_path)

# # find the range of pred_np
# print('shape of gt',np.shape(gt_np))
# print('shape of pred',np.shape(pred_np))

# ### chosse some points
# target_j3d = gt_np[:, 25:39, :]
# pred_j3d = pred_np[:, 25:39, :]

# gt_pelvis = (target_j3d[:, 2,:] + target_j3d[:, 3,:]) / 2
# target_j3d = target_j3d - gt_pelvis[:, None, :]
# pred_pelvis = (pred_j3d[:, 2,:] + pred_j3d[:, 3,:]) / 2
# pred_j3d = pred_j3d - pred_pelvis[:, None, :]


# pred_j3ds = torch.from_numpy(pred_j3d).float()
# target_j3ds = torch.from_numpy(target_j3d).float()

# S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)

# errors_pa_per_joint = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).cpu().numpy()
# ### save for time efficiency
# errors_pa_per_joint_save_path = '/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/kinect/final/repr_table6_you2me_kinect_model_output/errors_pa_per_joint.npy'
# #errors_pa_per_joint.tofile(errors_pa_per_joint_save_path)
# np.save(errors_pa_per_joint_save_path, errors_pa_per_joint)

# load
errors_pa_per_joint_save_path = '/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/kinect/final/repr_table6_you2me_kinect_model_output/errors_pa_per_joint.npy'
errors_pa_per_joint = np.load(errors_pa_per_joint_save_path)
mean_errors_pa_all_joints = np.mean(errors_pa_per_joint)

print("max error for in joints and all frames",np.max(errors_pa_per_joint))
max_error = np.max(errors_pa_per_joint)

mean_errors_pa_per_joint = np.mean(errors_pa_per_joint,axis=0)
joints_name = [        
        'rankle',         # 25
        'rknee',          # 26
        'rhip',           # 27
        'lhip',           # 28
        'lknee',          # 29
        'lankle',         # 30
        'rwrist',         # 31
        'relbow',         # 32
        'rshoulder',      # 33
        'lshoulder',      # 34
        'lelbow',         # 35
        'lwrist',         # 36
        'neck',           # 37
        'nose',        # 38
        ]
print(np.shape(mean_errors_pa_per_joint))
############################# Draw dynamic graph for per joint ############################
per_frame_error = np.mean(errors_pa_per_joint,axis=1)
print("per_frame_error",np.shape(per_frame_error))
x_frame_list = list(range(len(per_frame_error)))
frame_current = 100
frame_length = np.shape(per_frame_error)[0]
color_list = ["blue"]*len(per_frame_error)
red_min = max([frame_current-3,0])
red_max = min([frame_current+3, frame_length])
for red_index in range(red_min,red_max):
    color_list[red_index] = 'red'
plt.bar(x_frame_list,per_frame_error,color=color_list,width=2)

plt.xlabel("frame number")
plt.ylabel("normalized pa mpjpe")
plt.title("Error with frame changes")

plt.show()


############################# Draw hist graph for per joint ############################
# ### sub_graph
# fig, axs = plt.subplots(2, 7)
# for i in range(14):
#     row_ = i//7
#     col_ = i - 7*row_
#     # axs[0, 0].plot(x, y)
#     axs[row_, col_].hist(errors_pa_per_joint[:,i].T,density=True,bins=10 )
#     axs[row_, col_].set_xlim(0,max_error+0.05) # tune parameters
#     axs[row_, col_].set_title(joints_name[i])
# for ax in axs.flat:
#     ax.set(xlabel=' normalized pa mpjpe', ylabel='normalized number')

# # # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()
# plt.show()

############################# Draw bar graph ##########################################
# x = list(range(14))
# colors = [(1, 1, 0), (1, 0, 0)]
# my_cmap = clr.LinearSegmentedColormap.from_list(
#     'color_map', colors, N=1000)
# # cmap = clr.LinearSegmentedColormap.from_list('custom blue',     ['#ffff00','#002266'], N=256)

# rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))    
# print("rescale(mean_errors_pa_per_joint))",np.shape(rescale(mean_errors_pa_per_joint)))
# plt.bar(x, mean_errors_pa_per_joint,color = my_cmap(rescale(mean_errors_pa_per_joint)))
# point1 = [-2, mean_errors_pa_all_joints]
# point2 = [16, mean_errors_pa_all_joints]
# plt.plot([point1[0],point2[0]],[point1[1],point2[1]],color='green',linestyle="--" )
# plt.xticks(x, joints_name)
# plt.xlabel("joint type")
# plt.ylabel("pa mpjpe error normalized")
# plt.title("per joint error comparsion")
# plt.show()
# # plt.hist(x = errors_pa_per_joint[:,0].T)
# # plt.show()