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
import plotly.express as px
import os

# # target path
# #  '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/non_reg_kinect/you2me_output
# target_path ='/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/kinect/final/repr_table6_you2me_kinect_model_output'
# gt_path = osp.join(target_path,'gt.npy') # catch55_gt.npy 2-catch2_
# pred_path = osp.join(target_path,'pred.npy') # 2-catch2_
parser = argparse.ArgumentParser(description='Vis ego joints.')
parser.add_argument("--target_path", default='/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/kinect/final/trainseq/repr_table6_you2me_kinect_model_output')  # 'C:/Users/siwei/Desktop/record_20210907'
# set start/end frame (start/end frame = 10/1000: from frame_00010.jpg to frame_01000.jpg), only need for keypoints_folder_name='keypoints'
parser.add_argument("--start_frame", default=0, type=int)
parser.add_argument("--end_frame", default=-1, type=int)
parser.add_argument("--vis_seq", default='')
parser.add_argument("--save_dir", default='output')
args = parser.parse_args()

target_path = args.target_path
#'/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/cmu/repr_table6_you2me_cmu_model_output/'
vis_seq = args.vis_seq
gt_path = osp.join(target_path, vis_seq + '_gt.npy') # catch55_gt.npy 2-catch2_
pred_path = osp.join(target_path,vis_seq + '_pred.npy') # 2-catch2_
save_folder_path = osp.join('.','video',vis_seq)

# /home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/cmu/final/repr_table6_you2me_cmu_model_output/gt.npy
gt_np= np.load(gt_path)
pred_np= np.load(pred_path)

# find the range of pred_np
print('shape of gt',np.shape(gt_np))
print('shape of pred',np.shape(pred_np))

### chosse some points
target_j3d = gt_np#[:, 25:39, :]
pred_j3d = pred_np#[:, 25:39, :]

gt_pelvis = (target_j3d[:, 2,:] + target_j3d[:, 3,:]) / 2
target_j3d = target_j3d - gt_pelvis[:, None, :]
pred_pelvis = (pred_j3d[:, 2,:] + pred_j3d[:, 3,:]) / 2
pred_j3d = pred_j3d - pred_pelvis[:, None, :]


pred_j3ds = torch.from_numpy(pred_j3d).float()
target_j3ds = torch.from_numpy(target_j3d).float()

S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)

errors_pa_per_joint = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).cpu().numpy()
m2mm = 1 #1000

### add accel and pa accel error
# pve = np.mean(compute_error_verts(target_theta=target_theta, pred_verts=pred_verts)) * m2mm
accel = (compute_accel(S1_hat)) * m2mm
pa_accel_err = (compute_error_accel(joints_pred=S1_hat, joints_gt=target_j3ds)) * m2mm
print("shape of acc and acc error",np.shape(accel),np.shape(pa_accel_err))
### save for time efficiency
errors_pa_per_joint_save_path = osp.join(save_folder_path, 'errors_pa_per_joint.npy')
#'/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/kinect/final/repr_table6_you2me_kinect_model_output/errors_pa_per_joint.npy'
#errors_pa_per_joint.tofile(errors_pa_per_joint_save_path)
np.save(errors_pa_per_joint_save_path, errors_pa_per_joint)

# load
errors_pa_per_joint_save_path =  osp.join(save_folder_path, 'errors_pa_per_joint.npy')
errors_pa_per_joint = np.load(errors_pa_per_joint_save_path)
mean_errors_pa_all_joints = np.mean(errors_pa_per_joint)

print("input shape",np.shape(errors_pa_per_joint))
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

############################# Draw dynamic graph for per joint ############################
# save_folder_dyanmic_bar_path = osp.join(osp.join(save_folder_path,'dynamic'))
# print('max error',np.max(errors_pa_per_joint),np.max(accel),np.max(pa_accel_err))
# if not osp.exists(save_folder_dyanmic_bar_path):
#     os.makedirs(save_folder_dyanmic_bar_path)

# for frame_i in tqdm(range(len(pa_accel_err))):
#     save_figure_name = osp.join(save_folder_dyanmic_bar_path, str(frame_i).zfill(4)+'.png')

#     color_list = ['royalblue','gold','lightcoral']
#     per_frame_error = [np.mean(errors_pa_per_joint[frame_i,:]),accel[frame_i] ,pa_accel_err[frame_i] ]
#     x_frame_list = ['pa-mpjpe error','accel','accel error']

#     # color_list = ["blue"]*len(per_frame_error)
#     # red_min = max([frame_current-3,0])
#     # red_max = min([frame_current+3, frame_length])
#     # for red_index in range(red_min,red_max):
#     #     color_list[red_index] = 'red'
#     plt.figure(figsize=(6,6),dpi=100)
#     plt.bar(x_frame_list,per_frame_error,color=color_list)
#     plt.ylim(0,0.08)
#     # plt.ylabel("normalized pa mpjpe",fontsize=12)
#     plt.yticks(fontsize=12)
#     # plt.title("Error with frame changes")

#     plt.savefig(save_figure_name,dpi=100)
#     plt.close()



############################# Draw dynamic graph for per joint ############################


# per_frame_error = np.mean(errors_pa_per_joint,axis=1)
# print("per_frame_error",np.shape(per_frame_error))
# x_frame_list = list(range(len(per_frame_error)))
# frame_current = 100
# frame_length = np.shape(per_frame_error)[0]
# color_list = ["blue"]*len(per_frame_error)
# red_min = max([frame_current-3,0])
# red_max = min([frame_current+3, frame_length])
# for red_index in range(red_min,red_max):
#     color_list[red_index] = 'red'
# plt.bar(x_frame_list,per_frame_error,color=color_list,width=2)

# plt.xlabel("frame number")
# plt.ylabel("normalized pa mpjpe")
# plt.title("Error with frame changes")

# plt.show()


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
x = list(range(14))
colors = [(100/255, 149/255, 237/255), (240/255, 128/255, 128/255)]
my_cmap = clr.LinearSegmentedColormap.from_list(
    'color_map', colors, N=1000)

rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))    
print("rescale(mean_errors_pa_per_joint))",np.shape(rescale(mean_errors_pa_per_joint)))
plt.bar(x, mean_errors_pa_per_joint,color = my_cmap(rescale(mean_errors_pa_per_joint)))
point1 = [-2, mean_errors_pa_all_joints]
point2 = [16, mean_errors_pa_all_joints]
plt.plot([point1[0],point2[0]],[point1[1],point2[1]],color='green',linestyle="--" )
plt.xticks(x, joints_name, rotation = 45,fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("pa mpjpe error normalized",fontsize=18)
plt.title("per joint error comparsion",fontsize=20)
plt.show()
# plt.hist(x = errors_pa_per_joint[:,0].T)
# plt.show()

# fig = px.bar(x=joints_name, y=mean_errors_pa_per_joint,text_auto = '.2s')
# fig.show()