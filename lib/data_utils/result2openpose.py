import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
from mpl_toolkits.mplot3d import Axes3D
import json
import time
import argparse
import time
from tqdm import tqdm
import open3d as o3d
import sys
sys.path.append('/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/lib')
# from utils.eval_utils import batch_compute_similarity_transform_np
from glob import glob
import os.path as osp
import os

def get_you2me2d_joint_names():
    return [
        "nose",      # 0
        "neck",      # 1
        "rshoulder", # 2
        "relbow",    # 3
        "rwrist",    # 4
        "lshoulder", # 5
        "lelbow",    # 6
        "lwrist",    # 7
        "hip",       # 8
        "rhip",      # 9
        "rknee",     # 10
        "rankle",    # 11
        "lhip",      # 12
        "lknee",     # 13
        "lankle",    # 14
        "reye",      # 15
        "leye",      # 16
        "rear",      # 17
        "lear",      # 18
        "lbigtoe",   # 19
        "lsmalltoe", # 20
        "lheel",     # 21
        "rbigtoe",   # 22
        "rsmalltoe", # 23
        "rheel",     # 24
    ]

def get_spin_joint_names():
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'lbigtoe',     # 19
        'lsmalltoe',   # 20
        'lheel',       # 21
        'rbigtoe',     # 22
        'rsmalltoe',   # 23
        'rheel',       # 24
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
        'headtop',        # 38
        'hip',            # 39 'Pelvis (MPII)', # 39
        'thorax',         # 40 'Thorax (MPII)', # 40
        'Spine (H36M)',   # 41
        'Jaw (H36M)',     # 42
        'Head (H36M)',    # 43
        'nose',           # 44
        'leye',           # 45 'Left Eye', # 45
        'reye',           # 46 'Right Eye', # 46
        'lear',           # 47 'Left Ear', # 47
        'rear',           # 48 'Right Ear', # 48
    ]

def get_spin14_joint_names():
    return [
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
        'headtop',        # 38
    ]


def get_colors():
    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
    }
    return colors

def convert_kps(joints2d, src, dst):
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()

    out_joints2d = np.zeros((joints2d.shape[0], len(dst_names), 3))

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            out_joints2d[:, idx] = joints2d[:, src_names.index(jn)]

    return out_joints2d


### read ground truth
#  '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/non_reg_kinect/you2me_output
target_path = '/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/kinect/final/trainseq/repr_table6_you2me_kinect_model_output'
seq_name_list = glob(target_path + '/*_gt.npy')


for gt_file in seq_name_list:

    gt_path = gt_file #osp.join(target_path,'gt.npy') # catch55_gt.npy 2-catch2_
    pred_path = osp.join(target_path,osp.basename(gt_file).replace("gt","pred"))  #osp.join(target_path,'pred.npy') # 2-catch2_

    gt_np= np.load(gt_path)
    pred_np= np.load(pred_path)
    gt_sub_np = gt_np #[:, 25:39, :]
    # pred_np = pred_np[:, 25:39, :]

    # find the range of pred_np
    # print(np.min(pred_np),np.max(pred_np))
    print('shape of gt',np.shape(gt_np))
    print('shape of pred',np.shape(pred_np))

    ### rotate and set z to positive

    ### rotate and set z to positive
    seq_num = 0
    zxy2xyz_rotmat = np.array([[1, 0, 0 ],
                            [0, 0, 1,],
                            [0, -1, 0]])
    gt_sub_np = np.matmul(gt_sub_np, zxy2xyz_rotmat)
    # x_axis_int = gt_sub_np[seq_num,2+25, :] - gt_sub_np[seq_num,3+25, :] # work for 49
    x_axis_int = gt_sub_np[seq_num,2, :] - gt_sub_np[seq_num,3, :] # work for 14
    x_axis_int[-1] = 0
    x_axis_int = x_axis_int / np.linalg.norm(x_axis_int)
    z_axis_int = np.array([0, 0, 1])
    y_axis_int = np.cross(z_axis_int, x_axis_int)
    y_axis_int = y_axis_int / np.linalg.norm(y_axis_int)
    transf_rotmat_int = np.stack([x_axis_int, y_axis_int, z_axis_int], axis=1)  # [3, 3]

    # gt_sub_np = np.matmul(gt_sub_np[:,:] - np.expand_dims(gt_sub_np[:,0+25]+gt_sub_np[:,5+25], axis=1)/2, transf_rotmat_int)  # [T(/bs), 25, 3]
    gt_sub_np = np.matmul(gt_sub_np[:,:] - np.expand_dims(gt_sub_np[:,0]+gt_sub_np[:,5], axis=1)/2, transf_rotmat_int)  # [T(/bs), 25, 3]
    max_height = np.max(gt_sub_np[0,:,2])
    min_height = np.min(gt_sub_np[0,:,2])
    print("max diff", max_height - min_height,max_height,min_height)
    height_init = max_height - min_height
    scale = 1.8/height_init
    gt_sub_np = gt_sub_np*scale


    ### change shape 14 to 25 openpose
    # openpose_format_gt = convert_kps(gt_sub_np,'spin','you2me2d') # work for 49
    openpose_format_gt = convert_kps(gt_sub_np,'spin14','you2me2d')
    # openpose_format_gt[:,8,:] = ( openpose_format_gt[:,9,:]+ openpose_format_gt[:,12,:])/2 # manual cal hip
    # print('shape of gt',np.shape(openpose_format_gt)) # num x 25 x 3

    ### add one column confidence
    length_seq = len(openpose_format_gt)
    openpose_format_gt = np.concatenate([openpose_format_gt,np.zeros((length_seq,25,1))],axis = 2)
    # print('shape of gt',np.shape(openpose_format_gt)) # num x 25 x 4# 
    # print("openpose_format_gt",openpose_format_gt[0,2,:])
    openpose_format_gt[:,:,3] = np.linalg.norm(openpose_format_gt[:,:,:3], axis=2)
    openpose_format_gt[:,:,3] = openpose_format_gt[:,:,3].astype(bool) # all zeros will still be zero
    # openpose_format_gt[:,:,3] = np.any(openpose_format_gt[:,:,:3])
    # print("openpose_format_gt",openpose_format_gt[0,2,:])
    # print("openpose_format_gt",openpose_format_gt[0,:,:])
    ### save to openpose json format
    # read openpose to see the structure
    # json_file = '/home/qimaqi/Desktop/000003.json'
    # json_data = json.load(open(json_file, 'r'))
    # print(json_data[0].keys()) # dict_keys(['id', 'keypoints3d'])
    # print(type(json_data[0]['keypoints3d'])) # list 25 x 4



    save_path_new = osp.join(osp.dirname(gt_path),osp.basename(gt_path).split('_')[0],'output','keypoints3d')
    if not osp.exists(save_path_new):
        os.makedirs(save_path_new)
    for frame_num in range(length_seq):
        save_name = str(frame_num).zfill(6) + '.json' 
        save_path = osp.join(save_path_new,save_name)
        save_json = {}
        save_json['id'] = 1
        save_json['keypoints3d'] = openpose_format_gt[frame_num,:,:].tolist()
        with open(save_path, 'w') as fp:
            json.dump([save_json], fp)
        fp.close()
# def get_common_skeleton():
#     return np.array(
#         [
#             [ 0, 1 ],
#             [ 1, 2 ],
#             [ 3, 4 ],
#             [ 4, 5 ],
#             [ 6, 7 ],
#             [ 7, 8 ],
#             [ 8, 2 ],
#             [ 8, 9 ],
#             [ 9, 3 ],
#             [ 2, 3 ],
#             [ 8, 12],
#             [ 9, 10],
#             [12, 9 ],
#             [10, 11],
#             [12, 13],
#         ]
#     )

# gt_sub_np = gt_np[:, 25:39, :]

# pred_np = pred_np[:, 25:39, :]

# ############### 580,  800
# start_t = 150 #400
# end_t = 250 #len(gt_sub_np)

# # gt_sub_np[:,:,1] = - gt_sub_np[:,:,1]    
# # pred_np[:,:,1] = - pred_np[:,:,1]

# # -1.5 
# LIMBS = get_common_skeleton()
# color_input = np.zeros([len(LIMBS), 3])

# # color_input[:,] = np.array([252, 146, 114])

# vis = o3d.visualization.Visualizer()
# vis.create_window()


# rcolor = get_colors()['red']
# pcolor = get_colors()['green']
# lcolor = get_colors()['blue']

# # build color list
# common_lr = [0,0,1,1,0,0,0,0,1,0,0,1,1,1,0]
# for index, flag in enumerate(common_lr):
#     color_input[index,:] = rcolor/255 if flag == 0 else lcolor/255

# color_input[-1,:] = np.array([0, 0, 0])
# # color_input[:,] = np.array([215, 48, 39])/255

# # pred_rhip = pred_np[:,27,0]
# # pred_lhip = 
# x_pred = pred_np[:,:,0]
# y_pred= pred_np[:,:,1]
# z_pred = pred_np[:,:,2]
# pred_hip_distance = np.linalg.norm(pred_np[:,2,:] - pred_np[:,3,:],axis=1 )

# head_pred = pred_np[:,-1,:]

# x_gt = gt_sub_np[:,:,0]
# y_gt = gt_sub_np[:,:,1]
# z_gt = gt_sub_np[:,:,2]
# gt_pred = pred_np[:,-1,:]
# gt_hip_distance = np.linalg.norm(gt_sub_np[:,2,:] - gt_sub_np[:,3,:],axis=1 )
# print("gt_hip_distance",np.mean(gt_hip_distance))



# hip_ratio = pred_hip_distance / gt_hip_distance
# print("hip_ratio",np.shape(hip_ratio),np.mean(hip_ratio))
# print("gt_hip_distance variance",np.var(gt_hip_distance))
# print("gt_hip_distance max difference",np.max(gt_hip_distance)- np.min(gt_hip_distance))
# print("gt_hip_distance mean",np.mean(gt_hip_distance))
# # ## show first 5
# # print("x_pred first 5", x_pred[:5,-1])
# # print("gt first 5", x_gt[:5,-1])

# ### show head average


# ## for t in range(start_t, end_t):
#     #### drawing
#     # skeleton_input = o3d.geometry.LineSet(
#     #     points=o3d.utility.Vector3dVector(gt_sub_np[t]), # Convert float64 numpy array of shape (n, 3) to Open3D format
#     #     lines=o3d.utility.Vector2iVector(LIMBS))

#     # pcd = o3d.geometry.PointCloud()
#     # pcd.points = o3d.utility.Vector3dVector(gt_sub_np[t])


#     # skeleton_input.colors = o3d.utility.Vector3dVector(color_input)
#     # vis.add_geometry(skeleton_input)
#     # vis.add_geometry(pcd)
#     # vis.capture_screen_image('./video/2-catch2/gt/'+str(t).zfill(4)+'.png')
#     # # ######################################################################################
#     # # skeleton_input= o3d.geometry.LineSet(
#     # #     points=o3d.utility.Vector3dVector(pred_np[t]), # Convert float64 numpy array of shape (n, 3) to Open3D format
#     # #     lines=o3d.utility.Vector2iVector(LIMBS))

#     # # pcd = o3d.geometry.PointCloud()
#     # # pcd.points = o3d.utility.Vector3dVector(pred_np[t]) 

#     # # skeleton_input.colors = o3d.utility.Vector3dVector(color_input)
#     # # vis.add_geometry(skeleton_input)
#     # # vis.add_geometry(pcd)

#     # # vis.capture_screen_image('./video/2-catch2/pred/'+str(t).zfill(4)+'.png')

    
#     # vis.poll_events()
#     # vis.update_renderer()
#     # print('time',t )
#     # time.sleep(0.1)
#     # vis.remove_geometry(skeleton_input)
#     # #vis.remove_geometry(skeleton_input2)
#     # vis.remove_geometry(pcd)
#     # #vis.remove_geometry(pcd2)
#     # # vis.remove_geometry(skeleton_rec)


# ####################### Kinect #########################################


# # # '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/rui_wang/you2me_output_kinect_new_regressor'# '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/you2me_test_output/you2me_output'
# # gt_path = osp.join(target_path,'gt.npy')
# # pred_path = osp.join(target_path,'pred.npy')


# # target_path = './video/presentation_material'
# # interact_path = osp.join(target_path,'interact.npy')
# # ego_path = osp.join(target_path,'ego.npy')
# # interact_np= np.load(interact_path)
# # ego_np= np.load(ego_path)

# # # # find the range of pred_np
# # # print(np.min(pred_np),np.max(pred_np))
# # print('shape of gt',np.shape(interact_np))
# # print('shape of pred',np.shape(ego_np))

# # def get_common_skeleton():
# #     return np.array(
# #         [
# #             [ 0, 1 ],
# #             [ 1, 2 ],
# #             [ 3, 4 ],
# #             [ 4, 5 ],
# #             [ 6, 7 ],
# #             [ 7, 8 ],
# #             [ 8, 2 ],
# #             [ 8, 9 ],
# #             [ 9, 3 ],
# #             [ 2, 3 ],
# #             [ 8, 12],
# #             [ 9, 10],
# #             [12, 9 ],
# #             [10, 11],
# #             [12, 13],
# #         ]
# #     )

# # cmu2common = [14,13,12,6, 7,8,11,10,9,3,4,5,0,1]
# # interact_np = interact_np[:, cmu2common, :3]
# # interact_np[:,:,1] = -interact_np[:,:,1]
# # ego_np = ego_np[:, cmu2common, :3]
# # ego_np[:,:,1] = -ego_np[:,:,1]

# # ############### 580,  800
# # start_t = 0 #400
# # end_t = len(interact_np)

# # LIMBS = get_common_skeleton()
# # color_input = np.zeros([len(LIMBS), 3])

# # # color_input[:,] = np.array([252, 146, 114])

# # vis = o3d.visualization.Visualizer()
# # vis.create_window()


# # rcolor = get_colors()['red']
# # pcolor = get_colors()['green']
# # lcolor = get_colors()['blue']

# # # build color list
# # common_lr = [0,0,1,1,0,0,0,0,1,0,0,1,1,1,0]
# # for index, flag in enumerate(common_lr):
# #     color_input[index,:] = rcolor/255 if flag == 0 else lcolor/255

# # color_input[-1,:] = np.array([0, 0, 0])
# # # color_input[:,] = np.array([215, 48, 39])/255

# # for t in range(start_t, end_t):
# #     skeleton_input1 = o3d.geometry.LineSet(
# #         points=o3d.utility.Vector3dVector(interact_np[t]), # Convert float64 numpy array of shape (n, 3) to Open3D format
# #         lines=o3d.utility.Vector2iVector(LIMBS))

# #     pcd1 = o3d.geometry.PointCloud()
# #     pcd1.points = o3d.utility.Vector3dVector(interact_np[t])


# #     skeleton_input1.colors = o3d.utility.Vector3dVector(color_input)
# #     vis.add_geometry(skeleton_input1)
# #     vis.add_geometry(pcd1)

# #     skeleton_input2= o3d.geometry.LineSet(
# #         points=o3d.utility.Vector3dVector(ego_np[t]), # Convert float64 numpy array of shape (n, 3) to Open3D format
# #         lines=o3d.utility.Vector2iVector(LIMBS))

# #     pcd2 = o3d.geometry.PointCloud()
# #     pcd2.points = o3d.utility.Vector3dVector(ego_np[t]) 

# #     skeleton_input2.colors = o3d.utility.Vector3dVector(color_input)
# #     vis.add_geometry(skeleton_input2)
# #     vis.add_geometry(pcd2)

# #     # o3d.visualization.draw_geometries(o3d.utility.Vector3dVector(gt_sub_np[t]))
# #     # vis.add_geometry(skeleton_rec)

# #     # ctr = vis.get_view_control()
# #     # cam_param = ctr.convert_to_pinhole_camera_parameters()
# #     # cam_param = update_cam(cam_param, trans)
# #     # ctr.convert_from_pinhole_camera_parameters(cam_param)
# #     # vis.capture_screen_image('./video/'+str(t).zfill(4)+'.png')
# #     vis.poll_events()
# #     vis.update_renderer()
# #     print('time',t )
# #     time.sleep(0.1)
# #     vis.remove_geometry(skeleton_input1)
# #     vis.remove_geometry(skeleton_input2)
# #     vis.remove_geometry(pcd1)
# #     vis.remove_geometry(pcd2)
# #     # vis.remove_geometry(skeleton_rec)