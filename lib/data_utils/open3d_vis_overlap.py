import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
from mpl_toolkits.mplot3d import Axes3D
import json
import time

import argparse
import sys
import time
from tqdm import tqdm
import open3d as o3d
sys.path.append('/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/lib')
from utils.eval_utils import batch_compute_similarity_transform_torch
import torch
import os



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

parser = argparse.ArgumentParser(description='Vis ego joints.')
parser.add_argument("--target_path", default='/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/kinect/final/trainseq/repr_table6_you2me_kinect_model_output')  # 'C:/Users/siwei/Desktop/record_20210907'
# set start/end frame (start/end frame = 10/1000: from frame_00010.jpg to frame_01000.jpg), only need for keypoints_folder_name='keypoints'
parser.add_argument("--start_frame", default=0, type=int)
parser.add_argument("--end_frame", default=-1, type=int)
# parser.add_argument("--vis_seq", default='')
parser.add_argument("--save_dir", default='output')
args = parser.parse_args()

# target path
#  '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/non_reg_kinect/you2me_output
target_path = args.target_path
#'/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/cmu/repr_table6_you2me_cmu_model_output/'
#'/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/kinect/04_25_23/repr_table6_you2me_kinect_model_output/'
#'./outputs/kinect/04_25_13/repr_table6_you2me_kinect_model_output'
#'/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/kinect/repr_table6_you2me_kinect_model_output'
#'/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/cmu/repr_table6_you2me_cmu_model_output/'
# '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/kinect/repr_table6_you2me_kinect_model_output'
# '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/cmu/repr_table6_you2me_cmu_model_output/'
# # '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/rui_wang/you2me_output_kinect_new_regressor'# '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/you2me_test_output/you2me_output'
# seq_lists = ['patty26', 'patty27', 'patty28', 'patty30', 'patty31', 'patty32', 'patty34']
seq_lists = ['catch36', 'catch37', 'catch39', 'catch40', 'catch41', 'catch42', 'convo43', 'convo46', 'convo47', 'convo53', 'convo59']

for vis_seq in seq_lists:
    gt_path = osp.join(target_path, vis_seq + '_gt.npy') # catch55_gt.npy 2-catch2_
    pred_path = osp.join(target_path,vis_seq + '_pred.npy') # 2-catch2_

    # /home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/cmu/final/repr_table6_you2me_cmu_model_output/gt.npy
    gt_np= np.load(gt_path)
    pred_np= np.load(pred_path)
    save_folder_path = osp.join('.','video',vis_seq)
    print("saving images",save_folder_path)
    save_folder_overlap_path = osp.join(osp.join(save_folder_path,'overlap'))

    if not osp.exists(save_folder_overlap_path):
        os.makedirs(save_folder_overlap_path)

    # find the range of pred_np
    print(np.min(pred_np),np.max(pred_np))
    print('shape of gt',np.shape(gt_np))
    print('shape of pred',np.shape(pred_np))

    def get_common_skeleton():
        return np.array(
            [
                [ 0, 1 ],
                [ 1, 2 ],
                [ 3, 4 ],
                [ 4, 5 ],
                [ 6, 7 ],
                [ 7, 8 ],
                [ 8, 2 ],
                [ 8, 9 ],
                [ 9, 3 ],
                [ 2, 3 ],
                [ 8, 12],
                [ 9, 10],
                [12, 9 ],
                [10, 11],
                [12, 13],
            ]
        )

    gt_sub_np = gt_np#[:, 25:39, :]

    pred_np = pred_np#[:, 25:39, :]

    pred_j3ds = torch.from_numpy(pred_np).float()
    target_j3ds = torch.from_numpy(gt_sub_np).float()

    S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
    pred_np = S1_hat.cpu().numpy()
    ############### 580,  800
    start_t = args.start_frame
    if args.end_frame == -1:
        end_t = len(gt_sub_np)  
    else:  
        end_t = args.end_frame #len(gt_sub_np)

    # gt_sub_np[:,:,1] = - gt_sub_np[:,:,1]    
    # pred_np[:,:,1] = - pred_np[:,:,1]

    # -1.5 
    LIMBS = get_common_skeleton()
    color_input = np.zeros([len(LIMBS), 3])

    # color_input[:,] = np.array([252, 146, 114])

    vis = o3d.visualization.Visualizer()
    vis.create_window()


    rcolor = get_colors()['red']
    pcolor = get_colors()['green']
    lcolor = get_colors()['blue']

    # build color list
    common_lr_gt = np.array([lcolor/255]*len(LIMBS))
    common_lr_pred = np.array([rcolor/255]*len(LIMBS))
    # for index, flag in enumerate(common_lr):
    #     color_input[index,:] = rcolor/255 if flag == 0 else lcolor/255

    # color_input[-1,:] = np.array([0, 0, 0])
    # color_input[:,] = np.array([215, 48, 39])/255

    # pred_rhip = pred_np[:,27,0]
    # pred_lhip = 
    x_pred = pred_np[:,:,0]
    y_pred= pred_np[:,:,1]
    z_pred = pred_np[:,:,2]
    pred_hip_distance = np.linalg.norm(pred_np[:,2,:] - pred_np[:,3,:],axis=1 )

    head_pred = pred_np[:,-1,:]

    x_gt = gt_sub_np[:,:,0]
    y_gt = gt_sub_np[:,:,1]
    z_gt = gt_sub_np[:,:,2]
    gt_pred = pred_np[:,-1,:]
    gt_hip_distance = np.linalg.norm(gt_sub_np[:,2,:] - gt_sub_np[:,3,:],axis=1 )
    # print("gt_hip_distance",np.mean(gt_hip_distance))



    # hip_ratio = pred_hip_distance / gt_hip_distance
    # print("hip_ratio",np.shape(hip_ratio),np.mean(hip_ratio))
    # print("gt_hip_distance variance",np.var(gt_hip_distance))
    # print("gt_hip_distance max difference",np.max(gt_hip_distance)- np.min(gt_hip_distance))
    # print("gt_hip_distance mean",np.mean(gt_hip_distance))

    # ## show first 5
    # print("x_pred first 5", x_pred[:5,-1])
    # print("gt first 5", x_gt[:5,-1])

    ## show head average


    for t in tqdm(range(start_t, end_t)):
        ### drawing
        # if vis_object == 'gt':
        skeleton_input1 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(gt_sub_np[t]), # Convert float64 numpy array of shape (n, 3) to Open3D format
            lines=o3d.utility.Vector2iVector(LIMBS))

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(gt_sub_np[t])


        skeleton_input1.colors = o3d.utility.Vector3dVector(common_lr_gt)
        vis.add_geometry(skeleton_input1)
        vis.add_geometry(pcd1)
        # vis.capture_screen_image(osp.join(save_folder_gt_path, str(t).zfill(4)+'.png'))
        # vis.remove_geometry(skeleton_input)
        # vis.remove_geometry(pcd)
        
        # ######################################################################################
        # elif vis_object == 'pred':
        skeleton_input2= o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pred_np[t]), # Convert float64 numpy array of shape (n, 3) to Open3D format
            lines=o3d.utility.Vector2iVector(LIMBS))

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pred_np[t]) 

        skeleton_input2.colors = o3d.utility.Vector3dVector(common_lr_pred)
        vis.add_geometry(skeleton_input2)
        vis.add_geometry(pcd2)

        vis.capture_screen_image(osp.join(save_folder_overlap_path, str(t).zfill(4)+'.png'))

        vis.get_render_option().point_size = 0
        vis.poll_events()
        vis.update_renderer()
        # print('time',t )
        # time.sleep(0.1)
        vis.remove_geometry(skeleton_input1)
        vis.remove_geometry(skeleton_input2)
        vis.remove_geometry(pcd1)
        vis.remove_geometry(pcd2)
        # vis.remove_geometry(skeleton_rec)


####################### Kinect #########################################


######################## interact and ego

# # '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/rui_wang/you2me_output_kinect_new_regressor'# '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/you2me_test_output/you2me_output'
# target_path = '/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/kinect/final/repr_table6_you2me_kinect_model_output'
# gt_path = osp.join(target_path,'gt.npy')
# pred_path = osp.join(target_path,'pred.npy')


# target_path = './video/presentation_material'
# interact_path = osp.join(target_path,'interact.npy')
# ego_path = osp.join(target_path,'ego.npy')
# interact_np= np.load(interact_path)
# ego_np= np.load(ego_path)

# # # find the range of pred_np
# # print(np.min(pred_np),np.max(pred_np))
# print('shape of gt',np.shape(interact_np))
# print('shape of pred',np.shape(ego_np))

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

# cmu2common = [14,13,12,6, 7,8,11,10,9,3,4,5,0,1]
# interact_np = interact_np[:, cmu2common, :3]
# interact_np[:,:,1] = -interact_np[:,:,1]
# ego_np = ego_np[:, cmu2common, :3]
# ego_np[:,:,1] = -ego_np[:,:,1]

# ############### 580,  800
# start_t = 0 #400
# end_t = len(interact_np)

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

# for t in range(start_t, end_t):
#     skeleton_input1 = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(interact_np[t]), # Convert float64 numpy array of shape (n, 3) to Open3D format
#         lines=o3d.utility.Vector2iVector(LIMBS))

#     pcd1 = o3d.geometry.PointCloud()
#     pcd1.points = o3d.utility.Vector3dVector(interact_np[t])


#     skeleton_input1.colors = o3d.utility.Vector3dVector(color_input)
#     vis.add_geometry(skeleton_input1)
#     vis.add_geometry(pcd1)

#     # skeleton_input2= o3d.geometry.LineSet(
#     #     points=o3d.utility.Vector3dVector(ego_np[t]), # Convert float64 numpy array of shape (n, 3) to Open3D format
#     #     lines=o3d.utility.Vector2iVector(LIMBS))

#     # pcd2 = o3d.geometry.PointCloud()
#     # pcd2.points = o3d.utility.Vector3dVector(ego_np[t]) 

#     # skeleton_input2.colors = o3d.utility.Vector3dVector(color_input)
#     # vis.add_geometry(skeleton_input2)
#     # vis.add_geometry(pcd2)

#     # o3d.visualization.draw_geometries(o3d.utility.Vector3dVector(gt_sub_np[t]))
#     # vis.add_geometry(skeleton_rec)

#     # ctr = vis.get_view_control()
#     # cam_param = ctr.convert_to_pinhole_camera_parameters()
#     # cam_param = update_cam(cam_param, trans)
#     # ctr.convert_from_pinhole_camera_parameters(cam_param)
#     # vis.capture_screen_image('./video/'+str(t).zfill(4)+'.png')
#     vis.poll_events()
#     vis.update_renderer()
#     print('time',t )
#     time.sleep(0.1)
#     vis.remove_geometry(skeleton_input1)
#     # vis.remove_geometry(skeleton_input2)
#     vis.remove_geometry(pcd1)
#     # vis.remove_geometry(pcd2)
#     # vis.remove_geometry(skeleton_rec)