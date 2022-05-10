import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
from mpl_toolkits.mplot3d import Axes3D
import json
import time
import os
import os.path as osp
import argparse
import glob


from tqdm import tqdm
import open3d as o3d

def get_cmu_common_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 0, 3 ],
            [ 0, 9 ],
            [ 0, 2 ],
            [ 2, 12],
            [ 2, 6 ],
            [ 3, 4 ],
            [ 4, 5 ],
            [ 9, 10],
            [10, 11],
            [ 6, 7 ],
            [ 7, 8 ],
            [12 , 13],
            [13 , 14]
        ]
    )

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

def read_body3DScene(json_file):
    json_data = json.load(open(json_file, 'r'))
    people = json_data['bodies']
    body_0_joints = people[0]["joints19"]
    body_1_joints = people[1]["joints19"]
    return body_0_joints, body_1_joints

# target path using cmu
data_root = '/media/qimaqi/Alexander/you2me/cmu'
target_path = '1-catch1'
ground_truth = os.path.join(data_root, target_path + '/synchronized/gt-skeletons')
json_files_list = glob.glob(ground_truth + '/*.json')

if not os.path.exists(osp.join(data_root, target_path,'video')):
    os.mkdir(osp.join(data_root, target_path,'video'))

order_num = []
for json_file in json_files_list:
    json_file_name = json_file.split('/')[-1]
    order_num.append(int(json_file_name.split('_')[-1].split('.')[0]))
# reorder and get index
new_index = np.argsort(order_num)
json_files_list = np.array(json_files_list)[new_index]

vis = o3d.visualization.Visualizer()
vis.create_window()

LIMBS = get_cmu_common_skeleton()
color_input_1 = np.zeros([len(LIMBS), 3])
color_input_2 = np.zeros([len(LIMBS), 3])

for json_file in tqdm(json_files_list):
    json_file_name = json_file.split('/')[-1]
    order_num = (int(json_file_name.split('_')[-1].split('.')[0]))

    interactee_kp,ego_kp = read_body3DScene(json_file)
    interactee_kp = np.array(interactee_kp).reshape(19,4)
    interactee_kp = interactee_kp[:,:3]

    ego_kp = np.array(ego_kp).reshape(19,4)
    ego_kp = ego_kp[:,:3]

    skeleton_input1 = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(interactee_kp), # Convert float64 numpy array of shape (n, 3) to Open3D format
        lines=o3d.utility.Vector2iVector(LIMBS))

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(interactee_kp)
    skeleton_input1.colors = o3d.utility.Vector3dVector(color_input_1)
        
    vis.add_geometry(skeleton_input1)
    vis.add_geometry(pcd1)

    skeleton_input2= o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(ego_kp), # Convert float64 numpy array of shape (n, 3) to Open3D format
        lines=o3d.utility.Vector2iVector(LIMBS))

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(ego_kp) 

    skeleton_input2.colors = o3d.utility.Vector3dVector(color_input_2)
    vis.add_geometry(skeleton_input2)
    vis.add_geometry(pcd2)

    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.2)
    vis.capture_screen_image(osp.join(data_root, target_path,'video',str(order_num).zfill(4)+'.png'))
    vis.remove_geometry(skeleton_input1)
    vis.remove_geometry(skeleton_input2)
    vis.remove_geometry(pcd1)
    vis.remove_geometry(pcd2)


