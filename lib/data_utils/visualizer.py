import numpy as np
import json
import time
import os
import os.path as osp
import glob
import bpy

people_joints = ["Neck", "Nose", "BodyCenter", "lShoulder", "lElbow", "lWrist", "lHip", "lKnee", "lAnkle", "rShoulder", "rElbow", "rWrist", "rHip", "rKnee", "rAnkle", "lEye", "lEar", "rEye", "rEar"]
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

def read_body3DScene(json_file):
    json_data = json.load(open(json_file, 'r'))
    people = json_data['bodies']
    body_0_joints = people[0]["joints19"]
    body_1_joints = people[1]["joints19"]
    return body_0_joints, body_1_joints

# target path using cmu
#data_root = '/Users/lanlan/Downloads/you2me/cmu'
data_root = '/Users/lanlan/Downloads/'
target_path = '1-catch1'
ground_truth = os.path.join(data_root, target_path + '/synchronized/gt-skeletons')
json_files_list = glob.glob(ground_truth + '/*.json')

if not os.path.exists(osp.join(data_root, target_path, 'blender')):
    os.mkdir(osp.join(data_root, target_path, 'blender'))
    
order_num = []
for json_file in json_files_list:
    json_file_name = json_file.split('/')[-1]
    order_num.append(int(json_file_name.split('_')[-1].split('.')[0]))
# reorder and get index
new_index = np.argsort(order_num)
json_files_list = np.array(json_files_list)[new_index]

for json_file in json_files_list:
    json_file_name = json_file.split('/')[-1]
    order_num = (int(json_file_name.split('_')[-1].split('.')[0]))

    interactee_kp,ego_kp = read_body3DScene(json_file)
    interactee_kp = np.array(interactee_kp).reshape(19,4)
    interactee_kp = interactee_kp[:,:3]
    interactee_kp = - interactee_kp
    interactee_kp = interactee_kp.tolist()

    ego_kp = np.array(ego_kp).reshape(19,4)
    ego_kp = ego_kp[:,:3]
    ego_kp = -ego_kp
    ego_kp = ego_kp.tolist()
    
    for i in range(len(people_joints)):
        bpy.context.scene.objects["1" + people_joints[i]].location = [x / 100 for x in interactee_kp[i]]
        bpy.context.scene.objects["2" + people_joints[i]].location = [x / 100 for x in ego_kp[i]]
        
    bpy.context.scene.render.filepath = osp.join(data_root, target_path,'blender',str(order_num).zfill(4)+'.png')
    bpy.ops.render.render(write_still=True)