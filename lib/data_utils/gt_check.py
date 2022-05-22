import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os.path as osp
import glob

def draw_joints(joints, ax):
	ax.scatter(joints[:,0], joints[:,2], joints[:,1])
	# ax.plot3D(joints[0:2,0], joints[0:2,2], joints[0:2,1])
	# ax.plot3D(joints[3:5,0], joints[3:5,2], joints[3:5,1])
	# ax.plot3D(joints[4:6,0], joints[4:6,2], joints[4:6,1])
	# ax.plot3D(joints[7:9,0], joints[7:9,2], joints[7:9,1])
	# ax.plot3D(joints[8:10,0], joints[8:10,2], joints[8:10,1])
	# ax.plot3D(joints[9:11,0], joints[9:11,2], joints[9:11,1])
	# ax.plot3D([joints[11,0], joints[3,0]], [joints[11,2], joints[3,2]], [joints[11,1], joints[3,1]])
	# ax.plot3D([joints[11,0], joints[7,0]], [joints[11,2], joints[7,2]], [joints[11,1], joints[7,1]])

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.
    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

    
def show_upp(joints):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# ax.set_aspect('equal')
	draw_joints(joints, ax)
	set_axes_equal(ax)
	plt.show()

# import json
# json_file = '/media/qimaqi/Alexander/you2me/cmu/1-catch1/synchronized/gt-skeletons/body3DScene_453.json'
# json_data = json.load(open(json_file, 'r'))
# people = json_data['bodies']
# body_0_joints = people[0]["joints19"]
# body_0_joints = np.array(body_0_joints).reshape(19,4)
# body_1_joints = people[1]["joints19"]
# body_1_joints = np.array(body_1_joints).reshape(19,4)


# show_upp(body_0_joints)

# show_upp(body_1_joints)

# check cmu
json_pth = '/media/qimaqi/Alexander/you2me/cmu/8-convo5/synchronized/gt-skeletons/body3DScene_1.json'

# check kinect
txt_path = '/media/qimaqi/Alexander/you2me/kinect/catch37/synchronized/gt-interactee/pose2_216.txt'

def read_body3DScene(json_file):
    json_data = json.load(open(json_file, 'r'))
    people = json_data['bodies']
    body_0_joints = people[0]["joints19"]
    body_1_joints = people[1]["joints19"]
    return body_0_joints, body_1_joints

##########################################################################
json_dir = osp.dirname(json_pth)
json_files = glob.glob(osp.join(json_dir,'*.json'))

# reorder
order_num = []
for json_i in json_files:
    json_name = json_i.split('/')[-1]
    order_num.append(int(json_name.split('_')[-1].split('.')[0]))
# reorder and get index
new_index = np.argsort(order_num)
jsons_list = np.array(json_files)[new_index]
print('json_dir',jsons_list)
interactee_kp_list = []
ego_kp_kp_list = []
for json_file_i in jsons_list:
    interactee_kp,ego_kp = read_body3DScene(json_file_i)
    interactee_kp = np.reshape(interactee_kp, (1, 19, 4))
    # if np.sum(interactee_kp[:,1,:]*interactee_kp[:,2,:]*interactee_kp[:,2,:])!=0:
    #     height = np.linalg.norm(interactee_kp[:,1,:] - (interactee_kp[:,8,:]+interactee_kp[:,14,:])/2) #interactee_kp[:,2,:])#(
    #     print('height',height)
    #     height_list.append(height)
    interactee_kp_list.append(np.reshape(interactee_kp, (1, 19, 4)))
    ego_kp_kp_list.append(np.reshape(ego_kp, (1, 19, 4)))

interactee_kp_np = np.array(interactee_kp_list).reshape(-1,19,4)
ego_kp_kp_np = np.array(ego_kp_kp_list).reshape(-1,19,4)

pred_hip_distance = np.linalg.norm(interactee_kp_np[:,6,:] - interactee_kp_np[:,12,:],axis=1 )
print("mean pred_hip_distance", np.mean(pred_hip_distance))
# 

# save_name_interact = './video/presentation_material/interact.npy'
# save_name_ego = './video/presentation_material/ego.npy'
# np.save(save_name_interact, interactee_kp_np)
# np.save(save_name_ego,ego_kp_kp_np)
# # keypoints_np = np.array(keypoints_list) 

########################################################################
# txt_dir = osp.dirname(txt_path)
# txt_files = glob.glob(osp.join(txt_dir,'*.txt'))
# print('txt_dir',txt_dir)
# height_list = []
# for txt_file_i in txt_files:
#     file = open(txt_file_i)
#     joints_3d_raw = np.array(file.read().split()).astype(np.float64).reshape(25,3)

#     if np.sum(joints_3d_raw[3,:]*joints_3d_raw[14,:]*joints_3d_raw[18,:])!=0:
#         height = np.linalg.norm(joints_3d_raw[3,:] - (joints_3d_raw[14,:]+joints_3d_raw[18,:])/2) #interactee_kp[:,2,:])#(
#         print('height',height)
#         height_list.append(height)
#     # keypoints_list.append(np.reshape(interactee_kp, (1, 19, 4)))
#     # keypoints_list.append(np.reshape(ego_kp, (1, 19, 4)))

# # keypoints_np = np.array(keypoints_list) 

# print('interactee_kp height_list',np.shape(height_list))
# print('range',np.average(height_list))

# print('interactee_kp height_list',np.shape(height_list))
# print('range',np.average(height_list))

# joints_3d_raw = np.reshape(interactee_kp, (1, 19, 4)) # / 1000
# fig = plt.figure(figsize=plt.figaspect(0.5))
# ax2 = fig.add_subplot(1, 1, 1, projection='3d')
# color_list = np.array([10]+[0]*18)
# ax2.scatter(joints_3d_raw[:,:,0],joints_3d_raw[:,:,1],joints_3d_raw[:,:,2], c=color_list, s = 30)
# set_axes_equal(ax2)
# plt.show()

# ax2.set_box_aspect
# ax2.set_box_aspect((1, 1, 1))
# # with open(txt_path, 'r') as f:
# file = open(txt_path)
# joints_3d_raw = np.array(file.read().split()).astype(np.float64).reshape(25,3)
# print("h",np.shape(joints_3d_raw))
# color_list = np.array([10]+[0]*24)
# fig = plt.figure(figsize=plt.figaspect(0.5))
# ax2 = fig.add_subplot(1, 1, 1, projection='3d')
# color_list = np.array([0,30,0,0]+[0]*16 + [30,0,0,0,0])
# ax2.scatter(joints_3d_raw[:,0],joints_3d_raw[:,1],joints_3d_raw[:,2], c=color_list, s = 30)
# set_axes_equal(ax2)
# plt.show()
# show_upp(h)
# print('h',len(h))

