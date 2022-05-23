import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import time
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

def get_common_joint_names():
    return [
        "rankle",    # 0  "lankle",    # 0
        "rknee",     # 1  "lknee",     # 1
        "rhip",      # 2  "lhip",      # 2
        "lhip",      # 3  "rhip",      # 3
        "lknee",     # 4  "rknee",     # 4
        "lankle",    # 5  "rankle",    # 5
        "rwrist",    # 6  "lwrist",    # 6
        "relbow",    # 7  "lelbow",    # 7
        "rshoulder", # 8  "lshoulder", # 8
        "lshoulder", # 9  "rshoulder", # 9
        "lelbow",    # 10  "relbow",    # 10
        "lwrist",    # 11  "rwrist",    # 11
        "neck",      # 12  "neck",      # 12
        "headtop",   # 13  "headtop",   # 13
    ]

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
# target path
#  '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/non_reg_kinect/you2me_output
target_path = '/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/outputs/cmu/final/repr_table6_you2me_cmu_model_output'
#'/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/kinect/repr_table6_you2me_kinect_model_output'
#'/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/cmu/repr_table6_you2me_cmu_model_output/'
# '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/kinect/repr_table6_you2me_kinect_model_output'
# '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/cmu/repr_table6_you2me_cmu_model_output/'
# '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/rui_wang/you2me_output_kinect_new_regressor'# '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/you2me_test_output/you2me_output'
gt_path = osp.join(target_path,'gt.npy')
pred_path = osp.join(target_path,'pred.npy')

gt_np= np.load(gt_path)
pred_np= np.load(pred_path)

# find the range of pred_np
print(np.min(pred_np),np.max(pred_np))
print('shape of gt',np.shape(gt_np))
print('shape of pred',np.shape(pred_np))

gt_sub_np = gt_np[:, 25:39, :]

pred_np = pred_np[:, 25:39, :]

color_list  = np.array([0]*12 + [10,10] )
def close_event():
    plt.close() #timer calls this function after 3 seconds and closes the window 


# length = len(gt_np)
length = 1
for ii in range(length):
    # print('pred_sub_np[ii,:,0]',pred_np[ii,39,:])
    # print('gt_np]',gt_np[ii,-1,:])
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
    # timer.add_callback(close_event)
    ax2 = fig.add_subplot(2, 1, 1, projection='3d') 
    plt.title('ground truth')
    ax3 = fig.add_subplot(2, 1, 2, projection='3d')

    rcolor = get_colors()['red'].tolist()
    pcolor = get_colors()['green'].tolist()
    lcolor = get_colors()['blue'].tolist()

    common_lr = [0,0,1,1,0,0,0,0,1,0,0,1,1,1,0]

    plt.title('prediction')
    print('gt_sub_np',gt_sub_np[ii,:,:])
    print('pred_np',pred_np[ii,:,:])
    ax2.scatter(gt_sub_np[ii,:,0],gt_sub_np[ii,:,1],gt_sub_np[ii,:,2], c=color_list)
    ax3.scatter(pred_np[ii,:,0],pred_np[ii,:,1],pred_np[ii,:,2],c = color_list)
    skeleton = get_common_skeleton()
    for i,(j1,j2) in enumerate(skeleton):
        # if gt_sub_np[ii, j1, 2] * gt_sub_np[ii, j2, 2] >= 0: # if visible
        color = np.array(rcolor) if common_lr[i] == 0 else np.array(lcolor)
        line_x = np.array([gt_sub_np[ii,j1,0],gt_sub_np[ii,j2,0]])
        line_y = np.array([gt_sub_np[ii,j1,1],gt_sub_np[ii,j2,1]])
        line_z =  np.array([gt_sub_np[ii,j1,2],gt_sub_np[ii,j2,2]])
        ax2.plot3D(line_x, line_y, line_z, c = color/255)

    for i,(j1,j2) in enumerate(skeleton):
        # if gt_sub_np[ii, j1, 2] * gt_sub_np[ii, j2, 2] >= 0: # if visible
        color = np.array(rcolor) if common_lr[i] == 0 else np.array(lcolor)
        line_x = np.array([pred_np[ii,j1,0],pred_np[ii,j2,0]])
        line_y = np.array([pred_np[ii,j1,1],pred_np[ii,j2,1]])
        line_z =  np.array([pred_np[ii,j1,2],pred_np[ii,j2,2]])
        ax3.plot3D(line_x, line_y, line_z, c = color/255)

    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    ax3.set_xlabel('X axis')
    ax3.set_ylabel('Y axis')
    ax3.set_zlabel('Z axis')
    # ax2.view_init(azim=-90, elev=110) # kinect
    # ax3.view_init(azim=-90, elev=110)
    # ax2.view_init(azim=-90, elev=-80) # cmu
    # ax3.view_init(azim=-90, elev=-80)  # kinect -90 110
    # ax2.azim = 180
    set_axes_equal(ax2)
    set_axes_equal(ax3)
    # timer.start()
    plt.show()
    # plt.pause(3)

#plt.tight_layout(True)