import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
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
target_path = '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/you2me_test_output/you2me_output'
gt_path = osp.join(target_path,'gt.npy')
pred_path = osp.join(target_path,'pred.npy')

pred_np= np.load(gt_path)
gt_np= np.load(pred_path)

print('shape of gt',np.shape(gt_np))
print('shape of pred',np.shape(pred_np))

gt_sub_np = gt_np[:, 25:38, :]

color_list  = np.array([0]*12 + [10] )

# length = len(gt_np)
length = 2
for ii in range(length):
    # print('pred_sub_np[ii,:,0]',pred_np[ii,39,:])
    # print('gt_np]',gt_np[ii,-1,:])
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax2 = fig.add_subplot(2, 1, 1, projection='3d') 
    plt.title('ground truth')
    ax3 = fig.add_subplot(2, 1, 2, projection='3d')
    
    plt.title('prediction')
    ax2.scatter(gt_sub_np[ii,:,0],gt_sub_np[ii,:,1],gt_sub_np[ii,:,2], c=color_list)
    ax3.scatter(pred_np[ii,:,0],pred_np[ii,:,1],pred_np[ii,:,2],c = np.array([0]*12 + [10,10] ) )
    plt.show()
#plt.tight_layout(True)