import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def show_upp(joints):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# ax.set_aspect('equal')
	draw_joints(joints, ax)
	set_axes_equal(ax)
	plt.show()

import json
json_file = '/home/glanfaloth/TCMR_RELEASE/data/you2me/cmu/1-catch1/synchronized/gt-skeletons/body3DScene_453.json'
json_data = json.load(open(json_file, 'r'))
people = json_data['bodies']
body_0_joints = people[0]["joints19"]
body_0_joints = np.array(body_0_joints).reshape(19,4)
body_1_joints = people[1]["joints19"]
body_1_joints = np.array(body_1_joints).reshape(19,4)


show_upp(body_0_joints)

show_upp(body_1_joints)