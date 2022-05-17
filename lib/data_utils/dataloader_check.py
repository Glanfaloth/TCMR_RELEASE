import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

import joblib
import torch

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

db_path = '/Users/qima/Downloads/Klasse/Virtual Humans/dataset/norm_preprocessed_data/'
db_file = osp.join(db_path, 'you2me_val_db_kinect.pt')
db = joblib.load(db_file) 
joints3D_np = db['joints3D']
print(db.keys())
# print(np.shape(joints3D_np))
end_frame = 30
feature_torch = db['features']




# for i in range(end_frame):#len(joints3D_np):

#     # print('len(joints3D_np[0,25:39,:])',len(joints3D_np[0,25:40,:]))

#     # print('joints3D_np example', joints3D_np[0,25:40,:])

#     fig = plt.figure(figsize=plt.figaspect(0.5))
#     ax2 = fig.add_subplot(2, 1, 1, projection='3d') 
#     ax2.scatter(joints3D_np[i,25:39,0],joints3D_np[i,25:39,1],joints3D_np[i,25:39,2])
#     set_axes_equal(ax2)
#     plt.show()


