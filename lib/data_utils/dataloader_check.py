import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

import joblib


db_path = '/Users/qima/Downloads/Klasse/Virtual Humans/dataset/preprocessed_data/'
db_file = osp.join(db_path, 'you2me_train_db_kinect.pt')
db = joblib.load(db_file) 
joints3D_np = db['joints3D']
print(np.shape(joints3D_np))
print('len(joints3D_np[0,25:39,:])',len(joints3D_np[0,25:40,:]))

print('joints3D_np example', joints3D_np[0,25:40,:])

fig = plt.figure(figsize=plt.figaspect(0.5))
ax2 = fig.add_subplot(2, 1, 1, projection='3d') 
ax2.scatter(joints3D_np[:,:-1,0],joints3D_np[:,:-1,1],joints3D_np[:,:-1,2])