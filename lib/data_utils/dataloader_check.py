import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
import joblib


db_path = '/Users/qima/Downloads/Klasse/Virtual Humans/dataset/preprocessed_data/'
db_file = osp.join(db_path, 'you2me_train_db_kinect.pt')
db = joblib.load(db_file) 
joints3D_np = db['joints3D']
print(np.shape(joints3D_np))
print('len(joints3D_np[0,25:39,:])',len(joints3D_np[0,25:40,:]))

print('joints3D_np example', joints3D_np[0,25:40,:])
