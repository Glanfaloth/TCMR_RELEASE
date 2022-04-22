import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp

# target path
target_path = '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/you2me_test_output/you2me_output'
gt_path = osp.join(target_path,'gt.npy')
pred_path = osp.join(target_path,'pred.npy')

gt_np = np.fromfile(gt_path)
pred_np = np.fromfile(pred_path)

print('shape of gt',np.shape(gt_np))
print('shape of pred',np.shape(pred_np))