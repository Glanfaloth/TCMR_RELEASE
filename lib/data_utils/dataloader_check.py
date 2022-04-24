import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp

# target path
target_path = '/Users/qima/Downloads/Klasse/Virtual Humans/TCMR_RELEASE/outputs/you2me_test_output/you2me_output'
gt_path = osp.join(target_path,'gt.npy')
pred_path = osp.join(target_path,'pred.npy')

gt_np = np.load(gt_path)
pred_np = np.load(pred_path)

print('shape of gt',np.shape(gt_np))
print('shape of pred',np.shape(pred_np))

pred_sub_np = pred_np[:, 25:39, :]



# length = len(gt_np)
length = 2
for ii in range(length):
    print('pred_sub_np[ii,:,0]',pred_np[ii,38,:])
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax2 = fig.add_subplot(2, 1, 1, projection='3d') 
    plt.title('prediction')
    ax3 = fig.add_subplot(2, 1, 2, projection='3d')
    plt.title('ground truth')
    ax2.scatter(pred_sub_np[ii,:,0],pred_sub_np[ii,:,1],pred_sub_np[ii,:,2])
    ax3.scatter(gt_np[ii,:,0],gt_np[ii,:,1],gt_np[ii,:,2])
    plt.show()
#plt.tight_layout(True)