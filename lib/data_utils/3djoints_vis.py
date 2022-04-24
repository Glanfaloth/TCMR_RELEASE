import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp

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