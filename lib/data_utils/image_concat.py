from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
from mpl_toolkits.mplot3d import Axes3D
import json
import time

import argparse
import sys
import time
from tqdm import tqdm
import open3d as o3d
sys.path.append('/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/lib')
from utils.eval_utils import batch_compute_similarity_transform_torch
import torch
import os




parser = argparse.ArgumentParser(description='Vis ego joints.')
parser.add_argument("--start_frame", default=0, type=int)
parser.add_argument("--end_frame", default=-1, type=int)
parser.add_argument("--vis_seq", default='')
args = parser.parse_args()

path_1 = '/media/qimaqi/Alexander/you2me/kinect/xxxxxxxx/synchronized/frames'
path_2 = '/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/video/xxxxxxxx/gt'
path_3 = '/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/video/xxxxxxxx/pred'
save_path = '/home/qimaqi/workspace_ra/VH_group/TCMR_RELEASE/video/xxxxxxxx/concat'
vis_seq = args.vis_seq
path_1 = path_1.replace('xxxxxxxx',vis_seq)
path_2 = path_2.replace('xxxxxxxx',vis_seq)
path_3 = path_3.replace('xxxxxxxx',vis_seq)
save_path = save_path.replace('xxxxxxxx',vis_seq)

if not osp.exists(save_path):
    os.mkdir(save_path)
for i in tqdm(range(args.start_frame+1, args.end_frame)):
    input_image_path = osp.join(path_1,'imxx'+str(i) +'.jpg')
    gt_image_path = osp.join(path_2, str(i).zfill(4)+'.png')
    pred_image_path = osp.join(path_3, str(i).zfill(4)+'.png')
    input_image = Image.open(input_image_path)
    gt_image = Image.open(gt_image_path)
    pred_image = Image.open(pred_image_path)
    width,height = gt_image.size
    gt_image = gt_image.crop((400,0,1400,height))
    pred_image = pred_image.crop((400,0,1400,height))
    im_file_list = [input_image, gt_image,pred_image]
    # im_list = [Image.open(fn) for fn in im_file_list]
    im_output = []
    for image_i in im_file_list:
        resize_img = image_i.resize((600,600),Image.BILINEAR)
        im_output.append(resize_img)
    
    concat_img = Image.new(im_output[0].mode,(600*3, 600))

    for index,im in enumerate(im_output):
        concat_img.paste(im, box = (index*600,0))
    concat_img.save(osp.join(save_path, str(i).zfill(4)+'.jpg'))

