import os
import cv2
import glob
import h5py
import json
import joblib
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import scipy.io as sio

import sys
sys.path.append('.')

from lib.models import spin
from lib.core.config import YOU2ME_DIR, TCMR_DB_DIR
from lib.utils.utils import tqdm_enumerate
from lib.data_utils._kp_utils import convert_kps
from lib.data_utils._img_utils import get_bbox_from_kp2d
from lib.data_utils._feature_extractor import extract_features
from lib.utils.smooth_bbox import get_smooth_bbox_params, get_all_bbox_params

from lib.data_utils._occ_utils import load_occluders

VIS_THRESH = 0.3
# '11-hand2', '12-hand3', '3-catch3'
cmu_train_list = ['1-catch1', '10-hand1',  
                '13-sports1', '14-sports2', 
                '2-catch2',
                '4-convo1', '5-convo2', '6-convo3', '7-convo4', '8-convo5', '9-convo6']

kinect_train_list = ['catch36', 'catch37', 'catch39', 'catch40', 'catch41', 'catch42', 'catch55', 
'convo43', 'convo46', 'convo47', 'convo53', 'convo54', 'convo59', 
'patty1', 'patty2', 'patty26', 'patty27', 'patty28', 'patty30', 'patty31', 'patty32', 'patty34', 'patty35', 'patty5', 
'sport56', 'sport57', 'sport58']

def read_body3DScene(json_file):
    json_data = json.load(open(json_file, 'r'))
    people = json_data['bodies']
    body_0_joints = people[0]["joints19"]
    body_1_joints = people[1]["joints19"]
    return body_0_joints, body_1_joints

def read_openpose(json_file): # gt_part
        # get only the arms/legs joints
    op_to_12 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7]
    # read the openpose detection
    json_data = json.load(open(json_file, 'r'))
    people = json_data['people']
    if len(people) == 0:
        # no openpose detection
        keyp25 = np.zeros([25,3])
    else:
        # size of person in pixels
        # TODO scale of person
        #scale = max(max(gt_part[:,0])-min(gt_part[:,0]),max(gt_part[:,1])-min(gt_part[:,1]))
        # go through all people and find a match
        dist_conf = np.inf*np.ones(len(people))
        for i, person in enumerate(people):
            # openpose keypoints
            op_keyp25 = np.reshape(person['pose_keypoints_2d'], [25,3])
            op_keyp12 = op_keyp25[op_to_12, :2]
            op_conf12 = op_keyp25[op_to_12, 2:3] 
            # all the relevant joints should be detected
            if min(op_conf12) > 0:
                # weighted distance of keypoints
                # TODO try just get the closest one
                # print('op_conf12',op_conf12,np.shape(op_conf12))
                # print('op_keyp12',op_keyp12,np.shape(op_keyp12))
                dist_conf[i] = np.mean(op_conf12) # select most high conf one
                # np.mean(np.sum(op_conf12*op_keyp12), axis=1) # *(op_keyp12 - gt_part[:12, :2]
        # closest match
        # There maybe many matches and here we only wnat the cloest
        p_sel = np.argmin(dist_conf)
        # the exact threshold is not super important but these are the values we used
        thresh = 0
        # dataset-specific thresholding based on pixel size of person
        #if min(dist_conf)/scale > 0.1 and min(dist_conf) < thresh:
        #    keyp25 = np.zeros([25,3])
        #else:
        keyp25 = np.reshape(people[p_sel]['pose_keypoints_2d'], [25,3])
    return keyp25


def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i * 7 + 5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i * 7 + 6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3, :3]
        T = RT[:3, 3] / 1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts


def read_train_data(dataset_path, device,data_type, debug=False):
    h, w = 227, 227
    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'egojoints3D':[],
        'bbox': [],
        'img_name': [],
        'features': [],
    }

    # occluders = load_occluders('./data/VOC2012')

    model = spin.get_pretrained_hmr(device)

    # training data
    # two types of data
    # read cmu first
    if data_type == 'cmu':
        for seq_num, vid_i in enumerate(cmu_train_list):
            print("vid_i: ", vid_i)
            print("seq_num: ", seq_num)
            imgs_path = os.path.join(dataset_path,
                                        'cmu',
                                        vid_i,
                                        'synchronized',
                                        'frames')
            pattern = os.path.join(imgs_path, '*.jpg')
            img_list = sorted(glob.glob(pattern))
            # reset the sequence
            # save all the seq name
            order_num = []
            for img_i in img_list:
                img_name = img_i.split('/')[-1]
                order_num.append(int(img_name.split('x')[-1].split('.')[0]))
            # reorder and get index
            new_index = np.argsort(order_num)
            img_list = np.array(img_list)[new_index]
            # print('img_list',img_list)
            # TODO image list have sequnce issues
            num_frames = len(img_list)

            # j3ds = np.zeros((num_frames, 49, 3), dtype=np.float32)
            # j2ds = np.zeros((num_frames, 49, 3), dtype=np.float32)
            vid_used_frames = []
            vid_used_joints = []
            vid_used_bbox = []
            vid_segments = []
            print('imgs_path',imgs_path)
            openpose_path = os.path.join(dataset_path,
                                        'cmu',
                                        vid_i,
                                        'features',
                                        'openpose',
                                        'output_json')
            gt_skeletons_path = os.path.join(dataset_path,
                                        'cmu',
                                        vid_i,
                                        'synchronized',
                                        'gt-skeletons')

            for i, img_i in tqdm_enumerate(img_list):
                img_name = img_i.split('/')[-1]
                openpose_name = img_name.split('.')[0] + '_keypoints.json'
                openpose_i = os.path.join(openpose_path,openpose_name)
                # try read you2me keypiont
                joints_2d_raw = read_openpose(openpose_i).reshape(1, 25, 3)
                if np.sum(joints_2d_raw.reshape(-1,1))==0:
                    print('no joints img_name',i, img_name)
                # joints_2d_raw[:,:,2::3] = len(joints_2d_raw[:,:,2::3][2::3])*[1] # set confidence to 1
                # key2djnts[2::3] = len(key2djnts[2::3])*[1]
                # print('joints_2d',joints_2d_raw)
                joints_2d = convert_kps(joints_2d_raw, "you2me2d",  "spin").reshape((-1,3))
                # print('joints_2d',np.shape(joints_2d))
                # TODO what is the difference between openpose joint keypoint and convert one
                joints_3d_name = 'body3DScene_' + img_name.split('x')[-1].split('.')[0]
                interact_joints_3d, ego_1_joints_3d =  read_body3DScene(os.path.join(gt_skeletons_path,joints_3d_name+'.json'))
                joints_3d_raw = np.reshape(interact_joints_3d, (1, 19, 4)) / 1000 # TODO why divide 1000
                joints_3d_raw = joints_3d_raw[:,:,:3]
                joints_3d = convert_kps(joints_3d_raw, "you2me_cmu_3d", "spin").reshape((-1,3))
                joints_3d_ego_raw = np.reshape(ego_1_joints_3d, (1, 19, 4)) / 1000 # TODO why divide 1000
                joints_3d_ego_raw = joints_3d_ego_raw[:,:,:3]
                joints_3d_ego = convert_kps(joints_3d_ego_raw, "you2me_cmu_3d", "spin").reshape((-1,3))
                # print('joints_3d',joints_3d)
                # if joints_2d:
                #     bbox = get_bbox_from_kp2d(joints_2d[~np.all(joints_2d == 0, axis=1)]).reshape(4)
                bbox = np.array([113, 113, w, h])  # shape = (4,N)
                joints_3d = joints_3d - joints_3d[39]  # 4 is the root
                joints_3d_ego = joints_3d_ego - joints_3d_ego[39]  # 4 is the root

                # j3ds[i] = joints_3d
                # j2ds[i] = joints_2d
                # check that all joints are visible
                # joint x loc inside the image
                # since we generate it from openpose so always in
                # TODO other way
                # manual set
                x_in = np.logical_and(joints_2d[:, 0] < w, joints_2d[:, 0] >= 0)
                y_in = np.logical_and(joints_2d[:, 1] < h, joints_2d[:, 1] >= 0)
                ok_pts = np.logical_and(x_in, y_in)
                if np.sum(ok_pts) < joints_2d.shape[0]:
                    print('np.sum(ok_pts)',np.sum(ok_pts))
                    print(' joints_2d.shape[0]', joints_2d.shape[0])
                    print('img_name',img_name)
                    # vid_uniq_id = "_".join(vid_uniq_id.split("_")[:-1])+ "_seg" +\
                    #                     str(int(dataset['vid_name'][-1].split("_")[-1][3:])+1)
                    continue
            # bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(j2ds, vis_thresh=VIS_THRESH, sigma=8)
   
                # TODO video name
                dataset['vid_name'].append(vid_i)
                dataset['frame_id'].append(img_name.split(".")[0])
                dataset['img_name'].append(img_i)
                dataset['joints2D'].append(joints_2d)
                dataset['joints3D'].append(joints_3d)
                dataset['bbox'].append(bbox)
                dataset['egojoints3D'].append(joints_3d_ego)
                vid_segments.append(vid_i)
                vid_used_frames.append(img_i)
                vid_used_joints.append(joints_2d)
                vid_used_bbox.append(bbox)

            vid_segments= np.array(vid_segments)
            ids = np.zeros((len(set(vid_segments))+1))
            ids[-1] = len(vid_used_frames) + 1
            if (np.where(vid_segments[:-1] != vid_segments[1:])[0]).size != 0:
                ids[1:-1] = (np.where(vid_segments[:-1] != vid_segments[1:])[0]) + 1
            
            # debug
            # print('ids',ids) [   0. 3478.]
            for i in tqdm(range(len(set(vid_segments)))):
                features = extract_features(model,device, None, np.array(vid_used_frames)[int(ids[i]):int(ids[i+1])],
                                            vid_used_bbox[int(ids[i]):int((ids[i+1]))],
                                            kp_2d=np.array(vid_used_joints)[int(ids[i]):int(ids[i+1])],
                                            dataset='spin', debug=False, scale=1.0)
                dataset['features'].append(features)

    if data_type == 'kinect':
        for seq_num, vid_i in enumerate(kinect_train_list):
            print("vid_i: ", vid_i)
            print("seq_num: ", seq_num)
            imgs_path = os.path.join(dataset_path,
                                        'kinect',
                                        vid_i,
                                        'synchronized',
                                        'frames')
            pattern = os.path.join(imgs_path, '*.jpg')
            img_list = sorted(glob.glob(pattern))
            # reset the sequence
            # save all the seq name
            order_num = []
            for img_i in img_list:
                img_name = img_i.split('/')[-1]
                order_num.append(int(img_name.split('x')[-1].split('.')[0]))
            # reorder and get index
            new_index = np.argsort(order_num)
            img_list = np.array(img_list)[new_index]
            # print('img_list',img_list)
            # TODO image list have sequnce issues
            num_frames = len(img_list)

            # j3ds = np.zeros((num_frames, 49, 3), dtype=np.float32)
            # j2ds = np.zeros((num_frames, 49, 3), dtype=np.float32)
            vid_used_frames = []
            vid_used_joints = []
            vid_used_bbox = []
            vid_segments = []
            print('imgs_path',imgs_path)
            openpose_path = os.path.join(dataset_path,
                                        'kinect',
                                        vid_i,
                                        'features',
                                        'openpose',
                                        'output_json')
            gt_egopose_path = os.path.join(dataset_path,
                                        'kinect',
                                        vid_i,
                                        'synchronized',
                                        'gt-egopose')
            gt_interactee_path = os.path.join(dataset_path,
                                        'kinect',
                                        vid_i,
                                        'synchronized',
                                        'gt-interactee')

            for i, img_i in tqdm_enumerate(img_list):
                img_name = img_i.split('/')[-1]
                openpose_name = img_name.split('.')[0] + '_keypoints.json'
                openpose_i = os.path.join(openpose_path,openpose_name)
                # try read you2me keypiont
                joints_2d_raw = read_openpose(openpose_i).reshape(1, 25, 3)
                if np.sum(joints_2d_raw.reshape(-1,1))==0:
                    print('no joints img_name',i, img_name)
                # joints_2d_raw[:,:,2::3] = len(joints_2d_raw[:,:,2::3][2::3])*[1] # set confidence to 1
                # key2djnts[2::3] = len(key2djnts[2::3])*[1]
                # print('joints_2d',joints_2d_raw)
                joints_2d = convert_kps(joints_2d_raw, "you2me2d",  "spin").reshape((-1,3))
                # print('joints_2d',np.shape(joints_2d))
                # TODO what is the difference between openpose joint keypoint and convert one
                joints_3d_name = 'body3DScene_' + img_name.split('x')[-1].split('.')[0]
                interact_joints_3d, ego_1_joints_3d =  read_body3DScene(os.path.join(gt_skeletons_path,joints_3d_name+'.json'))
                joints_3d_raw = np.reshape(interact_joints_3d, (1, 19, 4)) / 1000 # TODO why divide 1000
                joints_3d_raw = joints_3d_raw[:,:,:3]
                joints_3d = convert_kps(joints_3d_raw, "you2me_cmu_3d", "spin").reshape((-1,3))
                joints_3d_ego_raw = np.reshape(ego_1_joints_3d, (1, 19, 4)) / 1000 # TODO why divide 1000
                joints_3d_ego_raw = joints_3d_ego_raw[:,:,:3]
                joints_3d_ego = convert_kps(joints_3d_ego_raw, "you2me_cmu_3d", "spin").reshape((-1,3))
                # print('joints_3d',joints_3d)
                # if joints_2d:
                #     bbox = get_bbox_from_kp2d(joints_2d[~np.all(joints_2d == 0, axis=1)]).reshape(4)
                bbox = np.array([113, 113, w, h])  # shape = (4,N)
                joints_3d = joints_3d - joints_3d[39]  # 4 is the root
                joints_3d_ego = joints_3d_ego - joints_3d_ego[39]  # 4 is the root

                # j3ds[i] = joints_3d
                # j2ds[i] = joints_2d
                # check that all joints are visible
                # joint x loc inside the image
                # since we generate it from openpose so always in
                # TODO other way
                # manual set
                x_in = np.logical_and(joints_2d[:, 0] < w, joints_2d[:, 0] >= 0)
                y_in = np.logical_and(joints_2d[:, 1] < h, joints_2d[:, 1] >= 0)
                ok_pts = np.logical_and(x_in, y_in)
                if np.sum(ok_pts) < joints_2d.shape[0]:
                    print('np.sum(ok_pts)',np.sum(ok_pts))
                    print(' joints_2d.shape[0]', joints_2d.shape[0])
                    print('img_name',img_name)
                    # vid_uniq_id = "_".join(vid_uniq_id.split("_")[:-1])+ "_seg" +\
                    #                     str(int(dataset['vid_name'][-1].split("_")[-1][3:])+1)
                    continue
            # bbox_params, time_pt1, time_pt2 = get_smooth_bbox_params(j2ds, vis_thresh=VIS_THRESH, sigma=8)
   
                # TODO video name
                dataset['vid_name'].append(vid_i)
                dataset['frame_id'].append(img_name.split(".")[0])
                dataset['img_name'].append(img_i)
                dataset['joints2D'].append(joints_2d)
                dataset['joints3D'].append(joints_3d)
                dataset['bbox'].append(bbox)
                dataset['egojoints3D'].append(joints_3d_ego)
                vid_segments.append(vid_i)
                vid_used_frames.append(img_i)
                vid_used_joints.append(joints_2d)
                vid_used_bbox.append(bbox)

            vid_segments= np.array(vid_segments)
            ids = np.zeros((len(set(vid_segments))+1))
            ids[-1] = len(vid_used_frames) + 1
            if (np.where(vid_segments[:-1] != vid_segments[1:])[0]).size != 0:
                ids[1:-1] = (np.where(vid_segments[:-1] != vid_segments[1:])[0]) + 1
            
            # debug
            # print('ids',ids) [   0. 3478.]
            for i in tqdm(range(len(set(vid_segments)))):
                features = extract_features(model,device, None, np.array(vid_used_frames)[int(ids[i]):int(ids[i+1])],
                                            vid_used_bbox[int(ids[i]):int((ids[i+1]))],
                                            kp_2d=np.array(vid_used_joints)[int(ids[i]):int(ids[i+1])],
                                            dataset='spin', debug=False, scale=1.0)
                dataset['features'].append(features)


    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
    dataset['features'] = np.concatenate(dataset['features'])

    return dataset

    

 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/you2me')
    parser.add_argument('--device', type=str,choices=['cpu','cuda'],help='way to preprocess', default='cuda')
    parser.add_argument('--data_type', type=str,choices=['cmu','kinect'],help='two kinds of data', default='cmu')
    parser.add_argument('--debug', type=bool, help='debug model', default=False)
    
    args = parser.parse_args()

    dataset = read_train_data(args.dir,args.device, args.data_type, args.debug)
    joblib.dump(dataset, osp.join(TCMR_DB_DIR, 'you2me_train_db.pt'))

    # dataset = read_test_data(args.dir,args.data_type,args.debug)
    # joblib.dump(dataset, osp.join(TCMR_DB_DIR, 'you2me_val_db.pt'))





