import copy
import json
import numpy as np
import os
import os.path as osp
import PIL.Image as pil_img
from PIL import ImageDraw
from tqdm import tqdm
import cv2
import argparse

parser = argparse.ArgumentParser(description='Vis openpose 2D joints.')
parser.add_argument("--recording_root", default='/media/qimaqi/Alexander/you2me/kinect')  # 'C:/Users/siwei/Desktop/record_20210907'
parser.add_argument("--recording_name", default='catch37')
parser.add_argument("--keypoints_folder_name", default='keypoints_vis')


args = parser.parse_args()

if __name__ == '__main__':
    keypoint_names = [name for name in os.listdir(osp.join(args.recording_root, args.recording_name, 'features/openpose/output_json/')) if name.endswith('_keypoints.json')]
    keypoint_names = sorted(keypoint_names)
    # print('keypoint_names',keypoint_names)
    # first_frame_id = int(keypoint_names[0][6:11])
    # if args.keypoints_folder_name == 'keypoints':
    # keypoint_names = keypoint_names[args.start_frame - first_frame_id: args.end_frame - first_frame_id + 1]

    # visualize 2d joints
    hand_joint_idx = [2, 4, 5, 8, 9, 12, 13, 16, 17, 20]  # vis 2 joints for each finger (end / tip)
    output_img_path = osp.join(args.recording_root, args.recording_name, 'features','openpose', '{}_img'.format(args.keypoints_folder_name))
    if not osp.exists(output_img_path):
        os.mkdir(output_img_path)
    print('process view {}...'.format(args.recording_name))
    # for cur_view in ['master', 'sub_1', 'sub_2']:
    for keypoint_name in tqdm(keypoint_names):
        keypoint_fn = osp.join(args.recording_root, args.recording_name,'features/openpose/output_json/', keypoint_name)
        if not os.path.exists(keypoint_fn):  # in case this frame is not captured
            continue
        with open(keypoint_fn) as keypoint_file:
            data = json.load(keypoint_file)   # data: dict, key: version/people
            data_reorder = copy.deepcopy(data)
            
        img_n = keypoint_name.split('_')[0]
        # print('img_n',img_n)
        img_fn = osp.join(args.recording_root, args.recording_name, 'synchronized','frames', img_n+'.jpg')
        cur_img = cv2.imread(img_fn)
        cur_img = cur_img[:, :, ::-1]
        # cur_img = cv2.resize(src=cur_img, dsize=(int(1920/args.scale), int(1080/args.scale)), interpolation=cv2.INTER_AREA)  # resolution/4
        

        num_people = len(data['people'])
        if num_people != 1:
            print('{} people detected for {}!'.format(len(data['people']), img_n))
        output_img = pil_img.fromarray(cur_img)
        draw = ImageDraw.Draw(output_img)

        cur_frame_body_keypoints = []
        cur_frame_hand_keypoints = []

        for idx, cur_data in enumerate(data['people']):
                body_keypoint = np.array(cur_data['pose_keypoints_2d'], dtype=np.float32).reshape([-1, 3])  # [25, 3]
                # lhand_keypoint = np.array(cur_data['hand_left_keypoints_2d'], dtype=np.float32).reshape([-1, 3])  # [25, 3]
                # rhand_keypoint = np.array(cur_data['hand_right_keypoints_2d'], dtype=np.float32).reshape([-1, 3])  # [25, 3]
                # hand_keypoint = np.concatenate([lhand_keypoint[hand_joint_idx, :], rhand_keypoint[hand_joint_idx, :]], axis=0)  # [20, 3]
                cur_frame_body_keypoints.append(body_keypoint)
                # cur_frame_hand_keypoints.append(hand_keypoint)

        if num_people >= 1:
            for k in range(len(cur_frame_body_keypoints[0])):
                # if k in [16, 18, 5, 6, 7, 12, 13, 14, 19, 20, 21]:  # left body
                #     fill_color = (255, 0, 0, 0)
                # else:
                #     fill_color = (255, 255, 0, 0)
                draw.ellipse((cur_frame_body_keypoints[0][k][0], cur_frame_body_keypoints[0][k][1] ,
                                cur_frame_body_keypoints[0][k][0], cur_frame_body_keypoints[0][k][1]),
                                fill=(255, 0, 0, 0))  # red: idx 0 in openpose detection
            # for k in range(len(cur_frame_hand_keypoints[0])):
            #     draw.ellipse((cur_frame_hand_keypoints[0][k][0] , cur_frame_hand_keypoints[0][k][1] ,
            #                     cur_frame_hand_keypoints[0][k][0] , cur_frame_hand_keypoints[0][k][1] ),
            #                     fill=(255, 0, 0, 0))  # red: idx 0 in openpose detection
            line_joint_indexs =[[0,1], [1,2],[1,5], [1,8], [2,3], [3,4], [5,6], [6,7], [8,9], [8,12], [9,10], [10,11], [12,13], [13,14]]

            # drawing line
            for index_pair in line_joint_indexs:
                # if the point is 0,0 or not
                if cur_frame_body_keypoints[0][index_pair[0]][0]!=0 and cur_frame_body_keypoints[0][index_pair[0]][1]!=0 and cur_frame_body_keypoints[0][index_pair[1]][0]!=0 and cur_frame_body_keypoints[0][index_pair[1]][1]!=0:                   
                    draw.line(xy = [cur_frame_body_keypoints[0][index_pair[0]][0],cur_frame_body_keypoints[0][index_pair[0]][1],
                                    cur_frame_body_keypoints[0][index_pair[1]][0],cur_frame_body_keypoints[0][index_pair[1]][1] ],
                                    fill=(255, 0, 0, 0))
        if num_people >= 2:
            for k in range(len(cur_frame_body_keypoints[0])):
                draw.ellipse((cur_frame_body_keypoints[1][k][0] , cur_frame_body_keypoints[1][k][1] ,
                                    cur_frame_body_keypoints[1][k][0] , cur_frame_body_keypoints[1][k][1] ),
                                    fill=(0, 0, 255, 0))  # blue: idx 1
            # for k in range(len(cur_frame_hand_keypoints[0])):
            #     draw.ellipse((cur_frame_hand_keypoints[1][k][0]  , cur_frame_hand_keypoints[1][k][1],
            #                         cur_frame_hand_keypoints[1][k][0] , cur_frame_hand_keypoints[1][k][1] ),
            #                         fill=(0, 0, 255, 0))  # blue: idx 1

            line_joint_indexs =[[0,1], [1,2],[1,5], [1,8], [2,3], [3,4], [5,6], [6,7], [8,9], [8,12], [9,10], [10,11], [12,13], [13,14]]

            
            if cur_frame_body_keypoints[1][index_pair[0]][0]!=0 and cur_frame_body_keypoints[1][index_pair[0]][1]!=0 and cur_frame_body_keypoints[1][index_pair[1]][0]!=0 and cur_frame_body_keypoints[1][index_pair[1]][1]!=0:
                draw.line(xy = [cur_frame_body_keypoints[1][index_pair[0]][0],cur_frame_body_keypoints[1][index_pair[0]][1],
                                cur_frame_body_keypoints[1][index_pair[1]][0],cur_frame_body_keypoints[1][index_pair[1]][1] ],
                                fill=(0, 0, 255, 0))

        save_path = osp.join(output_img_path, img_n + '.jpg')
        output_img.save(save_path)

