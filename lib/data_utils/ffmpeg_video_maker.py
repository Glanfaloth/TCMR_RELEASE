
# import ffmpeg
# (
#     ffmpeg
#     .input('/tmp/amz_kpi/filter_result/image/*.png', framerate = 1)
#     .output('moive.mp4')
#     .run()
# )

import cv2
import os
import tqdm
import numpy as np
frame_path = '/media/qimaqi/Alexander/you2me/kinect/sport57/synchronized/frames'
# '/media/qimaqi/My Passport/record_20220315/recording_20220315_s1_06_qi_cuixi/master/color_img/'#'/tmp/amz_kpi/kpi_video'# '/tmp/amz_kpi/filter_result/image'
frame_rate=20
output='out.avi'
images = [img for img in os.listdir(frame_path) if img.endswith(".jpg")]
# reformulate the imgs
img_name_list = []
order_num = []
for img_i in images:
    # img_name = img_i.split('/')[-1]
    order_num.append(int(img_i.split('x')[-1].split('.')[0]))
# reorder and get index
new_index = np.argsort(order_num)
img_name_list = np.array(images)[new_index]
print('img_name_list',img_name_list)
# for img_name in images:
#     img_name_list.append(img_name)
# img_name_list.sort()
frame = cv2.imread(os.path.join(frame_path, images[0]))
height, width, _ = frame.shape
# fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
save_path = os.path.join(frame_path, output)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video = cv2.VideoWriter(save_path, fourcc, frame_rate, (width, height),isColor =True)
count = 0
for img_name in tqdm.tqdm(img_name_list):
    if count > 200:
        video.write(cv2.imread(os.path.join(frame_path, img_name)))
    
    count+=1
cv2.destroyAllWindows()
video.release()
