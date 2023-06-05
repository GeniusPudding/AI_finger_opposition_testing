import cv2
import mediapipe as mp
import sys
import pickle
import numpy as np
import random
# 讀取你的 MP4 檔案
import os

#function: 讀取一個file_paths list裡面的所有影片，並且將影片中的手部關鍵點資料存成一個data_list
#input: file_paths list
#output: data_list list
def read_vedio(file_path,folder_label):
    data_list = []
    cap = cv2.VideoCapture(file_path)
    print(f'Processing {file_path}...')
    # 初始化 MediaPipe Hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        joint_data = []
        #把資料擴充成每30個frame的序列作為一筆label data，然後每個影片都用這樣拆分
        #例如100個frame長的影片就可以這樣拆解成71筆資料，每筆資料30個frame

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # 將影像的色彩空間從 BGR 轉換為 RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 使用 MediaPipe Hands 進行手部關節識別
            results = hands.process(image)
            # 檢查結果
            if results.multi_hand_landmarks:
                frame_joint_data = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        frame_joint_data.extend([lm.x, lm.y, lm.z])
                joint_data.append(frame_joint_data[:63])
        # input(f'joint_data: {joint_data}')
        cap.release()

    # 將這個影片的數據和標籤添加到 data_list 中
    d = np.array(joint_data)
    print(f'd.shape: {d.shape}')
    if d.shape[0] < frame_len or len(d.shape) < 2:
        print(f'異常資料: {file_path}, d:{d}')
        return []

    l = d.shape[0]
    for i in range(l - frame_len + 1):
        d30 = np.array(joint_data[i:i+frame_len])
        data_list.append({
            'data': d30,
            'label': folder_label,
        })
    return data_list

def read_dir(file_paths):
    data_list = []
    for file_path,folder_label in file_paths:
        data_list += read_vedio(file_path,folder_label)
    return data_list

folders = [
    # {'name': '正常右手', 'label': 0},
    # {'name': '正常左手', 'label': 0},
    # {'name': '異常右手', 'label': 1},
    # {'name': '異常左手', 'label': 1},
    {'name': 'normal', 'label': 0},
    {'name': 'abnormal/0', 'label': 1},
    {'name': 'abnormal/1', 'label': 1},
    {'name': 'abnormal/2', 'label': 1},
    {'name': 'abnormal/3', 'label': 1},
]
file_paths = []
frame_len = 30
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

for folder in folders:
    folder_name = folder['name']
    folder_label = folder['label']
    if not os.path.isdir(folder_name):
        continue
    # 獲取這個資料夾中所有的 .mp4 檔案
    # input(f'folder_label: {folder_label}')
    file_paths += [(os.path.join(*folder_name.split('/'), f),folder_label) for f in os.listdir(folder_name) if f.endswith('.mp4') or f.endswith('.MOV') or f.endswith('.avi')]
    # input(f'file_paths: {file_paths}')

# 隨機打亂順序
random.shuffle(file_paths)
partition = int(0.7*len(file_paths))
partition2 = int(0.8*len(file_paths))
print(f'file_paths: {file_paths}, len(file_paths): {len(file_paths)}') 
train_list = read_dir(file_paths[:partition])
valid_list = read_dir(file_paths[partition:partition2])
test_list = read_dir(file_paths[partition2:])
# print(f'test_list: {test_list}')
# print(f'test_list: {test_list}')
np.save('train_list.npy', train_list)
np.save('valid_list.npy', valid_list)
np.save('test_list.npy', test_list)

print("Part 2")

folders = [ #醫學上的0分對映到程式上的3分，兩者順序相反，反之醫學上的3分對映到程式上的0分
    {'name': 'abnormal/0', 'label': 3},
    {'name': 'abnormal/1', 'label': 2},
    {'name': 'abnormal/2', 'label': 1},
    {'name': 'abnormal/3', 'label': 0},
]

file_paths = []
for folder in folders:
    folder_name = folder['name']
    folder_label = folder['label']
    if not os.path.isdir(folder_name):
        continue
    file_paths += [(os.path.join(*folder_name.split('/'), f),folder_label) for f in os.listdir(folder_name) if f.endswith('.mp4') or f.endswith('.MOV') or f.endswith('.avi')]
# 隨機打亂順序
random.shuffle(file_paths)
partition = int(0.7*len(file_paths))
partition2 = int(0.8*len(file_paths))
print(f'file_paths: {file_paths}, len(file_paths): {len(file_paths)}') 
train_list = read_dir(file_paths[:partition])
valid_list = read_dir(file_paths[partition:partition2])
test_list = read_dir(file_paths[partition2:])

# print(test_list)
np.save('train_list_scorer.npy', train_list)
np.save('valid_list_scorer.npy', valid_list)
np.save('test_list_scorer.npy', test_list)
