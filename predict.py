import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
import cv2
import mediapipe as mp
from train_mediapipe import VideoClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VideoPredictor:
    def __init__(self, model_path):
        self.model = VideoClassifier().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    def process_video(self, video_path, frame_len=30):
        cap = cv2.VideoCapture(video_path)
        print(f'Processing {video_path}...')
        joint_data = []
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(image)
            if results.multi_hand_landmarks:
                frame_joint_data = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        frame_joint_data.extend([lm.x, lm.y, lm.z])
                joint_data.append(frame_joint_data)
        cap.release()
        predictions = []
        d = np.array(joint_data)
        # print(f'd.shape: {d.shape}')
        l = d.shape[0]
        for i in range(l - frame_len + 1):
            d30 = np.array(joint_data[i:i+frame_len])
            data_tensor = torch.tensor(d30, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = self.model(data_tensor)
                predicted_probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                abnormal_score = predicted_probs[0][1]  # 獲取異常類別的預測機率
                predictions.append(abnormal_score)
        return predictions

if __name__ == "__main__":
    model_path = 'model.pth'
    video_path = sys.argv[1]
    predictor = VideoPredictor(model_path)
    try:
        predictions = predictor.process_video(video_path)
    except:
        print(f'{video_path}無法處理，換一部試試')
        exit(1)
    score  = sum(predictions) / len(predictions)
    print(f'Abnormal Score = {score}')