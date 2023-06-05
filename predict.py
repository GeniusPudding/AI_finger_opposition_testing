import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
import cv2
import mediapipe as mp
from train_mediapipe import VideoClassifier
from tqdm import tqdm
from train_scoring import TransformerClassifier, GRUClassifier, ScorerDataset
import xgboost as xgb
from scipy import stats
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VideoPredictor:
    def __init__(self, model_path):
        self.model = VideoClassifier().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    def predict_video(self, video_path, frame_len=30):
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
                joint_data.append(frame_joint_data[:63])
                
        cap.release()
        predictions = []
        d = np.array(joint_data)

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
class DiseasePredictor:
    def __init__(self, model1_path, model2_path, model3_path):
        self.model1 = TransformerClassifier().to(device)
        self.model2 = GRUClassifier().to(device)
        self.model3 = xgb.Booster()
        # 載入模型參數
        self.model1.load_state_dict(torch.load(model1_path, map_location=device))
        self.model2.load_state_dict(torch.load(model2_path, map_location=device))
        self.model3.load_model(model3_path)
        # 設定模型為評估模式
        self.model1.eval()
        self.model2.eval()
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    # 將影片轉乘model可以使用的資料
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
                joint_data.append(frame_joint_data[:63])
                
        cap.release()
        d = np.array(joint_data)
        print(f'd.shape: {d.shape}')
        processed = []
        for i in range(d.shape[0] - frame_len + 1):
            d30 = np.array(joint_data[i:i+frame_len])
            processed.append(d30)
        return processed

    def predict_score(self, data):
        with torch.no_grad():
            outputs1 = self.model1(data)
            outputs2 = self.model2(data)
            labels_predicted = self.model3.predict(xgb.DMatrix(data.cpu().numpy().reshape(-1, 1890)))
            outputs3 = labels_predicted.astype(int)
            predicted3 = torch.from_numpy(outputs3).to(device)
            
            # Compute predicted classes
            # print(f'outputs1.data: {outputs1.data}, outputs2.data: {outputs2.data}')
            _, predicted1 = torch.max(outputs1.data, 1)
            _, predicted2 = torch.max(outputs2.data, 1)
            # Majority vote
            majority_vote = stats.mode(torch.stack([predicted1.cpu(), predicted2.cpu(), predicted3.cpu()]).numpy())[0]
            # Convert to tensor and to the same device (GPU)
            majority_vote = torch.from_numpy(majority_vote).to(device)
            return majority_vote      

if __name__ == "__main__":
    #Part 1 輸入影片路徑，輸出異常分數，判斷是否生病
    model_path = 'model.pth'
    video_path = sys.argv[1]
    predictor = VideoPredictor(model_path)
    try:
        predictions = predictor.predict_video(video_path)
    except:
        print(f'{video_path}無法處理，換一部試試')
        exit(1)
    score  = sum(predictions) / len(predictions)
    print(f'Abnormal Score = {score}')
    

    #Part 2 如果異常分數大於0.7，則視為異常，輸出生病警訊並需進一步判斷嚴重程度
    if score > 0.7:
        print('You are sick!')
        #引入"train_scoring.py"裡面訓練好的GRU、Transformer、xgboost三個模型，做集成預測然後輸出0到3分的分數，3為最嚴重
        #集成GRU，Transformer，xgboost的結果然後多數決
        model1_path, model2_path, model3_path = 'model_scorer - Transformer_basic.pth', 'model_scorer - GRU_basic.pth', 'xgb_model_scorer.model'
        disease_predictor = DiseasePredictor(model1_path, model2_path, model3_path) 
        #一樣處理影片資料變成(-1, 30, 63)形狀的tensor
        data = disease_predictor.process_video(video_path)
        #data轉成torch model可以用的data type
        data = torch.tensor(data, dtype=torch.float32).to(device)
        #輸出分數

        disease_score_list = disease_predictor.predict_score(data)
        #取disease_score_list裡面的眾數，取整數值
        disease_score = stats.mode(disease_score_list[0].cpu().numpy())[0][0]
        print(f'Disease Score = {disease_score}')
        
        # 根據disease_score_list的結果來給一個箱型圖 將其plot出來
        plt.boxplot(disease_score_list.cpu().numpy())
        plt.xlabel("frame")  # 加入x軸的標籤
        plt.ylabel("嚴重程度")  # 加入y軸的標籤
        plt.savefig('disease_score_list.png')
        plt.show()  # show出來
        plt.close()  # 關閉figure