import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp
import sys
import numpy as np
import pickle
import os
import torch.nn.functional as F
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'有CUDA，使用{torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    print('沒CUDA，使用CPU訓練')
# data_list = np.load('data_list.npy', allow_pickle=True)

# # input(f'data_list: {data_list}')
# # 分割數據為訓練集和測試集
# train_list, test_list = train_test_split(data_list, test_size=0.2)



# 定義 Dataset
class VideoDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = torch.tensor(self.data_list[idx]['data'], dtype=torch.float32).view(30, 63)  # 將資料重塑為 (S, E) 形狀
        label = torch.tensor(self.data_list[idx]['label'], dtype=torch.long)
        return data, label
    

# 定義模型
# class VideoClassifier(nn.Module):
#     def __init__(self):
#         super(VideoClassifier, self).__init__()
#         self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=63, nhead=7)
#         self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)
#         self.classifier = nn.Linear(63, 2)
    
#     def forward(self, x):
#         x = x.transpose(0, 1)  # Transformer期待的輸入形狀為 (S, N, E)，所以我們需要轉置
#         x = self.transformer_encoder(x)
#         x = x.mean(dim=0)  # 取序列上的平均值，現在形狀變為 (N, E)
#         x = self.classifier(x)
#         return x
    
# class VideoClassifier(nn.Module):
#     def __init__(self):
#         super(VideoClassifier, self).__init__()
#         self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=63, nhead=7)
#         self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)
#         self.fc = nn.Linear(63, 32)
#         self.classifier = nn.Linear(32, 2)
    
#     def forward(self, x):
#         x = x.transpose(0, 1)  # Transformer期待的輸入形狀為 (S, N, E)，所以我們需要轉置
#         x = self.transformer_encoder(x)
#         x = x.mean(dim=0)  # 取序列上的平均值，現在形狀變為 (N, E)
#         x = F.relu(self.fc(x))
#         x = self.classifier(x)
#         return x    

# class VideoClassifier(nn.Module):
#     def __init__(self):
#         super(VideoClassifier, self).__init__()
#         self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=63, nhead=7)
#         self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)
#         self.lstm = nn.LSTM(63, 32, batch_first=True)
#         self.classifier = nn.Linear(32, 2)
    
#     def forward(self, x):
#         x = x.transpose(0, 1)  # Transformer期待的輸入形狀為 (S, N, E)，所以我們需要轉置
#         x = self.transformer_encoder(x)
#         x = x.transpose(0, 1)  # LSTM期待的輸入形狀為 (N, S, E)，所以我們需要轉置
#         x, _ = self.lstm(x)
#         x = x[:, -1, :]  # 取最後一個時間步的輸出
#         x = self.classifier(x)
#         return x

class VideoClassifier(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=16, num_layers=2, num_classes=2):
        super(VideoClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # 初始化隱藏狀態和細胞狀態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))  # out: batch_size, seq_length, hidden_dim
        
        # 只選擇序列的最後一個輸出
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":


    # 將模型部署到 CUDA 設備上

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_list = np.load('train_list.npy', allow_pickle=True)
    test_list = np.load('test_list.npy', allow_pickle=True)

    train_dataset = VideoDataset(train_list)
    test_dataset = VideoDataset(test_list)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = VideoClassifier()
    # #load model, if model.pth exists
    if os.path.isfile('model.pth'):
        model.load_state_dict(torch.load('model.pth'))
        print('model.pth exists, load model')
    else:
        print('model.pth not exists, train model')


    model = model.to(device)
    # 定義損失函數和優化器
    num_params = sum(param.numel() for param in model.parameters())
    print(f'num_params: {num_params}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 設定訓練週期數 (epochs)
    num_epochs = 100

    # 訓練模型
    for epoch in tqdm(range(num_epochs)):
        for data, labels in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            # input(f'labels: {labels}\npredicted:{predicted}')
            labels = labels.to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch+1, num_epochs, loss.item()))

    # 將模型切換到評估模式

    model.eval()
    total_samples = 0
    correct_samples = 0
    total_loss = 0

    # 在進行評估時，我們不需要更新模型的參數，因此不需要計算梯度
    print(f'len(test_list): {len(test_list)}')
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)
            labels = labels.to(device)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            print(f'predicted: {predicted}, labels: {labels}, outputs: {outputs}, data: {data}')
            total_samples += labels.size(0)
            correct_samples += (predicted == labels).sum().item()
            total_loss += loss.item()

    # 計算平均損失值和準確率
    avg_loss = total_loss / total_samples
    accuracy = 100 * correct_samples / total_samples
    print('Average Loss: {:.8f}, Accuracy: {:.3f}%'.format(avg_loss, accuracy))

    # 儲存模型權重
    torch.save(model.state_dict(), 'model.pth')


