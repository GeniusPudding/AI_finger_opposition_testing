import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'有CUDA，使用{torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    print('沒CUDA，使用CPU訓練')


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

class VideoClassifier(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=16, num_layers=2, num_classes=2, dropout_prob=0.5):
        super(VideoClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x):
        # 初始化隱藏狀態和細胞狀態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))  # out: batch_size, seq_length, hidden_dim

        # Dropout層
        # out = self.dropout(out)

        # 只選擇序列的最後一個輸出
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":


    # 將模型部署到 CUDA 設備上

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_list = np.load('train_list.npy', allow_pickle=True)
    test_list = np.load('test_list.npy', allow_pickle=True)
    valid_list = np.load('valid_list.npy', allow_pickle=True)
    # print(f'train_list[1]: {train_list[1]}, train_list[1]["data"].shape: {train_list[1]["data"].shape}')
    # input(f'len(train_list): {len(train_list)}, len(test_list): {len(test_list)}, len(valid_list): {len(valid_list)}')
    
    train_dataset = VideoDataset(train_list)
    valid_dataset = VideoDataset(valid_list)
    test_dataset = VideoDataset(test_list)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    model = VideoClassifier()
    best_valid_loss = float('inf') # 初始設定最好的驗證損失為無窮大

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
    weight_decay = 1e-5  
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)

    # 設定訓練週期數 (epochs)
    num_epochs = 100

    # for epoch in tqdm(range(num_epochs)):
    #     model.train()
    #     train_loss = 0.0
    #     for data, labels in train_loader:
    #         optimizer.zero_grad()
    #         data = data.to(device)
    #         outputs = model(data)
    #         labels = labels.to(device)
    #         _, predicted = torch.max(outputs.data, 1)
    #         # input(f'predicted: {predicted}, labels: {labels},\ndiff: {predicted - labels}')
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()

    #     # 在每個訓練週期後，使用驗證資料集來評估模型性能
    #     valid_loss = 0.0
    #     model.eval()  # 將模型切換到評估模式
    #     with torch.no_grad():
    #         for data, labels in valid_loader:
    #             data = data.to(device)
    #             outputs = model(data)
    #             labels = labels.to(device)
    #             loss = criterion(outputs, labels)
    #             valid_loss += loss.item()

    #     model.train()  # 將模型切換回訓練模式

    #     avg_train_loss = train_loss / len(train_loader)
    #     avg_valid_loss = valid_loss / len(valid_loader)

    #     # 如果驗證損失是迄今為止最低的，則儲存模型的權重
    #     if avg_valid_loss < best_valid_loss:
    #         best_valid_loss = avg_valid_loss
    #         torch.save(model.state_dict(), 'model.pth')
    #         print(f'\nModel saved! , avg_valid_loss: {avg_valid_loss}')

    #     print('Epoch [{}/{}], Train Loss: {:.8f}, Valid Loss: {:.8f}'
    #         .format(epoch+1, num_epochs, avg_train_loss, avg_valid_loss))

    # 將模型切換到評估模式
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    total_samples = 0
    correct_samples = 0
    total_loss = 0

    # 在進行評估時，我們不需要更新模型的參數，因此不需要計算梯度
    # print(f'len(test_list): {len(test_list)}')
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


