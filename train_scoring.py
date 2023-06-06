import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import copy
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from scipy import stats

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'有CUDA，使用{torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    print('沒CUDA，使用CPU訓練')


class ScorerDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = torch.tensor(self.data_list[idx]['data'], dtype=torch.float32).view(30, 63)  # 將資料重塑為 (S, E) 形狀
        label = torch.tensor(self.data_list[idx]['label'], dtype=torch.long)
        return data, label - 1
class TransformerClassifier(nn.Module):#model_scorer - Transformer_basic.pth,  Test Loss: 0.044244, Accuracy: 98.109%
    def __init__(self, input_dim=63, num_heads=3, num_layers=1, num_classes=4):
        super(TransformerClassifier, self).__init__()
        self.embed_dim = input_dim
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])
        return x
class GRUClassifier(nn.Module): #model_scorer - GRU_basic.pth, Test Loss: 0.491785, Accuracy: 93.635%
    def __init__(self, input_dim=63, hidden_dim=32, num_layers=2, num_classes=4):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device) 
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_list = np.load('train_list_scorer.npy', allow_pickle=True)
    valid_list = np.load('valid_list_scorer.npy', allow_pickle=True)
    test_list = np.load('test_list_scorer.npy', allow_pickle=True)

    # # Part 1 : 將資料轉換為 XGBoost 能使用的格式，並訓練模型
    # X_train = [item['data'].reshape(-1) for item in train_list]
    # y_train = [item['label'] for item in train_list]

    # X_valid = [item['data'].reshape(-1) for item in valid_list]
    # y_valid = [item['label'] for item in valid_list]

    # X_test = [item['data'].reshape(-1) for item in test_list]
    # y_test = [item['label'] for item in test_list]

    # # 將 labels 編碼為 0 ~ (n_classes - 1)
    # le = LabelEncoder()
    # y_train = le.fit_transform(y_train)
    # y_valid = le.transform(y_valid)
    # y_test = le.transform(y_test)

    # # 轉換資料格式
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # dvalid = xgb.DMatrix(X_valid, label=y_valid)
    # dtest = xgb.DMatrix(X_test)

    # # 設定參數
    # param = {'max_depth': 6, 'num_class': 4, 'objective': 'multi:softmax', 'nthread': 4, 'eval_metric': 'mlogloss'}
    # evallist = [(dvalid, 'eval'), (dtrain, 'train')]

    # # 訓練模型
    # num_round = 50
    # bst = xgb.train(param, dtrain, num_round, evallist)

    # #儲存模型
    # bst.save_model('xgb_model_scorer.model')

    # # 進行預測
    # preds = bst.predict(dtest) # XGBoost 模型的準確率為: 96.71%

    # # 計算準確率
    # accuracy = (y_test == preds).sum().astype(float) / len(preds) * 100
    # print('XGBoost 模型的準確率為: %.2f%%' % accuracy)

    # Part 2 : 訓練 PyTorch NN 模型
    train_dataset = ScorerDataset(train_list)
    valid_dataset = ScorerDataset(valid_list)
    test_dataset = ScorerDataset(test_list)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model = GRUClassifier() #TransformerClassifier()
    # if os.path.isfile('model_scorer.pth'):
    #     model.load_state_dict(torch.load('model_scorer.pth'))
    #     print('model_scorer.pth exists, load model')
    # else:
    #     print('model_scorer.pth not exists, train model')

    # model = model.to(device)
    # num_params = sum(param.numel() for param in model.parameters())
    # print(f'num_params: {num_params}')
    # criterion = nn.CrossEntropyLoss()
    # weight_decay = 1e-5  # L2 regularization
    # optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    # best_loss = float('inf')
    # best_model_wts = copy.deepcopy(model.state_dict())
    # num_epochs = 100
    # for epoch in tqdm(range(num_epochs)):
    #     model.train()
    #     for data, labels in train_loader:
    #         optimizer.zero_grad()
    #         data = data.to(device)
    #         outputs = model(data)
    #         labels = labels.to(device)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #     # Validation
    #     model.eval()
    #     with torch.no_grad():
    #         valid_loss = 0.0
    #         correct = 0
    #         for data, labels in valid_loader:
    #             data = data.to(device)
    #             labels = labels.to(device)
    #             outputs = model(data)
    #             loss = criterion(outputs, labels)
    #             valid_loss += loss.item()
    #             _, predicted = torch.max(outputs.data, 1)
    #             correct += (predicted == labels).sum().item()
    #         avg_valid_loss = valid_loss / len(valid_loader)
    #         print('Epoch [{}/{}], Loss: {:.6f}, Validation Loss: {:.6f}, Accuracy: {:.3f}%'
    #             .format(epoch+1, num_epochs, loss.item(), avg_valid_loss, (100 * correct / len(valid_dataset))))
            
    #         # 如果這個 epoch 的驗證損失是迄今為止最低的，則儲存模型權重
    #         if avg_valid_loss < best_loss:
    #             best_loss = avg_valid_loss
    #             best_model_wts = copy.deepcopy(model.state_dict())
    #             torch.save(best_model_wts, 'model_scorer.pth')
    #             print('Get best valid loss {:.6f}, save best model.'.format(best_loss))
    # # # Testing
    # input()
    # # 加載最佳模型權重
    #集成GRU，Transformer，xgboost的結果然後多數決
    model = TransformerClassifier()
    model.load_state_dict(torch.load('model_scorer - Transformer_basic.pth'))
    model = model.to(device)
    
    model2 = GRUClassifier()
    model2.load_state_dict(torch.load('model_scorer - GRU_basic.pth'))
    model2 = model2.to(device)

    model3 = xgb.Booster(model_file='xgb_model_scorer.model')

    #Ensemble three models prediction:
    model.eval()    
    model2.eval()

    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    majority_vote_correct = 0
    one_hot_num = 4
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)

            # Get predictions from the three models
            outputs1 = model(data)
            outputs2 = model2(data)
            labels_predicted = model3.predict(xgb.DMatrix(data.cpu().numpy().reshape(-1,1890)))
            outputs3 = labels_predicted.astype(int)
            predicted3 = torch.from_numpy(outputs3).to(device)

            
            # Compute predicted classes
            # print(f'outputs1.data: {outputs1.data}, outputs2.data: {outputs2.data}')
            _, predicted1 = torch.max(outputs1.data, 1)
            _, predicted2 = torch.max(outputs2.data, 1)
            # Calculate accuracy
            correct1 += (predicted1 == labels).sum().item()
            correct2 += (predicted2 == labels).sum().item()
            correct3 += (predicted3.cpu() == labels.cpu()).sum().item()

            # Majority vote
            majority_vote = stats.mode(torch.stack([predicted1.cpu(), predicted2.cpu(), predicted3.cpu()]).numpy())[0]
            majority_vote_correct += (majority_vote.squeeze() == labels.cpu().numpy()).sum()

        print('Test Accuracy Model 1: {:.3f}%'.format(100 * correct1 / len(test_dataset)))
        print('Test Accuracy Model 2: {:.3f}%'.format(100 * correct2 / len(test_dataset)))
        print('Test Accuracy Model 3: {:.3f}%'.format(100 * correct3 / len(test_dataset)))
        print('Test Accuracy Majority Vote: {:.3f}%'.format(100 * majority_vote_correct / len(test_dataset)))


    # with torch.no_grad():
    #     for data, labels in test_loader:
    #         data = data.to(device)
    #         labels = labels.to(device)  
    #         labels_predicted = model3.predict(xgb.DMatrix(data.cpu().numpy().reshape(-1,1890)))
    #         outputs3 = labels_predicted.astype(int)
    #         correct3 += (outputs3 == labels.cpu().numpy()).sum()

    # print('XGBoost Test Accuracy: {:.3f}%'.format(100 * correct3 / len(test_dataset)))