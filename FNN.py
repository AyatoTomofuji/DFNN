#I want to implement a fuzzy neural network

import numpy as np
#ニューラルネットを,torchを使って実装して，wineデータセットを使って学習させる
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

class DNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(DNN, self).__init__()
        self.l1 = nn.Linear(n_input, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

#データの読み込み
data = load_wine()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
y_test = torch.tensor(y_test, dtype=torch.int64)

#モデルの定義
model = DNN(13, 128, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#学習
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch:{epoch}, loss:{loss.item()}")
#テスト
output = model(X_test)
_, predicted = torch.max(output, 1)
accuracy = accuracy_score(y_test, predicted)
print(f"accuracy:{accuracy}")

