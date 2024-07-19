import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Irisデータセットのロード
iris = load_iris()
wine = load_wine()
cancer = load_breast_cancer()
set = cancer

X = set.data
y = set.target

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# データの標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n_features = X.shape[1]
out = len(y)
# ニューラルネットワークの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

# 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# 訓練
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(torch.tensor(X_train, dtype=torch.float32))
    loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# テスト
net.eval()
with torch.no_grad():
    test_outputs = net(torch.tensor(X_test, dtype=torch.float32))
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted.numpy() == y_test).sum() / y_test.size
    print(f'Test Accuracy: {accuracy:.4f}')

print(predicted)
# 訓練データ全体の予測結果を取得
net.eval()
with torch.no_grad():
    train_outputs = net(torch.tensor(X_train, dtype=torch.float32))
    _, train_predicted = torch.max(train_outputs.data, 1)
    train_predicted = train_predicted.numpy()

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# 決定木の訓練
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

test_outputs = clf.predict(X_test)
accuracy_tree = np.sum(test_outputs == y_test) / len(y_test)
print(f"ランダムフォレストによる{accuracy_tree}")

clf2 = DecisionTreeClassifier(max_depth=4)
clf2.fit(X_train, y_train)
test_outputs2 = clf2.predict(X_test)
accuracy_tree2 = np.sum(test_outputs2 == y_test) / len(y_test)
print(f"決定木の精度{accuracy_tree2}")

clf3 = DecisionTreeClassifier(max_depth=4)

clf3.fit(X_train, train_predicted)
test_outputs3 = clf3.predict(X_test)
accuracy_tree3 = np.sum(test_outputs3 == y_test) / len(y_test)
print(f"ランダムフォレストから学習した決定木の精度{accuracy_tree3}")




# 決定木の可視化
plt.figure(figsize=(20,10))
tree.plot_tree(clf2, feature_names=set.feature_names, class_names=set.target_names, filled=True)
plt.show()
