import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Irisデータセットのロード
iris = load_iris()
wine = load_wine()
cancer = load_breast_cancer()
digits = load_digits()
set = digits

X = set.data
y = set.target

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# データの標準化

n_features = X.shape[1]
out = len(y)
# ニューラルネットワークの定義
class Net(nn.Module):
    def __init__(self, n_features, out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net(n_features, out)

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

# 訓練データ全体の予測結果を取得
net.eval()
with torch.no_grad():
    train_outputs = net(torch.tensor(X_train, dtype=torch.float32))
    _, train_predicted = torch.max(train_outputs.data, 1)
    train_predicted = train_predicted.numpy()


grader = tree.DecisionTreeClassifier()
grader.fit(X_train, y_train)
grade = grader.predict(X_test)

# 決定木の訓練
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

test_outputs = clf.predict(X_test)
accuracy_tree = np.sum(test_outputs == y_test) / len(y_test)
print(f"ランダムフォレストによる{accuracy_tree}")


dt = DecisionTreeClassifier(max_depth=4)

dt.fit(X_train, train_predicted)
test_outputs3 = dt.predict(X_test)
accuracy_tree3 = np.sum(test_outputs3 == y_test) / len(y_test)
print(f"ランダムフォレストから学習した決定木の精度{accuracy_tree3}")


class_names = [str(name) for name in digits.target_names]
# 決定木の可視化
plt.figure(figsize=(20,10))
tree.plot_tree(dt, feature_names=set.feature_names, class_names=class_names, filled=True)
plt.show()
