import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

N_pop = 30
N_rep = 10
count_max = 10000
p = 0.9
# 分割数
K = 3

class FuzzyClassifier:
    def __init__(self, K=3):
        self.K = K
        self.C_q = []
        self.CF_q = []
        self.rules = []
        self.Ruledict = {
            0: "x1 is small and x2 is small",
            1: "x1 is small and x2 is medium",
            2: "x1 is small and x2 is large",
            3: "x1 is small and x2 is don't care",
            4: "x1 is medium and x2 is small",
            5: "x1 is medium and x2 is medium",
            6: "x1 is medium and x2 is large",
            7: "x1 is medium and x2 is don't care",
            8: "x1 is large and x2 is small",
            9: "x1 is large and x2 is medium",
            10: "x1 is large and x2 is large",
            11: "x1 is large and x2 is don't care",
            12: "x1 is don't care and x2 is small",
            13: "x1 is don't care and x2 is medium",
            14: "x1 is don't care and x2 is large",
            15: "x1 is don't care and x2 is don't care"
        }

    def membership2(self, x, k):
        if k == 3: return 1.0
        b = 1 / (self.K - 1)
        a = k / (self.K - 1)
        return max(0, (1 - np.abs(a - x) / b))

    def fit_rule(self, x_array, i):
        q1 = i // 4
        q2 = i % 4
        m = np.array([self.membership2(x[0], q1) * self.membership2(x[1], q2) for x in x_array])
        return m

    def rule_change(self, CF_q):
        return [b for b in range(len(CF_q)) if CF_q[b] > 0]

    def rule_count(self, CF_q):
        count = 2 * len(CF_q[CF_q > 0])
        index = np.where(CF_q > 0)
        for k in index[0]:
            if k % 4 == 3: count -= 1
            if k // 4 == 3: count -= 1
        return count

    def fit(self, X, y):
        self.M = []
        self.C_q = []
        self.c = []
        X = MinMaxScaler().fit_transform(X)

        for i in range(16):
            m = self.fit_rule(X, i)
            m_sum = np.array([0.0, 0.0, 0.0])
            for l in range(3):
                m_sum[l] = np.sum(m[y == l])
            self.c.append(m_sum / np.sum(m_sum))

        self.C_q = np.argmax(self.c, axis=1)
        self.CF_q = 2 * np.max(self.c, axis=1) - np.sum(self.c, axis=1)
        self.rules = self.rule_change(self.CF_q)

    def predict(self, X):
        X = MinMaxScaler().fit_transform(X)
        pro = np.full((16, len(X)), -1.0)
        for i in range(16):
            m = self.fit_rule(X, i)
            pro[i] = m * self.CF_q[i]
        R_w = np.argmax(pro, axis=0)
        fit_x = self.C_q[R_w]
        return fit_x

    def score(self, X, y):
        fit_x = self.predict(X)
        accuracy = np.count_nonzero(fit_x == y) / len(X)
        return accuracy

    def print_rules(self):
        print(f"結論部:{self.C_q}")
        print(f"信頼度：\n{self.CF_q}")
        print(f"ルール数:{len(self.rules)}")
        print(f"総ルール長：{self.rule_count(self.CF_q)}")
        print(f"得られた識別器:")
        for rule in self.rules:
            print(f"{self.Ruledict[rule]} then Class{self.C_q[rule] + 1}")

    def plot_data(self, df):
        X_ax = np.arange(0, 1.001, 0.001)
        y_ax = np.arange(0, 1.001, 0.001)
        X, y = np.meshgrid(X_ax, y_ax)
        x_sep = df.groupby("label")["x"].apply(list)
        y_sep = df.groupby("label")["y"].apply(list)

        w = np.c_[X.ravel(), y.ravel()]
        pro = np.full((16, len(w)), -1.0)

        for q in range(16):
            m = self.fit_rule(w, q)
            pro[q] = m * self.CF_q[q]
        R_w = np.argmax(pro, axis=0)
        Z = self.C_q[R_w].reshape(X.shape)
        cont = plt.contour(X, y, Z, colors='black', linewidths=2.0)
        plt.scatter(x_sep[0], y_sep[0], c='r', label="Class 1")
        plt.scatter(x_sep[1], y_sep[1], c='b', label="Class 2")
        plt.scatter(x_sep[2], y_sep[2], c='g', label="Class 3")
        plt.grid()
        plt.legend()
        plt.show()



class Net(nn.Module):
    def __init__(self, n_features, out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# データの読み込み
df1 = pd.read_csv('data/kadai3_pattern1.txt', header=None, skiprows=[0])
df2 = pd.read_csv('data/kadai3_pattern2.txt', header=None, skiprows=[0])
scaler = StandardScaler()


classifier = FuzzyClassifier()
x_array = df1.values
X_train, y_train = scaler.fit_transform(x_array[:, :2]), x_array[:, 2]
X_test, y_test = scaler.fit_transform(df2.values[:, :2]), df2.values[:, 2]

classifier.fit(X_train, y_train)
# 識別率の表示
accuracy = classifier.score(X_train, y_train)
print(f"識別率:{accuracy}")
#テスト識別率の表示
accuracy_test = classifier.score(X_test, y_test)
print(f"テスト識別率:{accuracy_test}")

# ランダムフォレストを実装
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"ランダムフォレストのテスト識別率:{accuracy}")