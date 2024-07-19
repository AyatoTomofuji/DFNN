import numpy as np
import pandas as pd
import itertools

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#すべての可能なルールについて検討する，時間かかる
class FuzzyClassifier:
    def __init__(self, K=3):
        self.K = K
        self.C_q = []
        self.CF_q = []
        self.rules = []
        self.n_features = None
        self.Ruledict = {}

    def generate_rule_dict(self):
        terms = ["small", "medium", "large", "don't care"]
        features = [f"x{i+1}" for i in range(self.n_features)]
        rules = []

        for combination in itertools.product(range(4), repeat=self.n_features):
            rule = " and ".join([f"{features[i]} is {terms[combination[i]]}" for i in range(self.n_features)])
            print(rule)
            rules.append(rule)
        return {i: rule for i, rule in enumerate(rules)}

    def membership2(self, x, k):
        if k == 3: return 1.0
        b = 1 / (self.K - 1)
        a = k / (self.K - 1)
        return max(0, (1 - np.abs(a - x) / b))

    def fit_rule(self, x_array, q):
        m = np.ones(len(x_array))
        for j in range(self.n_features):
            membership_values = np.array([self.membership2(x[j], q[j]) for x in x_array])
            m *= membership_values
        return m

    def rule_change(self, CF_q):
        return [b for b in range(len(CF_q)) if CF_q[b] > 0]

    def rule_count(self, CF_q):
        count = self.n_features * len(CF_q[CF_q > 0])
        index = np.where(CF_q > 0)
        for k in index[0]:
            for j in range(self.n_features):
                if (k // (4**j)) % 4 == 3: count -= 1
        return count

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.Ruledict = self.generate_rule_dict()

        self.C_q = []
        self.CF_q = []
        self.rules = []
        self.c = []
        self.rule_indices = []

        total_rules = 4 ** self.n_features
        for i in range(total_rules):
            print(f"Rule {i}/{total_rules}")
            q = [(i // (4**j)) % 4 for j in range(self.n_features)]
            m = self.fit_rule(X, q)
            m_sum = np.array([0.0, 0.0, 0.0])
            for l in range(3):
                m_sum[l] = np.sum(m[y == l])
            self.c.append(m_sum / np.sum(m_sum) if np.sum(m_sum) != 0 else m_sum)
        self.C_q = np.argmax(self.c, axis=1)
        self.CF_q = 2 * np.max(self.c, axis=1) - np.sum(self.c, axis=1)
        self.rules = self.rule_change(self.CF_q)

    def predict(self, X):
        pro = np.full((4 ** self.n_features, len(X)), -1.0)
        for i in range(4 ** self.n_features):
            q = [(i // (4**j)) % 4 for j in range(self.n_features)]
            m = self.fit_rule(X, q)
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

# データの読み込み
iris = load_iris()
#iris.dataを0~1に正規化
X = iris.data
X = MinMaxScaler().fit_transform(X)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# クラス分類器のインスタンス作成と学習
classifier = FuzzyClassifier()
classifier.fit(X_train, y_train)

# 識別率の表示
accuracy = classifier.score(X_train, y_train)
print(f"識別率:{accuracy}")

accuracy_test = classifier.score(X_test, y_test)
print(f"テスト識別率:{accuracy_test}")

# 規則の表示
classifier.print_rules()
