import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier


# ファジィ分類器のクラス
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
        features = [f"x{i + 1}" for i in range(self.n_features)]
        rules = []

        for combination in itertools.product(range(4), repeat=self.n_features):
            rule = " and ".join([f"{features[i]} is {terms[combination[i]]}" for i in range(self.n_features)])
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
        CF_q = np.array(CF_q)
        count = self.n_features * len(CF_q[CF_q > 0])
        index = np.where(CF_q > 0)
        for k in index[0]:
            for j in range(self.n_features):
                if (k // (4 ** j)) % 4 == 3: count -= 1
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
            q = [(i // (4 ** j)) % 4 for j in range(self.n_features)]
            m = self.fit_rule(X, q)
            m_sum = np.array([0.0, 0.0, 0.0, 0.0])
            for l in range(4):
                m_sum[l] = np.sum(m[y == l])
            self.c.append(m_sum / np.sum(m_sum) if np.sum(m_sum) != 0 else m_sum)
        self.C_q = np.argmax(self.c, axis=1)
        self.CF_q = 2 * np.max(self.c, axis=1) - np.sum(self.c, axis=1)

        self.rules = self.rule_change(self.CF_q)

    def predict(self, X):
        pro = np.full((4 ** self.n_features, len(X)), -1.0)
        for i in range(4 ** self.n_features):
            q = [(i // (4 ** j)) % 4 for j in range(self.n_features)]
            m = self.fit_rule(X, q)
            pro[i] = m * self.CF_q[i]
        R_w = np.argmax(pro, axis=0)
        fit_x = self.C_q[R_w]
        return fit_x

    def score(self, X, y):
        fit_x = self.predict(X)
        accuracy = np.sum(fit_x == y) / len(X)
        return accuracy

    def print_rules(self):
        print(f"結論部:{self.C_q}")
        print(f"信頼度：\n{self.CF_q}")
        print(f"ルール数:{len(self.rules)}")
        print(f"総ルール長：{self.rule_count(self.CF_q)}")
        print(f"得られた識別器:")
        for rule in self.rules:
            print(f"{self.Ruledict[rule]} then Class{self.C_q[rule] + 1} with CF={self.CF_q[rule]}")


if __name__ == '__main__':
    # OpenMLデータセットの読み込み
    set_id = 1462  # 使用するデータセットIDを指定
    dataset = fetch_openml(data_id=set_id)
    num_features = 4
    X, y = make_classification(n_samples=8000,  # サンプル数
                               n_features=num_features,  # 次元数
                               flip_y=0,
                               class_sep=2.2,
                               n_informative=num_features,  # 有益な特徴量の数
                               n_redundant=0,  # 冗長な特徴量の数
                               n_clusters_per_class=1,  # クラスごとのクラスター数
                               n_classes=4,  # クラス数（4クラス分類）
                               random_state=42)  #

    X = MinMaxScaler().fit_transform(X)
    for class_value in range(4):
        indices = np.where(y == class_value)
        plt.scatter(X[indices, 0], X[indices, 1], label=f'Class {class_value + 1}', s=50, alpha=0.6)
    plt.xlim = (-0.05, 1.05)
    plt.ylim = (-0.05, 1.05)
    plt.legend()
    plt.show()

    # ターゲットがカテゴリカルの場合、ラベルをエンコード
    if y.dtype == 'O':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    X = MinMaxScaler().fit_transform(X)
    # 訓練データとテストデータに分割

    # クラス分類器のインスタンス作成と学習
    classifier = FuzzyClassifier()

    import time
    import numpy as np

    # 実行時間を記録するためのリスト
    train_times = []
    test_times = []
    train_accuracies = []
    test_accuracies = []
    # 30回繰り返して実行
    for i in range(30):
        classifier = FuzzyClassifier()
        # 訓練時間の計測
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        start_train = time.time()
        classifier.fit(X_train, y_train)
        end_train = time.time()
        train_accuracies.append(classifier.score(X_train, y_train))
        train_times.append(end_train - start_train)

        start_test = time.time()
        accuracy_test = classifier.score(X_test, y_test)
        end_test = time.time()
        test_accuracies.append(accuracy_test)

        # テスト時間をリストに追加
        test_times.append(end_test - start_test)

        print(
            f"Iteration {i + 1}: Train Time: {end_train - start_train:.4f} sec, Test Time: {end_test - start_test:.4f} sec")

    # 訓練とテストの平均実行時間を表示
    average_train_time = np.mean(train_times)
    average_test_time = np.mean(test_times)
    average_train_accuracy = np.mean(train_accuracies)
    average_test_accuracy = np.mean(test_accuracies)
    classifier.print_rules()
    print(f"{num_features}dim")
    print(f"\nAverage Train Time: {average_train_time:.4f} sec")
    print(f"Average Train Accuracy: {average_train_accuracy:.4f}")
    print(f"Average Test Time: {average_test_time:.4f} sec")
    print(f"Average Test Accuracy: {average_test_accuracy:.4f}")
