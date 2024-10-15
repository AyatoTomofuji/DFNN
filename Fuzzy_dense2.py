from collections import Counter

import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris, fetch_openml, make_classification, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from Fuzzy import FuzzyClassifier


# ファジィ分類器のクラス
class FuzzyClassifierWithDensity(FuzzyClassifier):
    def __init__(self, K=3):
        super().__init__(K)
        self.K = K
        self.C_q = []
        self.CF_q = []
        self.rules = []
        self.n_features = None
        self.Ruledict = {}
        self.selected_rules = []  # 選択されたルールを保存
        self.c = []  # ルールの適合度

    def fit(self, X, density=None):
        self.n_features = X.shape[1]
        self.Ruledict = self.generate_rule_dict()

        self.C_q = []
        self.CF_q = []
        self.rules = []
        self.c = []

        total_rules = 4 ** self.n_features

        # 全ルールを探索せず、選択されたルールのみ保存
        for i in range(total_rules):
            q = [(i // (4 ** j)) % 4 for j in range(self.n_features)]
            m = self.fit_rule(X, q)
            m_sum = np.zeros(4)

            # 各サンプルのdensityの最大値を使ってクラスを推定する
            for j in range(len(X)):
                predicted_class = np.argmax(density[j])  # densityが最も高いクラスを推定
                m_sum[predicted_class] += m[j] * density[j, predicted_class]

            if np.sum(m_sum) > 0:
                self.selected_rules.append(q)
                self.c.append(m_sum / np.sum(m_sum) if np.sum(m_sum) != 0 else m_sum)
            print(self.c)
        self.C_q = [np.argmax(c) for c in self.c]
        self.CF_q = [2 * np.max(c) - np.sum(c) for c in self.c]
        valid_indices = [i for i, cf in enumerate(self.CF_q) if cf > 0.0]
        self.selected_rules = [self.selected_rules[i] for i in valid_indices]
        self.C_q = [self.C_q[i] for i in valid_indices]
        self.CF_q = [self.CF_q[i] for i in valid_indices]
        self.rules = self.rule_change(self.CF_q)

    def predict(self, X, density=None):
        pro = np.full((len(self.selected_rules), len(X)), -1.0)
        for i, q in enumerate(self.selected_rules):
            m = self.fit_rule(X, q)
            if 0:
                pro[i] = m * self.CF_q[i] * np.sum(density[y == l])
            else:
                pro[i] = m * self.CF_q[i]
        R_w = np.argmax(pro, axis=0)
        fit_x = [self.C_q[r] for r in R_w]
        return fit_x

    def score(self, X, y):
        fit_x = self.predict(X)
        accuracy = np.sum(fit_x == y) / len(X)
        return accuracy


if __name__ == '__main__':
    num_features = 2
    X_origin, y_origin = make_classification(n_samples=8000,  # サンプル数
                                             n_features=num_features,  # 次元数
                                             flip_y=0,
                                             class_sep=2.2,
                                             n_informative=num_features,  # 有益な特徴量の数
                                             n_redundant=0,  # 冗長な特徴量の数
                                             n_clusters_per_class=1,  # クラスごとのクラスター数
                                             n_classes=4,  # クラス数（4クラス分類）
                                             random_state=42)  #
    X = MinMaxScaler().fit_transform(X_origin)
    # 3次元データを3次元プロットする
    if num_features == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for class_value in range(4):
            indices = np.where(y_origin == class_value)
            ax.scatter(X[indices, 0], X[indices, 1], X[indices, 2], label=f'Class {class_value + 1}', s=50, alpha=0.6)
        plt.legend()
        plt.show()
    # 2次元データなら2次元プロットする
    if num_features == 2:
        print("2次元")
        for class_value in range(4):
            indices = np.where(y_origin == class_value)
            plt.scatter(X[indices, 0], X[indices, 1], label=f'Class {class_value + 1}', s=50, alpha=0.6)
        plt.legend()
        plt.show()

    data = pd.read_csv(f"node_positions_all/all_{num_features}dim_50_050.csv", header=None)
    # dataのn_featuresより先の列だけをndarrayで取り出す
    print(data.iloc)
    density = np.array(data.iloc[:, num_features:].values)
    print(density)
    X = np.array(data.iloc[:, 0:num_features].values)
    y = np.array(data.iloc[:, (num_features + 1)].values)
    X = MinMaxScaler().fit_transform(X)
    class_counts = Counter(y)
    # 3次元データを3次元プロットする
    if num_features == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for class_value in range(4):
            indices = np.where(y == class_value)
            ax.scatter(X[indices, 0], X[indices, 1], X[indices, 2], label=f'Class {class_value + 1}', s=50, alpha=0.6)
        plt.legend()
        plt.show()
    # 2次元データなら2次元プロットする
    if num_features == 2:
        print("2次元")
        plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.6)
        plt.legend()
        plt.show()

    # ターゲットがカテゴリカルの場合、ラベルをエンコード
    """if y.dtype == 'O':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)"""

    # 特徴量を0~1に正規化
    X = MinMaxScaler().fit_transform(X)

    import time

    train_times = []
    test_times = []
    accuracy_tests = []
    accuracy_train = []
    for i in range(30):
        X_train, X_test, density_train, density_test = train_test_split(X,  density, test_size=0.2, )
        print(X_train.shape, X_test.shape, density_train.shape, density_test.shape)
        classifier = FuzzyClassifierWithDensity()
        start_train = time.time()
        # クラス分類器のインスタンス作成と学習
        classifier.fit(X_train,  density_train)
        end_train = time.time()
        # 訓練識別率
        #accuracy_train.append(classifier.score(X_train, y_train))
        #train_times.append(end_train - start_train)
        #start_test = time.time()
        #accuracy_test = classifier.score(X_test, y_test, )
        #end_test = time.time()
#
        #accuracy_tests.append(accuracy_test)
        #test_times.append(end_test - start_test)
        ## X_testとy_testを横向きに結合して表示する
        #print(
        #    f"Iteration {i + 1}: Train Time: {end_train - start_train:.4f} sec, Test Time: {end_test - start_test:.4f} sec")
    print(f"{num_features}dim")
    #average_train_time = np.mean(train_times)
    #average_test_time = np.mean(test_times)
    #average_accuracy = np.mean(accuracy_tests)
    #average_train_accuracy = np.mean(accuracy_train)
    #print(f"Average Train Time: {average_train_time:.4f} sec")
    #print(f"Average Test Time: {average_test_time:.4f} sec")
    #print(f"Average Train Accuracy: {average_train_accuracy:.4f}")
    #print(f"Average Test Accuracy: {average_accuracy:.4f}")
    #print(f"{num_features}次元の50_070")
    X = MinMaxScaler().fit_transform(X_origin)
    classifier.print_rules()
    print(classifier.score(X, y_origin))
