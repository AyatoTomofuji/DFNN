import openml
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets, tree
from sklearn.datasets import load_wine, load_breast_cancer, fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree, export_text
from torch import nn
from imblearn.over_sampling import SMOTE
import torch.nn.functional as F


set_ids = [1462, 1464, 1467, 1476, 59, 41193, 1510, 294, 1494, 40982, 54, 181]




class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_test(dataset):
    X, y = dataset.data, dataset.target
    X, y = np.array(X), np.array(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # StratifiedKFoldを使用して10-foldクロスバリデーション
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # クロスバリデーションの結果を格納するリスト
    base_accuracies = []
    final_accuracies = []
    hard_accuracies = []
    plus_accuracies = []
    hard_ratios = []
    train_hard_ratios = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Base classifier のトレーニング
        base_clf = tree.DecisionTreeClassifier(max_depth=4)
        base_clf.fit(X_train, y_train)

        # Deferral classifier のトレーニング
        defe_clf = RandomForestClassifier()
        defe_clf.fit(X_train, y_train)

        # Base classifier で識別できた部分を easy、できなかった部分を hard とする
        base_predictions_train = base_clf.predict(X_train)
        easy_mask_train = base_predictions_train == y_train
        hard_mask_train = ~easy_mask_train

        # Easy と Hard のデータを分け、Easy を 1, Hard を 0 としてラベル付け
        y_easy = np.ones_like(y_train)
        y_easy[hard_mask_train] = 0
        smote = SMOTE(random_state=0, k_neighbors=2)
        if sum(y_easy == 0) > 0:
            X_resampled, y_resampled = smote.fit_resample(X_train, y_easy)
        else:
            X_resampled, y_resampled = X_train, y_easy

        # Grader classifier のトレーニング
        grader_clf = tree.DecisionTreeClassifier(max_depth=4)
        grader_clf.fit(X_resampled, y_resampled)
        train_easy_hard = grader_clf.predict(X_train)
        easy_mask_train = train_easy_hard == 1
        hard_mask_train = train_easy_hard == 0
        train_hard_ratios.append(np.sum(hard_mask_train) / len(y_train))

        # テストデータのハード・イージー判定
        test_hard_easy = grader_clf.predict(X_test)
        easy_mask_test = test_hard_easy == 1
        hard_mask_test = test_hard_easy == 0
        hard_ratios.append(np.sum(hard_mask_test) / len(y_test))

        # plus_clf はテストデータの Deferral の予測結果を用いてトレーニングする
        defe_predictions_test = defe_clf.predict(X_test)

        plus_clf = tree.DecisionTreeClassifier(max_depth=4)
        plus_clf.fit(X_test, defe_predictions_test)

        # 「easy」データの予測
        base_predictions_test = base_clf.predict(X_test)

        # 「hard」データの予測
        plus_predictions_test = plus_clf.predict(X_test)
        plus_predictions_test[hard_mask_test] = defe_predictions_test[hard_mask_test]
        # finalは、easyの予測をそのまま使い、maskを使ってhardの予測をかぶせる
        final_predictions_test = base_predictions_test.copy()
        final_predictions_test[hard_mask_test] = defe_predictions_test[hard_mask_test]

        # 精度を計算
        base_all_test = accuracy_score(y_test, base_predictions_test)
        hard_all_test = accuracy_score(y_test, defe_predictions_test)
        final_accuracy = accuracy_score(y_test, final_predictions_test)
        plus_accuracy = accuracy_score(y_test, plus_predictions_test)
        # 結果を格納
        base_accuracies.append(base_all_test)
        hard_accuracies.append(hard_all_test)
        final_accuracies.append(final_accuracy)
        plus_accuracies.append(plus_accuracy)
        if 0:
            plt.figure(figsize=(20, 10))
            plot_tree(grader_clf, filled=True, class_names=['easy', 'hard'])
            plt.title('grader')
            plt.show()
            plt.figure(figsize=(20, 10))
            #categories = dataset.target.cat.categories
            plot_tree(base_clf, filled=True, feature_names=dataset.feature_names, )
            plt.title('base')
            plt.show()
            plt.figure(figsize=(20, 10))
            plot_tree(plus_clf, filled=True, feature_names=dataset.feature_names, )
            plt.title('plus')
            plt.show()

    # グラフのプロット
    #plt.figure(figsize=(20, 10))  # プロットのサイズを調整
    return np.mean(base_accuracies), np.nanmean(hard_accuracies), np.mean(final_accuracies), np.mean(
        plus_accuracies), np.mean(hard_ratios), np.mean(train_hard_ratios)


base_accuracies, hard_accuracies, final_accuracies, plus_accuracies, hard_ratios, train_hard_ratio = [], [], [], [], [], []
print("事後的に，すべて学習")
for id in set_ids:
    dataset = fetch_openml(data_id=id)
    for i in range(10):
        test = train_test(dataset)
        base_accuracies.append(test[0])
        hard_accuracies.append(test[1])
        final_accuracies.append(test[2])
        plus_accuracies.append(test[3])
        hard_ratios.append(test[4])
        train_hard_ratio.append(test[5])

    print(f"Dataset ID: {id}")
    print(f"Base classifier average accuracy in 10-fold CV: {np.mean(base_accuracies):.4f}")
    print(f"Hard classifier average accuracy in 10-fold CV: {np.mean(hard_accuracies):.4f}")
    print(f"Final classifier average accuracy in 10-fold CV: {np.mean(final_accuracies):.4f}")
    print(f"Plus classifier average accuracy in 10-fold CV: {np.mean(plus_accuracies):.4f}")
    print(f"Average hard ratio train: {np.mean(train_hard_ratio):.4f}")
    print(f"Average hard ratio : {np.mean(hard_ratios):.4f}")

print("事後的に，すべて学習")
