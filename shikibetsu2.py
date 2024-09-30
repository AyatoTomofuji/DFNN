import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import shap

# データセットの読み込み
df = pd.read_csv('data/yeast.data', header=None, sep='\s+')
df.columns = ['Gene', 'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Location']
X = df[['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8']].values
y = df['Location'].values

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ブラックボックスモデルの訓練
blackbox_model = RandomForestClassifier()
blackbox_model.fit(X_train, y_train)

# SHAPを使用して特徴量の影響を解析
explainer = shap.TreeExplainer(blackbox_model)
shap_values = explainer.shap_values(X_test)

print(f"shap_values:{shap_values}")
# ハードデータの識別
base_clf = DecisionTreeClassifier(max_depth=4)
base_clf.fit(X_train, y_train)
base_predictions_train = base_clf.predict(X_train)
easy_mask_train = base_predictions_train == y_train
hard_mask_train = ~easy_mask_train

# ハードデータを抽出
hard_mask = (y_test != blackbox_model.predict(X_test))
X_hard = X_test[hard_mask]
y_hard = y_test[hard_mask]

# LIMEを使用してローカルな説明を提供
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=df.columns[1:-1], class_names=np.unique(y), discretize_continuous=True)

# 任意のハードデータポイントのローカルな説明を表示
i = 0  # ハードデータポイントのインデックス
lime_exp = lime_explainer.explain_instance(X_hard[i], blackbox_model.predict_proba, num_features=5)
lime_exp.show_in_notebook()

# クラスタリングによるハードデータのグループ化
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_hard)

# クラスタごとにルールを抽出し、可視化
for cluster in np.unique(clusters):
    X_cluster = X_hard[clusters == cluster]
    y_cluster = y_hard[clusters == cluster]
    rule_model = DecisionTreeClassifier(max_depth=3)
    rule_model.fit(X_cluster, y_cluster)
    plot_tree(rule_model, feature_names=df.columns[1:-1], class_names=np.unique(y), filled=True)
    plt.title(f'Cluster {cluster} Decision Tree')
    plt.show()
