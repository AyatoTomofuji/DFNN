import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# データセットの作成
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ブラックボックスモデルのトレーニング
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ブラックボックスモデルからルールを抽出してホワイトボックスモデルを構築
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

# ストリームデータ処理のためのフレームワーク（簡易版）
def stream_data(X_test):
    for i in range(X_test.shape[0]):
        yield X_test[i]

# リアルタイム適用のシミュレーション
for data in stream_data(X_test):
    data = data.reshape(1, -1)
    bb_pred = rf.predict(data)
    wb_pred = dt.predict(data)
    print(f"ブラックボックス予測: {bb_pred}, ホワイトボックス解釈: {wb_pred}")

# ホワイトボックスモデルのルールを表示
rules = export_text(dt, feature_names=[f"feature_{i}" for i in range(X.shape[1])])
print(rules)
#最終的な識別率を表示
print(f"最終的な識別率: {dt.score(X_test, y_test)}")