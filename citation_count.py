import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

data_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1])
dataset = pd.read_csv(os.path.join(data_path, "content", "citations.csv"))
X = dataset.iloc[:, 2:8].values  # 8 lekhile 7 jaye naba
y = dataset.iloc[:, 8].values
print("--------------------X----------------------------")
print(X)
print("--------------------Y----------------------------")
print(y)


le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
print("--------------------X----------------------------")
print(X)

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough"
)
X = np.array(ct.fit_transform(X))
print("--------------------X----------------------------")
print(X)

X = X[:, 1]
print("--------------------X----------------------------")
print(X)

sc = StandardScaler()
X = sc.fit_transform(X)
print("--------------------X----------------------------")
print(X)
