# Import Libraries
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Upload Dataset
df = pd.read_csv("heart.csv")

# Data Cleaning
df = df[df.thal != 0]

# Separating Dependent Features
x = df.drop(['target'], axis=1)
y = df['target']

# Data Normalization using Min-Max Method
x = MinMaxScaler().fit_transform(x)

# Splitting Dataset into 80:20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# Create Model (KNN)
KNNClassifier = KNeighborsClassifier(n_neighbors=3)

# Train Model
KNNClassifier.fit(x_train, y_train)

# Prediction Model
y_pred_KNN = KNNClassifier.predict(x_test)

KNNAcc = accuracy_score(y_pred_KNN, y_test)
print('K-Nearest Neighbour Accuracy:'+'\033[1m {:.2f}%'.format(KNNAcc*100))

# Pickle Model
pd.to_pickle(KNNClassifier, r'cvd_model.pickle')
