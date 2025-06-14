import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Loading data
data = pd.read_csv('heart.csv')
global Accuracry

# Converting Textual or Non_numerical values to Neumerics
data['ChestPainType_num'] = data['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
data['RestingECG_num'] = data['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
data["ST_Slope_num"] = data["ST_Slope"].map({"Up": 0, "Flat": 1, "Down": 2})
data["ExerciseAngina"] = data["ExerciseAngina"].map({"N": 0, "Y": 1})
data["Sex"] = data["Sex"].map({"M": 1, "F": 0})

# Select features
features = ['Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
            'ChestPainType_num', 'RestingECG_num', 'ST_Slope_num', 'ExerciseAngina']
X = data[features]

#Select Prediction target
y = data['HeartDisease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
def test():
    Accuracy = accuracy_score(y_test, y_pred)
    return Accuracy

#Save as pkl
with open("logistic_model.pkl", "wb") as file:
    pickle.dump(model, file)

