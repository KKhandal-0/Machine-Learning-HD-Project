import kagglehub
import pandas as pd
import os
import numpy as np
import sklearn
import shutil
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = pd.read_csv("heart.csv")

def get_file(): #Simple import file code
    path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
    path_file = f"{path}\\heart.csv"
    path_destination = r"C:\Users\Kamal\Desktop\Projects\Project_SSG"
    try:
        shutil.copy(path_file,path_destination)
    except:   
        print("File is already present")
    else:
        print("File imported")

def transform(data): #data cleaning, transformation
    # Encode categorical variables
    data['ChestPainType_num'] = data['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
    data['RestingECG_num'] = data['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH':2})
    data["ST_Slope_num"] = data["ST_Slope"].map({"Up": 0, "Flat": 1, "Down":2})
    data["ExerciseAngina"] = data["ExerciseAngina"].map({"N": 0, "Y": 1})
    
# print(data["ChestPainType"].unique())
# print(data["RestingECG"].unique())
# print(data["ST_Slope"].unique())
    
    features = [
        "Age", "Sex", "ChestPainType_num", "RestingBP", "Cholesterol", "FastingBS", "RestingECG_num", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope_num"
    ]
    
    X = data[features]
    Y = data["HeartDisease"]
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.4, random_state=50) 
    
   #scaler = StandardScaler()
   #X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
   #X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
    
    return X_train, Y_train, X_test, Y_test

def test(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    print("Accuracy: ", accuracy_score(Y_test, y_pred))
    print("\nDetailed Classification Report:")
    print(classification_report(Y_test, y_pred))

def model_GradientBoosting(data):
    X_train, Y_train, X_test, Y_test = transform(data)
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Initialize base model
    base_model = GradientBoostingClassifier(random_state=50)
    
    # Perform grid search
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("\nBest cross-validation accuracy:", grid_search.best_score_)
    
    # Test the best model
    test(grid_search.best_estimator_, X_test, Y_test)

model_GradientBoosting(data)