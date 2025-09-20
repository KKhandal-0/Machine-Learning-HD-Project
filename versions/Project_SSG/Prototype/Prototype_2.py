import kagglehub
import pandas as pd
import numpy as np
import sklearn
import shutil
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier # Second implemenation using Random Forest Classifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("student_info.csv")

def get_file(): #Simple import file code
    kagglehub.dataset_download("therohithanand/student-academic-performance-dataset")
    path_file = r"C:\Users\Kamal\.cache\kagglehub\datasets\therohithanand\student-academic-performance-dataset\versions\1\student_info.csv"
    path_destination = r"C:\Users\Kamal\Desktop\Projects\Project_SSG"
    try:
        shutil.copy(path_file,path_destination)
    except:   
        print("File is already present")
    else:
        print("File imported")

def transform(data): #data cleaning, transformation
    data["final_result_num"] = data["final_result"].map({"Fail": 0, "Pass": 1})
    features = ["study_hours", "writing_score", "attendance_rate", "grade_level", "math_score"] 
    X = data[features]
    Y = data["final_result_num"]
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=50) 
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
    
    return X_train_scaled, Y_train, X_test_scaled, Y_test

def test(model, X_test_scaled, Y_test):
    y_pred = model.predict(X_test_scaled)
    print("Accuracy: ", accuracy_score(Y_test, y_pred))
    
def model_GradientBoosting(data):
    X_train_scaled, Y_train, X_test_scaled, Y_test = transform(data)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train_scaled, Y_train)
    test(model, X_test_scaled, Y_test)
    
model_GradientBoosting(data)
    