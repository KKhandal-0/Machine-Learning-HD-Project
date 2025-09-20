import pandas as pd

data = pd.read_csv("heart.csv")

contents = ["Sex", "ChestPainType", "ExerciseAngina", "ST_Slope",  "HeartDisease"]
for i in contents:
    print(data[i].unique())