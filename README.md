Heart Disease Prediction using Machine Learning

A Python-based machine learning project that predicts the likelihood of heart disease using patient health data. This project combines data preprocessing, model building using Logistic Regression, and a PyQt5 GUI to deliver real-time predictions with a user-friendly interface.

Features

- Logistic Regression model trained on real-world health dataset
- Over **85% accuracy** on test data
- Clean and functional **PyQt5 GUI** for user input and prediction
- Modular codebase (separate scripts for preprocessing, model, and GUI)
- Simple interface to help doctors/individuals check risk probability

Tech Stack

- **Language:** Python 3.x  
- **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`  
- **GUI:** PyQt5  
- **Dataset:** [`heart.csv`](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

Dataset Info

| Feature | Description |
|--------|-------------|
| **Feature**      | **Description**                                       |
| ---------------- | ----------------------------------------------------- |
| `Age`            | Age of the patient (in years)                         |
| `Sex`            | Gender of the patient (1 = male, 0 = female)          |
| `ChestPainType`  | Type of chest pain experienced                        |
| `RestingBP`      | Resting blood pressure (mm Hg)                        |
| `Cholesterol`    | Serum cholesterol level (mg/dl)                       |
| `FastingBS`      | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `RestingECG`     | Resting electrocardiogram results                     |
| `MaxHR`          | Maximum heart rate achieved                           |
| `ExerciseAngina` | Exercise-induced angina (Y = yes, N = no)             |
| `Oldpeak`        | ST depression induced by exercise relative to rest    |
| `ST_Slope`       | Slope of the peak exercise ST segment                 |
| `HeartDisease`   | Target variable (1 = has heart disease, 0 = does not) |




How to Run

```bash
# Clone the repo
git clone https://github.com/KKhandal-0/Machine-Learning-HD-Project.git

# Install requirements
pip install -r requirements.txt

# Run the GUI
python gui.py
```

> ⚠️ Make sure `heart.csv` is in the root directory.

GUI Screenshot

_Add a screenshot here (e.g., drag a PyQt5 window screenshot into GitHub or link it)_


Model Accuracy

- Train-Test Split: 80-20
- Accuracy: **~86%**
- Evaluation: Confusion matrix & classification report included

---
Skills Demonstrated

- ML pipeline (cleaning → training → evaluation)
- GUI integration with ML models
- Real-time data input + user-friendly predictions
- Model interpretation and presentation


Project Structure

```
├── gui.py             # PyQt5 user interface for real-time prediction
├── model.py           # Contains logistic regression model and prediction logic
├── preprocess.py      # Handles data preprocessing (scaling, encoding, cleaning)
├── heart.csv          # Dataset used for training and testing
├── README.md          # Project documentation (this file)
```


Acknowledgements

- Dataset by [Fedesoriano on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- PyQt5 Documentation for GUI integration
