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
| Age | Patient's age |
| Sex | 1 = Male, 0 = Female |
| ChestPainType | Typical/Asymptomatic |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol |
| ... | ...and more medical indicators |



How to Run

```bash
# Clone the repo
git clone https://github.com/KKhandal-0/Machine-Learning-HD-Project.git

# Run the GUI
python main.py
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

├── gui.py               # PyQt5 user interface
├── model.py             # ML model creation and prediction logic
├── preprocess.py        # Data preprocessing steps
├── heart.csv            # Dataset
├── README.md            # This file


Acknowledgements

- Dataset by [Fedesoriano on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- PyQt5 Documentation for GUI integration
