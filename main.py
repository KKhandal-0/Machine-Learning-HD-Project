import sys
import Final_Model as FM
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit,
                             QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox, QFrame)
from PyQt5.QtGui import QFont, QPixmap, QColor
from PyQt5.QtCore import Qt
import pickle
import numpy as np

try:
    model = pickle.load(open("logistic_model.pkl", "rb"))
except FileNotFoundError:
    QMessageBox.critical(None, "Model Error", "logistic_model.pkl not found. Please ensure the model file is in the same directory.")
    sys.exit(1)

accuracy = FM.test()

class HeartDiseasePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heart Disease Risk Predictor")
        self.setGeometry(100, 100, 700, 600)
        self.apply_modern_stylesheet()

        self.init_ui()

    def apply_modern_stylesheet(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                color: #343a40;
                font-family: 'Inter', sans-serif;
                font-size: 14px;
            }
            QLabel {
                color: #495057;
            }
            QLineEdit {
                background-color: #ffffff;
                color: #343a40;
                border: 1px solid #ced4da;
                border-radius: 8px;
                padding: 10px;
            }
            QLineEdit:focus {
                border: 1px solid #007bff;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 25px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QMessageBox {
                background-color: #ffffff;
                color: #343a40;
                font-size: 14px;
            }
            QFrame#headerFrame {
                background-color: #e9ecef;
                border-bottom: 1px solid #dee2e6;
                border-radius: 10px;
                padding: 10px;
                margin-bottom: 20px;
            }
        """)

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(30, 30, 30, 30)

        header_frame = QFrame(self)
        header_frame.setObjectName("headerFrame")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel(self)
        pixmap_banner = QPixmap()
        try:
            loaded_pixmap = QPixmap("image.png")
            if not loaded_pixmap.isNull():
                pixmap_banner = loaded_pixmap.scaledToWidth(550, Qt.SmoothTransformation)
            else:
                raise Exception("Loaded pixmap is null.")
        except Exception as e:
            print(f"Error loading heart_banner.png: {e}")
            self.image_label.setText("Banner Image Not Found")
            self.image_label.setStyleSheet("color: red; font-weight: bold;")

        self.image_label.setPixmap(pixmap_banner)
        self.image_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.image_label)

        header_layout.addStretch(1)

        self.logo_label = QLabel(self)
        pixmap_logo = QPixmap()
        try:
            loaded_pixmap = QPixmap("image2.png")
            if not loaded_pixmap.isNull():
                pixmap_logo = loaded_pixmap.scaledToWidth(90, Qt.SmoothTransformation)
            else:
                raise Exception("Loaded pixmap is null.")
        except Exception as e:
            print(f"Error loading heart_logo.png: {e}")
            self.logo_label.setText("Logo Not Found")
            self.logo_label.setStyleSheet("color: red; font-weight: bold;")

        self.logo_label.setPixmap(pixmap_logo)
        self.logo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_layout.addWidget(self.logo_label)

        main_layout.addWidget(header_frame)

        title = QLabel("Heart Disease Risk Prediction")
        title.setFont(QFont('Inter', 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)
        grid_layout.setHorizontalSpacing(20)

        labels = [
            "Age",
            "Sex",
            "Chest Pain Type", 
            "RestingBP",
            "Cholesterol",
            "FastingBS (0 or 1)", 
            "RestingECG (0-2)",   
            "MaxHR",
            "Exercise Angina", 
            "Oldpeak",
            "ST_Slope"
        ]


        self.inputs = []

        for i, label_text in enumerate(labels):
            label = QLabel(label_text)
            label.setFont(QFont('Inter', 13))
            input_field = QLineEdit()

            # Custom placeholder texts for specific categorical inputs
            if label_text == "Sex":
                input_field.setPlaceholderText("Enter Male or Female")
            elif label_text == "Chest Pain Type":
                input_field.setPlaceholderText("ATA, NAP, ASY, or TA")
            elif label_text == "Exercise Angina":
                input_field.setPlaceholderText("Yes (Y) or No (N)")
            elif label_text == "ST_Slope":
                input_field.setPlaceholderText("Up, Flat, or Down")
            else:
                input_field.setPlaceholderText(f"Enter {label_text.split('(')[0].strip()}...")
            self.inputs.append(input_field)

            row = i // 2
            col_offset = (i % 2) * 2
            grid_layout.addWidget(label, row, col_offset)
            grid_layout.addWidget(input_field, row, col_offset + 1)

        main_layout.addLayout(grid_layout)

        main_layout.addLayout(grid_layout)

        self.result_label = QLabel("Awaiting Input")
        self.result_label.setFont(QFont('Inter', 18, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: #28a745; margin-top: 20px;")
        main_layout.addWidget(self.result_label)

        predict_button = QPushButton("Predict possibility of Heart Disease")
        predict_button.clicked.connect(self.make_prediction)
        main_layout.addWidget(predict_button, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)

    def make_prediction(self):
        try:
            input_values = []
            # Redefine labels here to ensure consistency with current UI labels
            labels = [
                "Age", "Sex", "Chest Pain Type", "RestingBP",
                "Cholesterol", "FastingBS (0 or 1)", "RestingECG (0-2)",
                "MaxHR", "Exercise Angina", "Oldpeak", "ST_Slope"
            ]

            for i, field in enumerate(self.inputs):
                text = field.text().strip()
                if not text:
                    raise ValueError(f"Please fill in the '{labels[i]}' field.")

                lower_text = text.lower() # Convert to lowercase for case-insensitive matching

                if labels[i] == "Sex":
                    # Mappings: ['M', 'F'] -> [1, 0]
                    if lower_text in ['female', 'f', '0']:
                        input_values.append(0.0)
                    elif lower_text in ['male', 'm', '1']:
                        input_values.append(1.0)
                    else:
                        raise ValueError("For 'Sex', please enter 'Male', 'Female', 'M', 'F', 0, or 1.")
                elif labels[i] == "Chest Pain Type":
                    # Mappings: ['ATA', 'NAP', 'ASY', 'TA'] -> [0, 1, 2, 3] (assuming an ordering)
                    # Common assumption for this dataset: ATA=0, NAP=1, ASY=2, TA=3
                    if lower_text == 'ata':
                        input_values.append(0.0)
                    elif lower_text == 'nap':
                        input_values.append(1.0)
                    elif lower_text == 'asy':
                        input_values.append(2.0)
                    elif lower_text == 'ta':
                        input_values.append(3.0)
                    else:
                        raise ValueError("For 'Chest Pain Type', please enter 'ATA', 'NAP', 'ASY', or 'TA'.")
                elif labels[i] == "Exercise Angina":
                    # Mappings: ['N', 'Y'] -> [0, 1]
                    if lower_text in ['no', 'n', '0']:
                        input_values.append(0.0)
                    elif lower_text in ['yes', 'y', '1']:
                        input_values.append(1.0)
                    else:
                        raise ValueError("For 'Exercise Angina', please enter 'Yes', 'No', 'Y', 'N', 0, or 1.")
                elif labels[i] == "ST_Slope":
                    # Mappings: ['Up', 'Flat', 'Down'] -> [0, 1, 2] (assuming an ordering)
                    # Common assumption for this dataset: Up=0, Flat=1, Down=2
                    if lower_text == 'up':
                        input_values.append(0.0)
                    elif lower_text == 'flat':
                        input_values.append(1.0)
                    elif lower_text == 'down':
                        input_values.append(2.0)
                    else:
                        raise ValueError("For 'ST_Slope', please enter 'Up', 'Flat', or 'Down'.")
                else:
                    # General numerical conversion for other fields
                    try:
                        input_values.append(float(text))
                    except ValueError:
                        raise ValueError(f"Please enter a valid numerical value for '{labels[i]}'.")

            data = np.array(input_values).reshape(1, -1)
            prediction = model.predict(data)[0]

            result_text = "Positive for Heart Disease" if prediction == 1 else "No Heart Disease"
            prediction_output = f"Prediction: <span style='color: #dc3545;'>{result_text}</span>"
            accuracy_output = f"Model Accuracy: {accuracy * 100:.2f}%"
            self.result_label.setText(f"{prediction_output}\n\n{accuracy_output}")
            

            if prediction == 1:
                self.result_label.setStyleSheet("color: #dc3545; margin-top: 20px; font-weight: bold;")
            else:
                self.result_label.setStyleSheet("color: #28a745; margin-top: 20px; font-weight: bold;")

        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"An unexpected error occurred during prediction: {e}")





if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HeartDiseasePredictor()
    window.show()
    sys.exit(app.exec_())
