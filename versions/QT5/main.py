import sys
import PyQt5
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox, QLabel

def main():
    
    app = QApplication(sys.argv)
    
    window = QWidget()
    window.setWindowTitle("My First PyQt5 App")
    window.setGeometry(100,100,400,200)
    
    label = QLabel("Hello world", parent = window)
    label.move(150, 80)
    
    window.show()
    
    sys.exit(app.exec_())
    

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Button Example')
        self.setGeometry(100, 100, 300, 150)

        layout = QVBoxLayout() # Create a vertical layout

        button = QPushButton('Click Me!')
        button.clicked.connect(self.on_button_click) # Connect the button's clicked signal to a slot (method)

        layout.addWidget(button) # Add the button to the layout
        self.setLayout(layout) # Set the window's layout

    def on_button_click(self):
        QMessageBox.information(self, 'Info', '@+2=2=123')
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    main().show()
    sys.exit(app.exec_())