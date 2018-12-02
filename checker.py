# SWAMI KARUPPASWAMI THUNNAI
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import sys
import numpy as np
from glob import glob


def predict(image_file):
    arr = np.array(img_to_array(load_img(image_file, target_size=(128, 128), grayscale=True)))
    #print(arr)
    result = classifier.predict_classes([[arr,]])
    result = result[0]
    result += 1
    print(result)
    folder_name = "Fnt/Sample"+str(result).zfill(3)
    files = glob(folder_name+"/*.*")
    image = files[0]
    return image


class Widget(QWidget):


    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prediction")
        self.setGeometry(300, 300, 390, 390)
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("Original"))
        row1 = QHBoxLayout()
        label1 = QLabel()
        label2 = QLabel()
        label3 = QLabel()
        row1.addWidget(label1)
        row1.addWidget(label2)
        row1.addWidget(label3)
        label1.setPixmap(QPixmap("check/1.png"))
        label2.setPixmap(QPixmap("check/2.png"))
        label3.setPixmap(QPixmap("check/3.png"))
        main_layout.addLayout(row1)
        main_layout.addWidget(QLabel("Prediction using CNN:"))
        row2 = QHBoxLayout()
        label11 = QLabel()
        label21 = QLabel()
        label31 = QLabel()
        row2.addWidget(label11)
        row2.addWidget(label21)
        row2.addWidget(label31)
        label11.setPixmap(QPixmap(predict("check/1.png")))
        label21.setPixmap(QPixmap(predict("check/2.png")))
        label31.setPixmap(QPixmap(predict("check/3.png")))
        main_layout.addLayout(row2)
        self.setLayout(main_layout)


if __name__ == "__main__":
    classifier = load_model("model.h5")
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec())
