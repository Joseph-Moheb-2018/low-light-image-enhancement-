##Importing Required libraries for GUI
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
import os
import sys
from os import path

##Importing Required libraries for the code
import numpy as np
import cv2
import imageio
from tensorflow.keras.models import load_model


#loading path of GUI design
FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "main2.ui"))

#main class of GUI
class Main(QWidget, FORM_CLASS):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        QWidget.__init__(self)
        self.setupUi(self)
        self.Handler_Ui()
        self.Handel_Buttons()
        
    #Buttons Handler
    def Handel_Buttons(self):
        self.upload.clicked.connect(self.GetImageFile)
        self.tryAgain.clicked.connect(self.TryAgain)
        self.cl.clicked.connect(self.close)
        self.Save.clicked.connect(self.Saving)
        self.enhance.clicked.connect(self.Handel_Enhance)
        
    #First Handler function in the init
    def Handler_Ui(self):
        self.setWindowTitle("Low Light Image Enhancement")


    def Handel_Enhance(self):
        Input = imageio.imread(self.name)   #Reading the input image
        
        self.image = output                    #have variable holding the output image

        height, width, channel = output.shape
        bytesPerLine = 3 * width
        qImg = QImage(output.data, width, height, bytesPerLine, QImage.Format_RGB888)

        w = self.labelImageIn.width()
        h = self.labelImageIn.height()
        self.labelImageOut.setPixmap(QPixmap(qImg).scaled(w, h))           #show output image inside the GUI

    def GetImageFile(self):
        file_name, _ = QFileDialog.getOpenFileName(self , 'Open Image File',r"C:\\Users\\m_rab\\Desktop\\","Image files (*.jpg *.jpeg *.png)")      #Upload the photo from file system
        self.name = file_name      #have variable holding the path of the input image
        w = self.labelImageIn.width()
        h = self.labelImageIn.height()
        self.labelImageIn.setPixmap(QPixmap(file_name).scaled(w,h))       #show uploaded image in the GUI

    def Saving(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save As", "." , "Image files (*.jpg *.jpeg *.png)")      #choose path of the saved image
        cv2.imwrite(filename,  cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))   #saving image to the specified path

    def TryAgain(self):
        self.labelImageIn.clear()     #clear the QLabel holding the imput image
        self.labelImageOut.clear()    #clear the QLabel holding the output image

    def close(self):
        os._exit(0)

def main():
    app = QApplication(sys.argv)
    Window = Main()
    Window.show()
    app.exec_()

if __name__ == "__main__":
    main()