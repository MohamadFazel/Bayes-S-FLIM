import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QProgressBar, QSizePolicy, QLineEdit, QFormLayout
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QFormLayout, QLineEdit, QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QPixmap

from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from datetime import datetime
from modules.sflim import run_sflim_sampler
from modules.forward import gen_data
import scipy.io as sio
import os
import time
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
import qdarktheme

import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QFormLayout, QLineEdit, QFrame
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from datetime import datetime
import scipy.io as sio
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QFormLayout, QLineEdit,
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from datetime import datetime
import scipy.io as sio
# from PySide2.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit


def apply_stylesheet(app):
    app.setStyle("Fusion")

    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.Highlight, QColor(142, 45, 197))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

class FLIMAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()

        self.file_path = ""
        self.save_path = ""
        self.TInterP = 12.85
        self.TauIRF = 2.506
        self.SigIRF = 0.51
        self.num_species = 3
        self.NIter = 200000
        self.img_size = (128, 128)
        self.slice_params = slice(54, 118, None), slice(0, 64, None)

        self.initUI()

    def initUI(self):
        apply_stylesheet(QApplication.instance())

        self.setWindowTitle('FLIM Analyzer')
        self.setGeometry(100, 100, 800, 600)

        # Widgets
        self.fileLabel = QLabel('Select File:')
        self.fileButton = QPushButton(QIcon('upload.png'), 'Browse')
        self.fileButton.clicked.connect(self.browseFile)

        self.saveLabel = QLabel('Save Path:')
        self.saveButton = QPushButton(QIcon('upload.png'), 'Browse')
        self.saveButton.clicked.connect(self.browseSavePath)

        # Arrange file and save path horizontally
        fileSaveLayout = QHBoxLayout()
        fileSaveLayout.addWidget(self.fileLabel)
        fileSaveLayout.addWidget(self.fileButton)
        fileSaveLayout.addWidget(self.saveLabel)
        fileSaveLayout.addWidget(self.saveButton)

        self.paramsLabel = QLabel('Analysis Parameters:')
        self.paramsGridLayout = QGridLayout()

        # Add parameters with two in each line
        self.addParameter('Inrepulse Time:', 'TInterP', self.TInterP, placeholder_color="#FFD700", row=0, col=0)
        self.addParameter('IRF Offset:', 'TauIRF', self.TauIRF, placeholder_color="#FFD700", row=0, col=2)
        self.addParameter('IRF Std:', 'SigIRF', self.SigIRF, placeholder_color="#FFD700", row=1, col=0)
        self.addParameter('Number of Species:', 'num_species', self.num_species, placeholder_color="#FFD700", row=1, col=2)
        self.addParameter('Number of Iterations:', 'NIter', self.NIter, placeholder_color="#FFD700", row=2, col=0)
        self.addParameter('Image Size (e.g., 128x128):', 'img_size', str(self.img_size), placeholder_color="#FFD700", row=2, col=2)
        self.addParameter('Slice Parameters (e.g., 54:118, 0:64):', 'slice_params', str(self.slice_params), placeholder_color="#FFD700", row=3, col=0, col_span=3)

        self.displayButton = QPushButton('Display Image')
        self.displayButton.clicked.connect(self.displayImage)
        self.displayButton.setStyleSheet("background-color: #34A853; color: white;")

        self.runButton = QPushButton('Run Analysis')
        self.runButton.clicked.connect(self.runAnalysis)
        self.runButton.setStyleSheet("background-color: #4285F4; color: white;")

        self.plotLabel = QLabel('Data Plot:')
        self.frame =QFrame()
        self.frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.frame.setLineWidth(2)
        self.canvas = MatplotlibCanvas(self.frame)

        # Layout
        layout = QVBoxLayout()
        layout.addLayout(fileSaveLayout)
        layout.addWidget(self.paramsLabel)
        layout.addLayout(self.paramsGridLayout)

        # Button layout
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.displayButton)
        buttonLayout.addWidget(self.runButton)

        layout.addLayout(buttonLayout)
        layout.addWidget(self.plotLabel)
        
        layout2 = QHBoxLayout()
        layout2.addLayout(layout)
        layout2.addWidget(self.canvas)
        layout2.setStretchFactor(layout,1)
        layout2.setStretchFactor(self.canvas,4)
        self.setLayout(layout2)

    def addParameter(self, label, name, value, placeholder_color=None, row=None, col=None, col_span=None):
        labelWidget = QLabel(label)
        valueWidget = QLineEdit(str(value))
        valueWidget.setObjectName(name)

        # Set the placeholder text color
        if placeholder_color:
            style_sheet = f"color: {placeholder_color};"
            valueWidget.setStyleSheet(style_sheet)

        if row is not None and col is not None:
            self.paramsGridLayout.addWidget(labelWidget, row, col, 1, 1)
            self.paramsGridLayout.addWidget(valueWidget, row, col + 1, 1, col_span or 1)
        else:
            self.paramsGridLayout.addWidget(labelWidget)
            self.paramsGridLayout.addWidget(valueWidget)

    def browseFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("MAT files (*.mat)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            self.file_path = file_dialog.selectedFiles()[0]

    def browseSavePath(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        save_dialog = QFileDialog()
        save_dialog.setFileMode(QFileDialog.Directory)
        if save_dialog.exec_():
            self.save_path = save_dialog.selectedFiles()[0]

    def displayImage(self):
        if not self.file_path:
            return

        mix = sio.loadmat(self.file_path)
        dt_mix = np.squeeze(mix["Dt"]).reshape(*self.img_size)
        dt_mix = dt_mix[self.slice_params]
        dummy_size = dt_mix.shape[1]

        dt = dt_mix.reshape(-1)
        l = [len(dt[i]) for i in range(len(dt))]
        l = np.array(l).reshape(-1, dummy_size)
        self.canvas.plotData(l)

    def runAnalysis(self):
        # Get updated parameter values from the GUI
        self.TInterP = float(self.findChild(QLineEdit, 'TInterP').text())
        self.TauIRF = float(self.findChild(QLineEdit, 'TauIRF').text())
        self.SigIRF = float(self.findChild(QLineEdit, 'SigIRF').text())
        self.num_species = int(self.findChild(QLineEdit, 'num_species').text())
        self.NIter = int(self.findChild(QLineEdit, 'NIter').text())

        # Update img_size and slice_params from the GUI
        img_size_str = self.findChild(QLineEdit, 'img_size').text()
        slice_params_str = self.findChild(QLineEdit, 'slice_params').text()


        try:
            self.img_size = tuple(map(int, img_size_str.split('x')))
            self.slice_params = tuple(slice(map(int, s.split(':'))) for s in slice_params_str.split(','))
        except ValueError:
            # Handle invalid input gracefully
            print("Invalid input for img_size or slice_params")


        if not self.file_path or not self.save_path:
            return

        mix = sio.loadmat(self.file_path)
        dt_mix = np.squeeze(mix["Dt"]).reshape(*self.img_size)
        dt_mix = dt_mix[self.slice_params]
        dummy_size = dt_mix.shape[1]

        dt = dt_mix.reshape(-1)
        l = [len(dt[i]) for i in range(len(dt))]
        l = np.array(l).reshape(-1, dummy_size)
        self.canvas.plotData(l)
        self.canvas.draw()


        lam_mix = mix["Lambda"]
        lam_mix = lam_mix.reshape(-1, self.img_size[1], lam_mix.shape[1])
        lam_mix = lam_mix[self.slice_params]

        lambda_ = lam_mix.reshape(-1, lam_mix.shape[2])

        t0 = datetime.now()
        timestr = time.strftime("%m%d%H%M%S")

        pi, photon_int, eta, bg = run_sflim_sampler(dt, lambda_, self.TInterP, self.TauIRF, self.SigIRF, self.TInterP, self.NIter, self.num_species)

        np.save(f"{self.save_path}/Pi_{timestr}.npy", pi)
        np.save(f"{self.save_path}/int_{timestr}.npy", photon_int)
        np.save(f"{self.save_path}/Eta_{timestr}.npy", eta)

        num_avg = self.NIter // 8
        phi = np.mean(photon_int[-num_avg:, :, :], axis=0)
        phi = phi.reshape(phi.shape[0], -1, dummy_size)
        for it in range(self.num_species):
            self.canvas.plotData(phi[it])
            self.canvas.draw()

        for it in range(self.num_species):
            self.canvas.plotHistogram(1 / eta[-num_avg:, 0], bins=100, color="red", label="Viafluor")
            self.canvas.draw()

        pin = np.mean(pi[-num_avg:, :, :], axis=0)
        for it in range(self.num_species):
            self.canvas.plotData(pin[1] / np.sum(pin[1]), 'r', label="2_Learned")
            self.canvas.draw()

        self.canvas.plotData(bg[-num_avg:])
        self.canvas.draw()


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

    def plotData(self, image):
        self.ax.clear()
        self.ax.imshow(image, cmap='gray')  # Use the appropriate cmap for your data
        self.draw()

class MyWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = MatplotlibCanvas(self)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def updateImage(self, image):
        self.canvas.showImage(image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # qdarktheme.setup_theme("dark")

    ex = FLIMAnalyzerApp()
    ex.show()
    sys.exit(app.exec_())
