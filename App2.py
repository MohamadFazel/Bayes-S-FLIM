import numpy as np
from modules.sflim import run_sflim_sampler
from modules.forward import gen_data
import scipy.io as sio
from PyQt5 import QtWidgets
from PyQt5 import uic
import sys
from qt_material import apply_stylesheet
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout
import matplotlib.pylab as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QPixmap, QImage

# plt.style.use('seaborn-v0_8-dark-palette')
# print(plt.style.available)
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.file_path = ""
        self.save_path = ""
        self.img_size = (128, 128)
        self.slice_params = slice(54, 118, None), slice(0, 64, None)
        
        self.ui = uic.loadUi('ui/app.ui', self)
        # Ensure galleryWidget has a layout set
        if self.galleryWidget.layout() is None:
            self.galleryWidget.setLayout(QVBoxLayout())  # Assuming a QVBoxLayout, change as needed

        self.ui.load_data.clicked.connect(self.browseFile)
        self.ui.save_path.clicked.connect(self.browseSavePath)
        self.ui.run_analy.clicked.connect(self.runAnalysis)
        self.ui.num_species.setText(str(3))
        self.ui.intepulse_time.setText(str(12.85))
        self.ui.sig_irf.setText(str(0.51))
        self.ui.tau_irf.setText(str(2.506))
        self.ui.num_iter.setText(str(200000))
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.galleryWidget.layout().addWidget(self.canvas)
        # Call a function to update the plot
        


        # self.ui.num_species.setText(str(128),str(128))
    def browseFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("MAT files (*.mat)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            self.file_path = file_dialog.selectedFiles()[0]
            self.file_data = sio.loadmat(self.file_path)
            #Extract time-related information
            self.time_data = np.squeeze(self.file_data["Dt"]).reshape(128,128)
            self.time_data = self.time_data[self.slice_params]
            dummy_size = self.time_data.shape[1]
            time_values = self.time_data.reshape(-1)
            photon_count = [len(time_values[i]) for i in range(len(time_values))]
            photon_count = np.array(photon_count).reshape(-1, dummy_size)
            self.update_plot(photon_count)


    def browseSavePath(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        save_dialog = QFileDialog()
        save_dialog.setFileMode(QFileDialog.Directory)
        if save_dialog.exec_():
            self.save_path = save_dialog.selectedFiles()[0]

    def runAnalysis(self):
        # Get updated parameter values from the GUI
        self.t_inter_pulse = self.ui.intepulse_time.text()
        self.tau_irf = self.ui.tau_irf.text()
        self.sig_irf = self.ui.sig_irf.text()
        self.num_species = self.ui.num_iter.text()
        self.n_iter = self.ui.num_iter.text()  # Corrected the variable name
        # self.img_size = self.ui.intepulse_time.text()

        # Extract wavelength-related information
        wavelength_data = self.file_data["Lambda"]
        wavelength_data = wavelength_data.reshape(-1, self.img_size[1], wavelength_data.shape[1])
        wavelength_data = wavelength_data[self.slice_params]
        self.wavelength_values = wavelength_data.reshape(-1, wavelength_data.shape[2])
        
        self.pi, self.photon_int, self.eta, self.bg = run_sflim_sampler(
            self.time_values,  # Assuming dt corresponds to time values
            self.wavelength_values,  # Assuming lambda_ corresponds to wavelength values
            self.t_inter_pulse,
            self.tau_irf,
            self.sig_irf,
            self.n_iter,  # Assuming NIter corresponds to the number of iterations
            self.num_species
        )

    def update_plot(self, img_data):
        # Your Matplotlib plotting code here
        self.ax.imshow(img_data, cmap='gray')
        self.canvas.draw()

app = QtWidgets.QApplication(sys.argv)
mainWindow = MainWindow()

apply_stylesheet(app, 'dark_amber.xml')

mainWindow.show()
sys.exit(app.exec_())
