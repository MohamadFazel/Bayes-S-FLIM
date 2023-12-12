import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt  # Add this line to import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QLineEdit, QFormLayout

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import datetime
import scipy as sc
import os
from modules.sflim import run_sflim_sampler 


class FLIMAnalyzerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.file_path = None
        self.save_path = None
        self.img_size = (128, 128)
        self.slice_params = (54, 118, None), (0, 64, None)

        self.setWindowTitle('FLIM Analyzer GUI')
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.file_label = QLabel('Select File:')
        self.file_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.layout.addWidget(self.file_label)

        self.file_button = QPushButton('Browse')
        self.file_button.clicked.connect(self.showFileDialog)
        self.layout.addWidget(self.file_button)

        self.save_label = QLabel('Select Save Path:')
        self.save_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.layout.addWidget(self.save_label)

        self.save_button = QPushButton('Browse')
        self.save_button.clicked.connect(self.showSaveDialog)
        self.layout.addWidget(self.save_button)

        self.param_label = QLabel('Analysis Parameters:')
        self.param_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.layout.addWidget(self.param_label)

        self.param_layout = QFormLayout()

        self.TInterP_edit = QLineEdit()
        self.param_layout.addRow('Inrepulse Time (s):', self.TInterP_edit)

        self.TauIRF_edit = QLineEdit()
        self.param_layout.addRow('IRF Offset (s):', self.TauIRF_edit)

        self.SigIRF_edit = QLineEdit()
        self.param_layout.addRow('IRF Std (s):', self.SigIRF_edit)

        self.num_species_edit = QLineEdit()
        self.param_layout.addRow('Number of Species:', self.num_species_edit)

        self.NIter_edit = QLineEdit()
        self.param_layout.addRow('Number of Iterations:', self.NIter_edit)

        self.img_size_edit = QLineEdit()
        self.param_layout.addRow('Image Size (e.g., 128x128):', self.img_size_edit)

        self.slice_params_edit = QLineEdit()
        self.param_layout.addRow('Slice Parameters (e.g., 54:118, 0:64):', self.slice_params_edit)

        self.layout.addLayout(self.param_layout)

        self.run_button = QPushButton('Run Analysis')
        self.run_button.clicked.connect(self.runAnalysis)
        self.run_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.layout.addWidget(self.run_button)

        self.setLayout(self.layout)

    
    def showFileDialog(self):
        """Show a file dialog to select a data file."""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setNameFilter("MAT files (*.mat);;All Files (*)")

        if file_dialog.exec_():
            self.file_path = file_dialog.selectedFiles()[0]
            self.file_label.setText(f'Selected File: {self.file_path}')

    def showSaveDialog(self):
        """Show a save dialog to select a directory for saving results."""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        save_dialog = QFileDialog()
        save_dialog.setFileMode(QFileDialog.Directory)

        if save_dialog.exec_():
            self.save_path = save_dialog.selectedFiles()[0]
            self.save_label.setText(f'Selected Save Path: {self.save_path}')

    def run_sflim_analysis(self, dt, lambda_, TInterP, TauIRF, SigIRF, NIter, M):
            """Run the FLIM analysis using the specified parameters."""
            return run_sflim_sampler(dt, lambda_, TInterP, TauIRF, SigIRF, TInterP, NIter, M)

    def runAnalysis(self):
        """Run the FLIM analysis."""
        if self.file_path is None or self.save_path is None:
            return

        # Retrieve parameter values from user input
        TInterP = float(self.TInterP_edit.text())
        TauIRF = float(self.TauIRF_edit.text())
        SigIRF = float(self.SigIRF_edit.text())
        NIter = int(self.NIter_edit.text())
        img_size_str = self.img_size_edit.text().split('x')
        img_size_str = self.img_size_edit.text().split('*')
        self.img_size = (int(img_size_str[0]), int(img_size_str[1]))
        slice_params_str = self.slice_params_edit.text().split(',')
        self.slice_params = tuple(map(lambda x: slice(*map(int, x.split(':'))), slice_params_str))

        # Load data from the specified file
        mix = sc.io.loadmat(self.file_path)
        dt_mix = np.squeeze(mix["Dt"]).reshape(*self.img_size)
        dt_mix = dt_mix[self.slice_params]

        # Calculate dummy_size
        self.dummy_size = dt_mix.shape[1]

        dt = dt_mix.reshape(-1)
        
        lam_mix = mix["Lambda"]
        lam_mix = lam_mix.reshape(-1, self.img_size[1], lam_mix.shape[1])
        lam_mix = lam_mix[self.slice_params]

        lambda_ = lam_mix.reshape(-1, lam_mix.shape[2])

        # Display the image for confirmation
        confirmation = self.confirmationDialog(dt)
        if confirmation == QDialog.Rejected:
            return  # User canceled the operation

        # Assuming you have a function run_sflim_analysis, replace this with your actual analysis code
        pi, photon_int, eta, bg = self.run_sflim_analysis(dt, lambda_, TInterP, TauIRF, SigIRF, NIter, self.num_species)

        # Save the results
        timestr = datetime.now().strftime("%m%d%H%M%S")
        np.save(f"{self.save_path}/Pi_results_{timestr}.npy", pi)
        np.save(f"{self.save_path}/photon_int_results_{timestr}.npy", photon_int)
        np.save(f"{self.save_path}/Eta_results_{timestr}.npy", eta)
        np.save(f"{self.save_path}/Bg_results_{timestr}.npy", bg)

        # Show the results (you can modify this part based on your requirements)
        self.showResults(photon_int, eta, bg)

    def run_sflim_analysis(self, dt, lambda_, TInterP, TauIRF, SigIRF, NIter):
        """Placeholder for your FLIM analysis code."""
        # Replace this with your actual analysis function
        return run_sflim_sampler(dt, lambda_, TInterP, TauIRF, SigIRF, TInterP, NIter, M)
    
    def confirmationDialog(self, image_data):
        """Show a confirmation dialog with the displayed image."""
        dialog = QDialog(self)
        dialog.setWindowTitle('Confirmation Dialog')

        layout = QVBoxLayout()

        # Display the image using matplotlib
        fig, ax = plt.subplots()
        ax.imshow(image_data.reshape(self.img_size), cmap='gray')
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        # Add a confirmation message
        confirmation_label = QLabel('Do you want to proceed with the analysis?')
        layout.addWidget(confirmation_label)

        # Add confirmation buttons
        confirm_button = QPushButton('Yes')
        confirm_button.clicked.connect(dialog.accept)
        layout.addWidget(confirm_button)

        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(dialog.reject)
        layout.addWidget(cancel_button)

        dialog.setLayout(layout)

        # Show the dialog and return the user's choice
        return dialog.exec_()
    
    def showResults(self, photon_int, eta, bg):
        """Show or visualize the FLIM analysis results."""
        # Extract file name without extension
        file_name = os.path.basename(self.file_path)
        name_without_extension = os.path.splitext(file_name)[0]

        # Example: Display intensity images
        num_avg = self.NIter // 8
        phi = np.mean(photon_int[-num_avg:, :, :], axis=0)
        phi = phi.reshape(phi.shape[0], -1, self.dummy_size)
        for it in range(self.num_species):
            plt.imshow(phi[it])
            plt.savefig(f"{self.save_path}/intensity_{name_without_extension}_{it}.png")

        # Example: Display lifetime histograms
        for it in range(self.num_species):
            plt.hist(1 / eta[-num_avg:, 0], bins=100, color="red", label="Viafluor")
            plt.savefig(f"{self.save_path}/lifetime_{name_without_extension}_{it}.png")

        pin = np.mean(pi[-num_avg:,:,:], axis=0)
        for it in range(self.num_species):
            plt.plot(pin[1]/np.sum(pin[1]),'r', label="2_Learned")
            plt.savefig(f"{self.save_path}/spectrum_{name_without_extension}_{it}.png")
        plt.plot(bg[-num_avg:])
        plt.savefig(f"{self.save_path}/bg_{name_without_extension}_{it}.png")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FLIMAnalyzerGUI()
    ex.show()
    sys.exit(app.exec_())
