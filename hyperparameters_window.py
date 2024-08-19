from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import pyqtSignal
import os
from utils import set_default_values


class HyperparametersWindow(QtWidgets.QWidget):
    parameters_set = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        uic.loadUi(os.path.join("ui", "hyperr.ui"), self)
        self.setWindowTitle("Hyperparameters")
        self.setup_ui()
        self.set_default_values()

    def setup_ui(self):
        self.alpha_prop_life = self.findChild(QtWidgets.QLineEdit, "alpha_prop_life")
        self.alpha_prior_life = self.findChild(QtWidgets.QLineEdit, "alpha_prior_life")
        self.beta_prior_life = self.findChild(QtWidgets.QLineEdit, "beta_prior_life")
        self.alpha_prop_int = self.findChild(QtWidgets.QLineEdit, "alpha_prop_int")
        self.alpha_prior_int = self.findChild(QtWidgets.QLineEdit, "alpha_prior_int")
        self.beta_prior_int = self.findChild(QtWidgets.QLineEdit, "beta_prior_int")
        self.alpha_prop_pi = self.findChild(QtWidgets.QLineEdit, "alpha_prop_pi")
        self.apply_button = self.findChild(QtWidgets.QPushButton, "apply_button")
        self.apply_button.clicked.connect(self.send_parameters)

    def set_default_values(self):
        default_values = set_default_values()
        self.alpha_prop_life.setText(str(default_values["alpha_prop_life"]))
        self.alpha_prior_life.setText(str(default_values["alpha_prior_life"]))
        self.beta_prior_life.setText(str(default_values["beta_prior_life"]))
        self.alpha_prop_int.setText(str(default_values["alpha_prop_int"]))
        self.alpha_prior_int.setText(str(default_values["alpha_prior_int"]))
        self.beta_prior_int.setText(str(default_values["beta_prior_int"]))
        self.alpha_prop_pi.setText(str(default_values["alpha_prop_pi"]))

    def send_parameters(self):
        params = {
            "alpha_prop_life": int(self.alpha_prop_life.text()),
            "alpha_prior_life": int(self.alpha_prior_life.text()),
            "beta_prior_life": int(self.beta_prior_life.text()),
            "alpha_prop_int": int(self.alpha_prop_int.text()),
            "alpha_prior_int": int(self.alpha_prior_int.text()),
            "beta_prior_int": int(self.beta_prior_int.text()),
            "alpha_prop_pi": int(self.alpha_prop_pi.text()),
        }
        self.parameters_set.emit(params)
        self.close()
