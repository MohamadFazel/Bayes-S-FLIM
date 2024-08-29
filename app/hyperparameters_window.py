from PyQt6 import QtWidgets as qt, uic
from PyQt6.QtCore import pyqtSignal


class HyperparametersWindow(qt.QWidget):
    parameters_set = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hyperparameters")
        self.setup_ui()
        self.set_default_values()

    def setup_ui(self):
        uic.loadUi("ui/hyperr.ui", self)

        # Find and connect UI elements
        self.alpha_prop_life = self.findChild(qt.QLineEdit, "alpha_prop_life")
        self.alpha_prior_life = self.findChild(qt.QLineEdit, "alpha_prior_life")
        self.beta_prior_life = self.findChild(qt.QLineEdit, "beta_prior_life")

        self.alpha_prop_int = self.findChild(qt.QLineEdit, "alpha_prop_int")
        self.alpha_prior_int = self.findChild(qt.QLineEdit, "alpha_prior_int")
        self.beta_prior_int = self.findChild(qt.QLineEdit, "beta_prior_int")

        self.alpha_prop_pi = self.findChild(qt.QLineEdit, "alpha_prop_pi")

        self.apply_button = self.findChild(qt.QPushButton, "apply_button")
        self.apply_button.clicked.connect(self.send_parameters)

        # Add tooltips for each parameter
        self.alpha_prop_life.setToolTip("Proposal distribution parameter for lifetime")
        self.alpha_prior_life.setToolTip(
            "Prior distribution alpha parameter for lifetime"
        )
        self.beta_prior_life.setToolTip(
            "Prior distribution beta parameter for lifetime"
        )
        self.alpha_prop_int.setToolTip("Proposal distribution parameter for intensity")
        self.alpha_prior_int.setToolTip(
            "Prior distribution alpha parameter for intensity"
        )
        self.beta_prior_int.setToolTip(
            "Prior distribution beta parameter for intensity"
        )
        self.alpha_prop_pi.setToolTip("Proposal distribution parameter for spectra")

    def set_default_values(self):
        self.alpha_prop_life.setText(str(1000))
        self.alpha_prior_life.setText(str(1))
        self.beta_prior_life.setText(str(10))

        self.alpha_prop_int.setText(str(5000))
        self.alpha_prior_int.setText(str(1))
        self.beta_prior_int.setText(str(2100))

        self.alpha_prop_pi.setText(str(5000))

    def send_parameters(self):
        try:
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
        except ValueError:
            qt.QMessageBox.critical(self, "Error", "All values must be valid integers.")

    def show_help(self):
        help_text = """
        Hyperparameters Explanation:

        For Lifetime:
        - alpha_prop_life: Controls the spread of the proposal distribution for lifetime sampling.
        - alpha_prior_life and beta_prior_life: Shape and scale parameters for the gamma prior on lifetimes.

        For Intensity:
        - alpha_prop_int: Controls the spread of the proposal distribution for intensity sampling.
        - alpha_prior_int and beta_prior_int: Shape and scale parameters for the gamma prior on intensities.

        For Spectra:
        - alpha_prop_pi: Controls the spread of the proposal distribution for spectra sampling.

        Higher values for proposal parameters (alpha_prop_*) lead to smaller step sizes in the MCMC sampling.
        Adjust prior parameters to reflect your prior beliefs about the parameter distributions.
        """
        qt.QMessageBox.information(self, "Hyperparameters Help", help_text)

    def closeEvent(self, event):
        result = qt.QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit? Unsaved changes will be lost.",
            qt.QMessageBox.StandardButton.Yes | qt.QMessageBox.StandardButton.No,
        )
        if result == qt.QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()
