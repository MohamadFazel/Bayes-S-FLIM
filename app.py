import pickle
import sys
import PyQt6.QtWidgets as qt
from PyQt6 import uic
from PyQt6.QtCore import pyqtSignal
import numpy as np
import cupy as cp
from datetime import datetime
from matplotlib import cm

# from cupyx.scipy import special
from scipy import special
import scipy.stats as sc
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class PlotWindow(qt.QWidget):
    def __init__(self, eta, photon_int, pi, select_plot, nsb, img_sz, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Results")

        self.eta = eta
        self.photon_int = photon_int
        self.pi = pi
        self.select_plot = select_plot
        self.nsb = nsb
        self.img_sz = img_sz

        # Create a layout for the plot
        layout = qt.QVBoxLayout()
        self.setLayout(layout)

        # Create a Matplotlib figure and canvas
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Define a colormap
        self.colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "black",
            "magenta",
            "yellow",
            "brown",
            "pink",
        ]
        self.cmaps = [
            "Reds",
            "Blues",
            "Greens",
            "Oranges",
            "Purples",
            "Greys",
            "spring",
            "Wistia",
            "copper",
            "cool",
        ]

        # Plot the data
        self.plot_results()

    def plot_results(self):
        self.figure.clear()
        if self.select_plot == "Spectra":
            ax = self.figure.add_subplot(1, 1, 1)
            pin = np.mean(self.pi[-20000:, :, :], axis=0)
            x = np.linspace(375, 760, self.nsb)
            for ii in range(pin.shape[0]):
                color = self.colors[ii % len(self.colors)]
                plt.plot(
                    x, pin[ii] / np.sum(pin[ii]), color=color, label=f"Species #{ii+1}"
                )
            ax.set_title(f"Species Spectra")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Distribution")
            ax.legend()
            self.canvas.draw()
        elif self.select_plot == "Lifetime Histogram":
            color = self.colors[ii % len(self.colors)]
            ax = self.figure.add_subplot(1, 1, 1)
            for ii in range(self.eta.shape[0]):
                plt.hist(
                    1 / self.eta[-20000:, ii],
                    bins=100,
                    color=color,
                    label=f"Species #{ii+1}",
                    density=True,
                )
            ax.set_title(f"Lifetimes Histogram")
            ax.set_xlabel("Lifetime (ns)")
            ax.set_ylabel("Distribution")
            ax.legend()
            self.canvas.draw()
        elif self.select_plot == "Maps":
            phi = np.mean(self.photon_int[-20000:, :, :], axis=0)
            phi = phi.reshape(phi.shape[0], -1, self.img_sz)

            num_images = phi.shape[0]
            cols = 3  # Number of columns
            rows = (num_images // cols) + int(num_images % cols != 0)
            for ii in range(phi.shape[0]):
                cmap = self.cmaps[ii % len(self.cmaps)]
                ax = self.figure.add_subplot(rows, cols, ii + 1)
                plt.imshow(phi[ii], cmap=cmap)
                ax.set_title(f"Species #{ii + 1}")
                ax.axis("off")
            self.figure.suptitle("Maps")
            self.canvas.draw()


class HyperparametersWindow(qt.QWidget):
    parameters_set = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        uic.loadUi("ui/hyperr.ui", self)
        self.setWindowTitle("Hyperparameters")
        self.alpha_prop_life = self.findChild(qt.QLineEdit, "alpha_prop_life")
        self.alpha_prior_life = self.findChild(qt.QLineEdit, "alpha_prior_life")
        self.beta_prior_life = self.findChild(qt.QLineEdit, "beta_prior_life")

        self.alpha_prop_int = self.findChild(qt.QLineEdit, "alpha_prop_int")
        self.alpha_prior_int = self.findChild(qt.QLineEdit, "alpha_prior_int")
        self.beta_prior_int = self.findChild(qt.QLineEdit, "beta_prior_int")

        self.alpha_prop_pi = self.findChild(qt.QLineEdit, "alpha_prop_pi")

        self.apply_button = self.findChild(qt.QPushButton, "apply_button")
        self.apply_button.clicked.connect(self.send_parameters)

        # Set default values
        self.set_default_values()

    def set_default_values(self):

        self.alpha_prop_life.setText(str(1000))
        self.alpha_prior_life.setText(str(1))
        self.beta_prior_life.setText(str(10))

        self.alpha_prop_int.setText(str(5000))
        self.alpha_prior_int.setText(str(1))
        self.beta_prior_int.setText(str(2100))

        self.alpha_prop_pi.setText(str(5000))

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


class MainWindow(qt.QMainWindow):

    def __init__(self):
        super().__init__()
        self.file_path = ""
        self.ui = uic.loadUi("ui/app.ui", self)
        self.setWindowTitle("Spectral FLIM Analysis")
        self.ui.progressBar.setVisible(False)
        self.ui.load_data.clicked.connect(self.browseFile)
        self.ui.run_.clicked.connect(self.runAnalysis)
        self.ui.HyperParameters_.triggered.connect(self.openHyperparameters)
        # Assuming you have a button in your UI named 'plot_button'
        self.ui.plot_.clicked.connect(self.show_plot)
        self.ui.plot_.setEnabled(False)

        self.ui.save_results.clicked.connect(self.export_results)
        self.ui.save_results.setEnabled(False)

        self.analysis_results = True
        self.set_default_values()

    def set_default_values(self):
        self.n_iter = 10
        self.num_species.setText(str(3))
        self.ui.t_inter_p.setText(str(12.85))
        self.ui.irf_sigma.setText(str(0.51))
        self.ui.irf_tau.setText(str(2.506))
        self.ui.num_iter.setText(str(200000))
        self.ui.img_size.setText(str((128, 128)))
        self.dt_ = None
        self.lambda_ = None

        # Default hyperparameters
        self.alpha_prop_life = 1000
        self.alpha_prior_life = 1
        self.beta_prior_life = 10
        self.alpha_prop_int = 5000
        self.alpha_prior_int = 1
        self.beta_prior_int = 2100
        self.alpha_prop_pi = 5000

    def get_values(self):
        self.num_species = int(self.ui.num_species.text())
        self.num_iter = int(self.ui.num_iter.text())
        self.t_inter_p = float(self.ui.t_inter_p.text())
        self.tau_irf = float(self.ui.irf_tau.text())
        self.sig_irf = float(self.ui.irf_sigma.text())

    def browseFile(self):
        options = qt.QFileDialog.Option.DontUseNativeDialog
        file_dialog = qt.QFileDialog(self)
        file_dialog.setOptions(options)
        filters = "MAT Files (*.mat);;JSON Files (*.json)"
        file_name, _ = file_dialog.getOpenFileName(
            self, "Open File", "", filters, options=options
        )
        if file_name:
            if file_name.endswith(".mat"):
                file_data = loadmat(file_name)
            self.dt_ = file_data["Dt"]
            if self.dt_.ndim > 1:
                self.dt_ = np.squeeze(self.dt_)
            self.lambda_ = file_data["Lambda"]

    def openHyperparameters(self):
        self.hyperparameters_window = HyperparametersWindow()
        self.hyperparameters_window.parameters_set.connect(self.receiveParameters)
        self.hyperparameters_window.show()

    def receiveParameters(self, params):
        self.alpha_prop_life = params.get("alpha_prop_life", self.alpha_prop_life)
        self.alpha_prior_life = params.get("alpha_prior_life", self.alpha_prior_life)
        self.beta_prior_life = params.get("beta_prior_life", self.beta_prior_life)
        self.alpha_prop_int = params.get("alpha_prop_int", self.alpha_prop_int)
        self.alpha_prior_int = params.get("alpha_prior_int", self.alpha_prior_int)
        self.beta_prior_int = params.get("beta_prior_int", self.beta_prior_int)
        self.alpha_prop_pi = params.get("alpha_prop_pi", self.alpha_prop_pi)

    def init_chain(self):
        self.n_pix = self.lambda_.shape[0]
        self.nsb = self.lambda_.shape[-1]
        self.eta = np.zeros((self.num_iter, self.num_species))
        self.pi = np.random.rand(self.num_iter, self.num_species, self.nsb)
        self.photon_int = np.zeros((self.num_iter, self.num_species, self.n_pix))
        self.eta[0, :] = np.random.rand(self.num_species)
        for mm in range(self.num_species):
            self.pi[0, mm, :] /= np.sum(self.pi[0, mm, :])
        for ii in range(self.n_pix):
            self.photon_int[0, 0 : self.num_species, ii] = np.random.gamma(
                1, 1500, size=self.num_species
            )

    def sample_int(self, it_, numerator):
        if it_ < (numerator + 1):
            i_old = self.photon_int[0, :, :].copy()
            pi = self.pi[0, :, :].copy()
            eta = self.eta[0, :].copy()
        else:
            i_old = self.photon_int[it_ - numerator - 1, :, :].copy()
            pi = self.pi[it_ - numerator - 1, :, :].copy()
            eta = self.eta[it_ - numerator - 1, :].copy()

        i_new = i_old.copy()
        i_new[:, :] = np.random.gamma(self.alpha_prop_int, i_old / self.alpha_prop_int)
        lf_top = self.calculate_lifetime_likelihood_gpu_int(i_new, eta)
        lf_bot = self.calculate_lifetime_likelihood_gpu_int(i_old, eta)

        tmp_top = np.sum((i_new[:, :, None] * pi[:, None, :]), axis=0)
        a_top = np.sum(sc.poisson.logpmf(self.lambda_, tmp_top), axis=1)
        a_top[np.isnan(a_top)] = 0
        a_top[np.abs(a_top) == np.inf] = 0

        tmp_bot = np.sum((i_old[:, :, None] * pi[:, None, :]), axis=0)
        a_bottom = np.sum(sc.poisson.logpmf(self.lambda_, tmp_bot), axis=1)
        a_bottom[np.isnan(a_bottom)] = 0
        a_bottom[np.abs(a_bottom) == np.inf] = 0

        a_prior = np.sum(
            sc.gamma.logpdf(
                i_new[:, :], self.alpha_prior_int, scale=self.beta_prior_int
            ),
            axis=0,
        ) - np.sum(
            sc.gamma.logpdf(
                i_old[:, :], self.alpha_prior_int, scale=self.beta_prior_int
            ),
            axis=0,
        )
        a_prop = np.sum(
            sc.gamma.logpdf(
                i_old[:, :],
                self.alpha_prop_int,
                scale=(i_new[:, :] / self.alpha_prop_int),
            ),
            axis=0,
        ) - np.sum(
            sc.gamma.logpdf(
                i_new[:, :],
                self.alpha_prop_int,
                scale=(i_old[:, :] / self.alpha_prop_int),
            ),
            axis=0,
        )

        a = (a_top - a_bottom) + (lf_top - lf_bot) + a_prop + a_prior

        cond = a > np.log(np.random.rand(self.n_pix))[None, :]
        i_old = np.where(cond, i_new, i_old)
        self.accept_i += np.sum(cond.astype(int)) * (1 / self.n_pix)
        if it_ < (numerator + 1):
            self.photon_int[0, :, :] = i_old.copy()
        else:
            self.photon_int[it_ - numerator, :, :] = i_old.copy()

    def sample_photon_spectra(self, it_, numerator):
        if it_ < (numerator + 1):
            photon_int = self.photon_int[0, :, :].copy()
            pi_old = self.pi[0, :, :].copy()
        else:
            photon_int = self.photon_int[it_ - numerator, :, :].copy()
            pi_old = self.pi[it_ - numerator - 1, :, :].copy()

        n_species, n_channel = pi_old.shape
        alpha = np.ones(n_channel) / n_channel

        pi_new = pi_old.copy()

        m = np.random.choice(n_species)

        pi_new[m, :] = np.random.gamma(
            self.alpha_prop_pi, pi_old[m, :] / self.alpha_prop_pi
        )
        pi_new[m] /= np.sum(pi_new[m])
        tmp_top = (photon_int[:, :, None] * pi_new[:, None, :]).sum(axis=0)
        tmp_t = tmp_top[tmp_top > 0.001]
        lam = self.lambda_[tmp_top > 0.001]
        a_top = sc.poisson.logpmf(lam, tmp_t).sum()

        tmp_bot = (photon_int[:, :, None] * pi_old[:, None, :]).sum(axis=0)
        tmp_b = tmp_bot[tmp_bot > 0.001]
        lam = self.lambda_[tmp_bot > 0.001]
        a_bottom = sc.poisson.logpmf(lam, tmp_b).sum()

        # Calculating priors and proposal distributions
        a_prop = 0
        a_prior = 0
        for mm in range(pi_old.shape[0]):
            a_prop += np.sum(
                sc.dirichlet.logpdf(pi_old[mm, :], pi_new[mm, :])
            ) - np.sum(sc.dirichlet.logpdf(pi_new[mm, :], pi_old[mm, :]))
            a_prior += np.sum(
                sc.dirichlet.logpdf(pi_new[mm, :], alpha / n_channel)
            ) - np.sum(sc.dirichlet.logpdf(pi_old[mm, :], alpha / n_channel))

        a = (a_top - a_bottom) + a_prop + a_prior
        if a > np.log(np.random.rand()):
            pi_old = pi_new
            self.accept_pi += 1

        if it_ < (numerator + 1):
            self.pi[0, :, :] = pi_old.copy()
        else:
            self.pi[it_ - numerator, :, :] = pi_old.copy()

    def sample_lifetime(self, it_, numerator):
        if it_ < (numerator + 1):
            photon_int = self.photon_int[0, :, :].copy()
            eta_old = self.eta[0, :].copy()
        else:
            photon_int = self.photon_int[it_ - numerator, :, :].copy()
            eta_old = self.eta[it_ - numerator - 1, :].copy()

        eta_prop = np.random.gamma(
            self.alpha_prior_life, eta_old / self.alpha_prior_life
        )

        lf_top = self.calculate_lifetime_likelihood_gpu(photon_int, eta_prop)
        lf_bot = self.calculate_lifetime_likelihood_gpu(photon_int, eta_old)
        lik_ratio = lf_top - lf_bot

        a_prior = np.sum(
            sc.gamma.logpdf(eta_prop, self.alpha_prior_life, scale=self.beta_prior_life)
        ) - np.sum(
            sc.gamma.logpdf(eta_old, self.alpha_prior_life, scale=self.beta_prior_life)
        )
        a_prop = np.sum(
            sc.gamma.logpdf(
                eta_old, self.alpha_prior_life, scale=(eta_prop / self.alpha_prior_life)
            )
        ) - np.sum(
            sc.gamma.logpdf(
                eta_prop, self.alpha_prior_life, scale=(eta_old / self.alpha_prior_life)
            )
        )

        acc_ratio = lik_ratio + a_prop + a_prior
        if acc_ratio > np.log(np.random.rand()):
            eta_old = eta_prop
            self.accept_eta += 1

        if it_ < (numerator + 1):
            self.eta[0, :] = eta_old.copy()
        else:
            self.eta[it_ - numerator, :] = eta_old.copy()

    def calculate_lifetime_likelihood_gpu_int(self, photon_int_, eta_):
        lf_cont = photon_int_[:, :, None, None] * (
            (eta_[:, None, None, None] / 2)
            * cp.exp(
                (eta_[:, None, None, None] / 2)
                * (
                    2
                    * (
                        self.tau_irf
                        - self.dt_padded[None, :, :, None]
                        - self.num * self.t_inter_p
                    )
                    + eta_[:, None, None, None] * self.sig_irf**2
                )
            )
            * special.erfc(
                (
                    self.tau_irf
                    - self.dt_padded[None, :, :, None]
                    - self.num * self.t_inter_p
                    + eta_[:, None, None, None] * self.sig_irf**2
                )
                / (self.sig_irf * cp.sqrt(2))
            )
        )
        lf_cont *= self.tiled_mask
        masked_arr = cp.sum(lf_cont, axis=(0, 3))
        masked = masked_arr.copy()
        log_masked_arr = cp.log(masked_arr)
        log_masked_arr[masked == 0] = 0
        return cp.asnumpy(cp.sum(log_masked_arr, axis=1))

    def calculate_lifetime_likelihood_gpu(self, photon_int_, eta_):
        lf_cont = photon_int_[:, :, None, None] * (
            (eta_[:, None, None, None] / 2)
            * cp.exp(
                (eta_[:, None, None, None] / 2)
                * (
                    2
                    * (
                        self.tau_irf
                        - self.dt_padded[None, :, :, None]
                        - self.num * self.t_inter_p
                    )
                    + eta_[:, None, None, None] * self.sig_irf**2
                )
            )
            * special.erfc(
                (
                    self.tau_irf
                    - self.dt_padded[None, :, :, None]
                    - self.num * self.t_inter_p
                    + eta_[:, None, None, None] * self.sig_irf**2
                )
                / (self.sig_irf * cp.sqrt(2))
            )
        )
        lf_cont *= self.tiled_mask
        masked_arr = cp.sum(lf_cont, axis=(0, 3))
        masked = masked_arr.copy()
        masked_arr = masked_arr[masked != 0]
        return float(cp.sum(cp.log(masked_arr)))

    def runAnalysis(self):
        if self.dt_ is None or self.lambda_ is None:
            qt.QMessageBox.critical(
                self, "Error", "Please upload data before running the analysis."
            )
        else:
            self.ui.run_.setEnabled(False)
            try:
                self.get_values()
                numeric = 4
                self.num = cp.arange(numeric)[None, None, None, :]
                # Find the maximum length
                max_len = max(len(np.squeeze(x)) for x in self.dt_)
                self.dt_padded = np.zeros((len(self.dt_), max_len))
                mask_ = np.zeros((len(self.dt_), max_len))
                for i, x_ in enumerate(self.dt_):
                    x = np.squeeze(x_)
                    self.dt_padded[i, : len(x)] = x
                    mask_[i, : len(x)] = 1
                del x, x_
                self.tiled_mask = cp.asarray(
                    np.tile(mask_[None:, :, None], (self.num_species, 1, 1, numeric))
                )
                self.dt_padded = cp.asarray(self.dt_padded)
                # Sampling the parameters
                self.accept_i = 0
                self.accept_pi = 0
                self.accept_eta = 0
                self.ui.progressBar.setVisible(True)
                self.ui.progressBar.setValue(0)
                self.init_chain()
                t0 = datetime.now()
                numerator = self.num_iter - self.n_iter
                for jj in range(1, self.num_iter):

                    self.sample_int(jj, numerator)
                    self.sample_photon_spectra(jj, numerator)
                    self.sample_lifetime(jj, numerator)
                    percentage = int((jj / self.num_iter) * 100)
                    self.progressBar.setValue(percentage)
                    qt.QApplication.processEvents()
                percentage = 100
                self.progressBar.setValue(percentage)
                qt.QApplication.processEvents()

                qt.QMessageBox.information(
                    self, "Analysis Complete", "The analysis has finished successfully!"
                )

                self.ui.plot_.setEnabled(True)
                self.ui.save_results.setEnabled(True)
            finally:
                self.ui.run_.setEnabled(True)

    def show_plot(self):
        selected_plot = self.ui.combo_plot.currentText()

        self.plot_window = PlotWindow(
            self.eta,
            self.photon_int,
            self.pi,
            selected_plot,
            self.nsb,
            self.ui.img_size,
            self,
        )
        self.plot_window.show()

    def export_results(self):
        options = qt.QFileDialog.Option.DontUseNativeDialog
        file_dialog = qt.QFileDialog(self)
        file_dialog.setOptions(options)
        filters = "MAT Files (*.mat);;Python Dictionary (*.pkl)"
        file_name, _ = file_dialog.getSaveFileName(
            self, "Save Results", "", filters, options=options
        )
        if file_name:
            if file_name.endswith(".mat"):
                self.save_as_mat(file_name)
            elif file_name.endswith(".pkl"):
                self.save_as_dict(file_name)

    def save_as_mat(self, file_name):
        data = {"eta": self.eta, "pi": self.pi, "photon_int": self.photon_int}
        savemat(file_name, data)
        qt.QMessageBox.information(
            self, "Success", "Results saved successfully as .mat file."
        )

    def save_as_dict(self, file_name):
        data = {"eta": self.eta, "pi": self.pi, "photon_int": self.photon_int}
        with open(file_name, "wb") as f:
            pickle.dump(data, f)
        qt.QMessageBox.information(
            self, "Success", "Results saved successfully as .pkl file."
        )


app = qt.QApplication(sys.argv)

mainWindow = MainWindow()

mainWindow.show()
sys.exit(app.exec())
