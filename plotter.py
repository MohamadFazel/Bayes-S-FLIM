import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QWidget as qt
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

        # Set up the layout and canvas for plotting
        layout = qt.QVBoxLayout()
        self.setLayout(layout)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Define colors and colormaps
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

        # Plot the data based on the selection
        self.plot_results()

    def plot_results(self):
        self.figure.clear()

        if self.select_plot == "Spectra":
            self.plot_spectra()
        elif self.select_plot == "Lifetime Histogram":
            self.plot_lifetime_histogram()
        elif self.select_plot == "Maps":
            self.plot_maps()

        self.canvas.draw()

    def plot_spectra(self):
        ax = self.figure.add_subplot(1, 1, 1)
        pin = np.mean(self.pi[-20000:, :, :], axis=0)
        x = np.linspace(375, 760, self.nsb)
        for ii in range(pin.shape[0]):
            color = self.colors[ii % len(self.colors)]
            plt.plot(
                x, pin[ii] / np.sum(pin[ii]), color=color, label=f"Species #{ii+1}"
            )
        ax.set_title("Species Spectra")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Distribution")
        ax.legend()

    def plot_lifetime_histogram(self):
        ax = self.figure.add_subplot(1, 1, 1)
        for ii in range(self.eta.shape[0]):
            color = self.colors[ii % len(self.colors)]
            plt.hist(
                1 / self.eta[-20000:, ii],
                bins=100,
                color=color,
                label=f"Species #{ii+1}",
                density=True,
            )
        ax.set_title("Lifetimes Histogram")
        ax.set_xlabel("Lifetime (ns)")
        ax.set_ylabel("Distribution")
        ax.legend()

    def plot_maps(self):
        phi = np.mean(self.photon_int[-20000:, :, :], axis=0).reshape(-1, self.img_sz)
        num_images = phi.shape[0]
        cols, rows = 3, (num_images // cols) + (num_images % cols != 0)
        for ii in range(num_images):
            cmap = self.cmaps[ii % len(self.cmaps)]
            ax = self.figure.add_subplot(rows, cols, ii + 1)
            plt.imshow(phi[ii], cmap=cmap)
            ax.set_title(f"Species #{ii + 1}")
            ax.axis("off")
        self.figure.suptitle("Maps")
