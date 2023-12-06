import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from datetime import datetime
from modules.sflim import run_sflim_sampler
from modules.data_loader import load_data_and_lambda
from modules.visualization import plot_image
import time
import tkinter as tk
from tkinter import filedialog

class SpectralFittingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Spectral Fitting App")

        # Data variables
        self.dt_mix = None
        self.lam_mix = None

        # Create widgets
        self.load_button = tk.Button(self.master, text="Load Data", command=self.load_data)
        self.load_button.pack()

        # Parameters
        self.interpulse_time_var = tk.StringVar()
        self.irf_tau_var = tk.StringVar()
        self.num_spec_var = tk.StringVar()
        self.irf_sigma_var = tk.StringVar()
        self.num_iter_var = tk.StringVar()

        tk.Label(self.master, text="Interpulse Time:").pack()
        tk.Entry(self.master, textvariable=self.interpulse_time_var).pack()

        tk.Label(self.master, text="IRF Tau:").pack()
        tk.Entry(self.master, textvariable=self.irf_tau_var).pack()

        tk.Label(self.master, text="Number of Species:").pack()
        tk.Entry(self.master, textvariable=self.num_spec_var).pack()

        tk.Label(self.master, text="IRF Sigma:").pack()
        tk.Entry(self.master, textvariable=self.irf_sigma_var).pack()

        tk.Label(self.master, text="Number of Iterations:").pack()
        tk.Entry(self.master, textvariable=self.num_iter_var).pack()

        # Save results
        self.save_button = tk.Button(self.master, text="Select Save Folder", command=self.select_save_folder)
        self.save_button.pack()

        self.run_button = tk.Button(self.master, text="Run Spectral Fitting", command=self.run_spectral_fitting)
        self.run_button.pack()

        self.quit_button = tk.Button(self.master, text="Quit", command=self.master.quit)
        self.quit_button.pack()

    def load_data(self):
        file_path = filedialog.askopenfilename(title="Select Data File", filetypes=[("MAT files", "*.mat")])
        if file_path:
            self.dt_mix, self.lam_mix = load_data_and_lambda(file_path)
            plot_image(np.array([len(dt) for dt in self.dt_mix]).reshape(-1, 64), "visualization.png")

    def select_save_folder(self):
        self.save_folder = filedialog.askdirectory(title="Select Save Folder")

    def run_spectral_fitting(self):
        if self.validate_inputs():
            interpulse_time = float(self.interpulse_time_var.get())
            irf_tau = float(self.irf_tau_var.get())
            num_spec = int(self.num_spec_var.get())
            irf_sigma = float(self.irf_sigma_var.get())
            num_iter = int(self.num_iter_var.get())

            t0 = datetime.now()
            timestr = time.strftime("%m%d%H%M%S")

            pi, photon_int, eta = run_sflim_sampler(self.dt_mix, self.lam_mix, interpulse_time, irf_tau, irf_sigma, interpulse_time, num_iter, num_spec)

            save_path = f"{self.save_folder}/mix_results_{timestr}"

            np.save(f"{save_path}_Pi.npy", pi[-50000:])
            np.save(f"{save_path}_Phot.npy", photon_int[-50000:])
            np.save(f"{save_path}_Eta.npy", eta[-50000:])

            print(datetime.now() - t0)
        else:
            print("Please fill in all the required fields.")

    def validate_inputs(self):
        # Check if all required fields are filled
        return all([self.interpulse_time_var.get(), self.irf_tau_var.get(), self.num_spec_var.get(),
                    self.irf_sigma_var.get(), self.num_iter_var.get(), hasattr(self, 'save_folder')])

if __name__ == "__main__":
    root = tk.Tk()
    app = SpectralFittingApp(root)
    root.mainloop()

