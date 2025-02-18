import hashlib
import os
import sys
import time
from datetime import datetime

import matplotlib.pylab as plt
import numpy as np
import scipy as sc
import scipy.io as sio

from src.forward import gen_data
from src.sflim import run_sflim_sampler

# ______________________________________________________________________________________________________________________
#########################
########## Params #######
#########################
# File paths you want to analyze
file_path = "sample_data/sample_1739624166.6487927.mat"
file_path = os.path.abspath(file_path)

# The path you want to save results
save_path = "results"
os.makedirs(save_path, exist_ok=True)
save_path = os.path.abspath(save_path)

# Inrepulse time
t_inter_p = 12.85

# IRF offset
tau_irf = 2.506

# IRF std
sig_irf = 0.51

# Number of species
num_species = 3

# Number of iterations
n_iter = 100000

# Image size
img_size = (32, 32)

# Choose image piece you want to analyze
# slice_params = slice(0, 32, None), slice(0, 32, None)

file_name = os.path.basename(file_path)
name_without_extension = os.path.splitext(file_name)

show_images_first = False
# ______________________________________________________________________________________________________________________
data = sio.loadmat(file_path)
dt_ = np.squeeze(data["Dt"]).reshape(*img_size)
# dt_ = dt_[slice_params]
dummy_size = dt_.shape[1]
dt = dt_.reshape(-1)
del dt_

image = np.array([len(sublist) for sublist in dt]).reshape(-1, dummy_size)

if show_images_first:
    plt.imshow(image, cmap="gray")
    plt.savefig(f"{save_path}/{name_without_extension}.png")
    plt.show()

lambda_ = data["Lambda"]
lambda_ = lambda_.reshape(-1, img_size[1], lambda_.shape[1])
# lambda_ = lambda_[slice_params]
lambda_ = lambda_.reshape(-1, lambda_.shape[2])

t0 = datetime.now()
timestr = time.strftime("%m%d%H%M%S")
pi, photon_int, eta, bg = run_sflim_sampler(dt, lambda_, tau_irf, sig_irf, t_inter_p, n_iter, num_species)

np.save(f"{save_path}/Pi_{name_without_extension}_{timestr}.npy", pi)
np.save(f"{save_path}/int_{name_without_extension}_{timestr}.npy", photon_int)
np.save(f"{save_path}/Eta_{name_without_extension}_{timestr}.npy", eta)
print(datetime.now() - t0)
