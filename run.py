import json
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

# Load parameters from JSON file
with open("params.json", "r") as f:
    params = json.load(f)

# Extract parameters
file_path = params["file_path"]
save_path = params["save_path"]
t_inter_p = params["t_inter_p"]
tau_irf = params["tau_irf"]
sig_irf = params["sig_irf"]
num_species = params["num_species"]
n_iter = params["n_iter"]
img_size = tuple(params["img_size"])
slice_params = tuple(slice(*s) if s[2] is not None else slice(*s[:2]) for s in params["slice_params"])

name_without_extension = params["name_without_extension"]

file_name = os.path.basename(file_path)
mix = sc.io.loadmat(file_path)

dt_mix = np.squeeze(mix["Dt"]).reshape(*img_size)
dt_mix = dt_mix[slice_params]
dummy_size = dt_mix.shape[1]
print(dummy_size)

dt = dt_mix.reshape(-1)
lengths = [np.size(np.squeeze(dt[i])) for i in range(len(dt))]
lengths = np.array(lengths).reshape(-1, dummy_size)

plt.figure(figsize=(8, 8))
plt.imshow(lengths, cmap="Grays")
plt.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)
plt.savefig(f"{save_path}/{name_without_extension}.png")
plt.show()

lam_mix = mix["Lambda"]
lam_mix = lam_mix.reshape(-1, img_size[1], lam_mix.shape[1])
lam_mix = lam_mix[slice_params]

t0 = datetime.now()
time_str = time.strftime("%m%d%H%M%S")
pi, photon_int, eta, bg = run_sflim_sampler(dt, lam_mix, tau_irf, sig_irf, t_inter_p, n_iter, num_species)

np.save(f"{save_path}/Pi_{name_without_extension}_{time_str}.npy", pi)
np.save(f"{save_path}/int_{name_without_extension}_{time_str}.npy", photon_int)
np.save(f"{save_path}/Eta_{name_without_extension}_{time_str}.npy", eta)

print(datetime.now() - t0)
print("Results are saved now!")

################################################################
num_avg = n_iter // 8
phi = np.mean(photon_int[-num_avg:, :, :], axis=0)
phi = phi.reshape(phi.shape[0], -1, dummy_size)

for species in range(num_species):
    plt.imshow(phi[species])
    plt.savefig(f"{save_path}/intensity_{name_without_extension}_{species}.png")
plt.close()

for species in range(num_species):
    plt.hist(1 / eta[-num_avg:, 0], bins=100, color="red", label="Viafluor")
    plt.savefig(f"{save_path}/lifetime_{name_without_extension}_{species}.png")
plt.close()

pi_mean = np.mean(pi[-num_avg:, :, :], axis=0)

for species in range(num_species):
    plt.plot(pi_mean[1] / np.sum(pi_mean[1]), "r", label="2_Learned")
    plt.savefig(f"{save_path}/spectrum_{name_without_extension}_{species}.png")
plt.close()
