import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from src.sflim import run_sflim_sampler
from src.forward import gen_data
import scipy.io as sio
import hashlib
import sys

import time
import os
import numpy as np
import scipy as sc
import matplotlib.pylab as plt

#########################
########## Params #######
#########################
#File paths you want to analyze
file_path = "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/SpectralFlim/data_flim/1color_viaFluor_Lifetime_4point9ns_shiftedIRF_Offset_2point5_Sigma_point6_Bg_10percent.mat"
#The path you want to save results
save_path = "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/via"
#Inrepulse time
TInterP = 12.85
#IRF offset
TauIRF = 2.506
#IRF std
SigIRF = 0.51
#Number of species
num_species = 1 
#Number of iterations
NIter = 240000
#Image size
img_size = (128,128)
#choose image piece you want to analyze
slice_params = slice(33, 55, None), slice(23, 44, None)
file_name = os.path.basename(file_path)
name_without_extension = os.path.splitext(file_name)[0]

#################################################################
mix = sc.io.loadmat(file_path)
dt_mix = np.squeeze(mix["Dt"]).reshape(*img_size)
dt_mix = dt_mix[slice_params]
dummy_size = dt_mix.shape[1]

dt = dt_mix.reshape(-1)
l = []
for i in range(len(dt)):
    l.append(len(dt[i]))

l = np.array(l).reshape(-1, dummy_size)
plt.figure(figsize=(16,16))
plt.imshow(l, cmap='gray')
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
plt.savefig(f"{save_path}/{name_without_extension}.png")
plt.show()

lam_mix = mix["Lambda"]
lam_mix = lam_mix.reshape(-1, img_size[1], lam_mix.shape[1])
lam_mix = lam_mix[slice_params]

lambda_ = lam_mix.reshape(-1, lam_mix.shape[2])

t0 = datetime.now()
timestr = time.strftime("%m%d%H%M%S")

pi, photon_int, eta, bg = run_sflim_sampler(dt, lambda_, TauIRF, SigIRF, TInterP, NIter, num_species)

np.save(f"{save_path}/Pi_{name_without_extension}_{timestr}.npy", pi)
np.save(f"{save_path}/int_{name_without_extension}_{timestr}.npy", photon_int)
np.save(f"{save_path}/Eta_{name_without_extension}_{timestr}.npy", eta)
print(datetime.now()-t0)
################################################################
num_avg = NIter//8
phi = np.mean(photon_int[-num_avg:, :,:], axis=0)
phi = phi.reshape(phi.shape[0], -1, dummy_size)
for it in range(num_species):
    plt.imshow(phi[it])
    plt.savefig(f"{save_path}/intensity_{name_without_extension}_{it}.png")

for it in range(num_species):
    plt.hist(1/eta[-num_avg:,0], bins=100, color="red", label="Viafluor")
    plt.savefig(f"{save_path}/lifetime_{name_without_extension}_{it}.png")

pin = np.mean(pi[-num_avg:,:,:], axis=0)
for it in range(num_species):
    plt.plot(pin[1]/np.sum(pin[1]),'r', label="2_Learned")
    plt.savefig(f"{save_path}/spectrum_{name_without_extension}_{it}.png")
plt.plot(bg[-num_avg:])
plt.savefig(f"{save_path}/bg_{name_without_extension}_{it}.png")

################################################################