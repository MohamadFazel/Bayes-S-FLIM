import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from modules.sflim import run_sflim_sampler
from modules.forward import gen_data
import scipy.io as sio
import hashlib
import sys

import time

import numpy as np
import scipy as sc
import matplotlib.pylab as plt

# ly = sc.io.loadmat("New Folder/1-color LysoRed.mat")
# mi = sc.io.loadmat("New Folder/1-color MitoOrange.mat")
# vi = sc.io.loadmat("New Folder/1-color Viafluor488.mat")
mix = sc.io.loadmat("New Folder/3color_shiftedIRF_Data#1.mat")

# dt_l = np.squeeze( ly['Dt']).reshape(128,128)
# dt_m = np.squeeze(mi["Dt"]).reshape(128,128)
# dt_v = np.squeeze(vi["Dt"]).reshape(128,128)
dt_mix = np.squeeze(mix["Dt"]).reshape(128,128)
# dt_l = dt_l[:64, :64]
# dt_m = dt_m[64:, :64]
# dt_v = dt_v[64:, :64]
dt_mix = dt_mix[64:, :64]
# dt_l = dt_l.reshape(-1)
# dt_m = dt_m.reshape(-1)
# dt_v = dt_v.reshape(-1)

dt = dt_mix.reshape(-1)

# dt = []
# l1 = []
# l2 = []
# l3 = []
l = []
for i in range(len(dt)):
    # d1 = dt_l[i].reshape(-1)
    # l1.append(len(d1)) 
    # d2 = dt_m[i].reshape(-1)
    # l2.append(len(d2))
    # d3 =dt_v[i].reshape(-1)
    # l3.append(len(d3))
    # d = np.concatenate([d1,d2, d3])
    l.append(len(dt[i]))
    # dt.append(d)

# l1 = np.array(l1).reshape(-1, 64)
# l2 = np.array(l2).reshape(-1, 64)
# l3 = np.array(l3).reshape(-1, 64)
l = np.array(l).reshape(-1, 64)
# print(np.max(l1))
# plt.imshow(l1, cmap='gray')
# plt.savefig("/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/LysotrackerRed22.png")
# plt.show()
# plt.imshow(l2, cmap='gray')
# plt.savefig("/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/MitotrackerOrange22.png")
# plt.show()
# plt.imshow(l3, cmap='gray')
# plt.savefig("/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/ViaFluor48822.png")
# plt.show()
plt.imshow(l, cmap='gray')
plt.savefig("/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/SpectralFlim/NewResult")
plt.show()

# print(np.max(l1))
# print(np.max(l2))
# print(np.max(l3))
# print(np.max(l))

# lam_l = ly['Lambda']
# lam_m = mi["Lambda"]
# lam_v = vi["Lambda"]
lam_mix = mix["Lambda"]
# print(lam_l.shape)

# lam_l = lam_l.reshape(-1, 128, lam_l.shape[1])
# lam_m = lam_m.reshape(-1, 128, lam_m.shape[1])
# lam_v = lam_v.reshape(-1, 128, lam_v.shape[1])

lam_mix = lam_mix.reshape(-1, 128, lam_mix.shape[1])
# lam_l = lam_l[64:, :64]
# lam_m = lam_m[64:, :64]
# lam_v = lam_v[64:, :64]

lam_mix = lam_mix[64:, :64]

# lambda_ = lam_l + lam_m + lam_v
lambda_ = lam_mix.reshape(-1, lam_mix.shape[2])

TInterP = 12.85
TauIRF = 6.54
SigIRF = 0.62
M = 3
NIter = 300000
t0 = datetime.now()
timestr = time.strftime("%m%d%H%M%S")

pi, photon_int, eta = run_sflim_sampler(dt, lambda_, TInterP, TauIRF, SigIRF, TInterP, NIter, M)

np.save(f"/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/SpectralFlim/NewResult/mix_Pi_{timestr}.npy", pi[-50000:])
np.save(f"/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/SpectralFlim/NewResult/mix_Phot_{timestr}.npy", photon_int[-50000:])
np.save(f"/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/SpectralFlim/NewResult/Mix_Eta_{timestr}.npy", eta[-50000:])

# np.save(f"Results/LysotrackerRed.npy", l1)
# np.save(f"Results/MitotrackerOrange.npy", l2)
# np.save(f"Results/ViaFluor488.npy", l3)
# np.save(f"Results/mix.npy", l)


print(datetime.now()-t0)
