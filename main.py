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
#nakon lore, nakon
#########################
########## Params #######
#########################
#File paths you want to analyze
file_path = "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/SpectralFlim/data_flim/3color_shiftedIRF_Data#1.mat"
#The path you want to save results
save_path = "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results"
#Inrepulse time
TInterP = 12.8
#IRF offset
TauIRF = 2.516
#IRF std
SigIRF = 0.51
#Number of species
num_species = 3
#Number of iterations
NIter = 350000
#Image size
img_size = (128,128)
#choose image piece you want to analyze
slice_params = slice(40, 104, None), slice(64, 128, None)
file_name = os.path.basename(file_path)
name_without_extension = "data#11"
#os.path.splitext(file_name)[0]

#################################################################

ly = sio.loadmat("/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/SpectralFlim/data_flim/1color_LysoRed_Lifetime_3point3ns_shiftedIRF_Offset_2point5_Sigma_point54_Bg_23percent.mat")
vi = sio.loadmat("/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/SpectralFlim/data_flim/1color_viaFluor_Lifetime_4point9ns_shiftedIRF_Offset_2point5_Sigma_point6_Bg_10percent.mat")
mi = sio.loadmat("./1color_shiftedIRF_MitoOrange_EntireData.mat")

dt_l = np.squeeze( ly['Dt']).reshape(-1,128)
dt_m = np.squeeze(mi["Dt"]).reshape(-1,256)
dt_v = np.squeeze(vi["Dt"]).reshape(-1,128)

dt_l = dt_l[10:42,5:37]
dt_m = dt_m[178:210, 40:72]
dt_v = dt_v[10:42, 42:74]

dt_l = dt_l.reshape(-1)
dt_m = dt_m.reshape(-1)
dt_v = dt_v.reshape(-1)

dt = []
l1 = []
l2 = []
l3 = []
l = []
for i in range(len(dt_l)):
    d1 = dt_l[i].reshape(-1)
    l1.append(len(d1))
    d2 = dt_m[i].reshape(-1)
    l2.append(len(d2))
    d3 =dt_v[i].reshape(-1)
    l3.append(len(d3))
    d = np.concatenate([d1,d2, d3])
    l.append(len(d))
    dt.append(d)
ggg = 32
l1 = np.array(l1).reshape(-1, ggg)
l2 = np.array(l2).reshape(-1, ggg)
l3 = np.array(l3).reshape(-1, ggg)
l = np.array(l).reshape(-1, ggg)
print(np.max(l1))
plt.imshow(l1, cmap='Blues')
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
plt.title("LysoRed")
# plt.savefig(f"LysoRed.png")
plt.show()
np.save(save_path+"/LysoRed.npy", l1)

plt.imshow(l2, cmap='Greens')
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
plt.title("MitoOrange")
# plt.savefig(f"MitoOrange.png")
plt.show()
np.save(save_path+"/MitoOrange.npy", l2)
plt.imshow(l3, cmap='Reds')
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
plt.title("ViaFlour")
# plt.savefig(f"ViaFlour.png")
plt.show()
np.save(save_path+"/ViaFlour.npy", l3)
plt.imshow(l, cmap='gray')
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
plt.title("Mix")
# plt.savefig(f"mix.png")
plt.show()
exit()
# lam_l = ly['Lambda']
# lam_m = mi["Lambda"]
# lam_v = vi["Lambda"]

# lam_l = lam_l.reshape(-1, 128, lam_l.shape[1])
# lam_m = lam_m.reshape(-1, 256, lam_m.shape[1])
# lam_v = lam_v.reshape(-1, 128, lam_v.shape[1])

# lam_l = lam_l[10:42,5:37]
# lam_m = lam_m[178:210, 40:72]
# lam_v = lam_v[10:42, 42:74]


# lam_mix = lam_l + lam_m + lam_v
# lambda_ = lam_mix.reshape(-1, lam_mix.shape[2])
#################################################################
mix = sc.io.loadmat(file_path)
dt_mix = np.squeeze(mix["Dt"]).reshape(*img_size)
dt_mix = dt_mix[slice_params]
dummy_size = dt_mix.shape[1]
print(dummy_size)

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
#################################################################

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