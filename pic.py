import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from modules.sflim import run_sflim_sampler
from modules.forward import gen_data
import scipy.io as sio
import hashlib
import sys
data = sio.loadmat("Data_08_2023/3-color mixture.mat")

LysotrackerRed = sio.loadmat('Data_08_2023/1-color - LysotrackerRed.mat')
MitotrackerOrange = sio.loadmat('Data_08_2023/1-color - MitotrackerOrange.mat')
ViaFluor488 = sio.loadmat('Data_08_2023/1-color - ViaFluor488.mat')

dt_ly = LysotrackerRed["Dt"].reshape((128, 128))
dt_mi = MitotrackerOrange["Dt"].reshape((128, 128))
dt_vi = ViaFluor488["Dt"].reshape((128, 128))

# dt_l=dt_ly[:32, :32].reshape(-1)
# dt_m=dt_mi[40:72, :32].reshape(-1)
# dt_v=dt_vi[-32:, :32].reshape(-1)


dt_l=dt_ly[:64, :64].reshape(-1)
dt_m=dt_mi[24:88, :64].reshape(-1)
dt_v=dt_vi[-64:, :64].reshape(-1)
# dtt = []
ln = []
for i in range(dt_v.shape[0]):
    ln.append(len(dt_v[i]))

ln = np.array(ln).reshape(64,-1)
plt.imshow(ln)
plt.savefig("via6.png")
plt.show()

# dtt = []
ln = []
for i in range(dt_m.shape[0]):
    ln.append(len(dt_m[i]))

ln = np.array(ln).reshape(64,-1)
plt.imshow(ln)
plt.savefig("mit6.png")
plt.show()

# dtt = []
ln = []
for i in range(dt_l.shape[0]):
    ln.append(len(dt_l[i]))

ln = np.array(ln).reshape(64,-1)
plt.imshow(ln)
plt.savefig("lys6.png")
plt.show()