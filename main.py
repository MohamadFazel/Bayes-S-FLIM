import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from modules.sflim import run_sflim_sampler
from modules.forward import gen_data
import scipy.io as sio

data = sio.loadmat("Data_08_2023/3-color mixture.mat")
lambda_ = np.array(data["Lambda"])
dt = data["Dt"][0]
dtt = []
for i in range(len(dt)):
    dtt.append(np.squeeze(dt[i]))

n_spec = 4
max_ph = 0.006
NPix = 2000
NPulse = 10**5
TInterP = 12.8
Lifetimes = np.array([0.5, 2.1, 3.5, 6])
SpecInd = np.arange(1, n_spec+1)
ExcProbs = max_ph * np.random.rand(n_spec, NPix)
TauIRF = 8
SigIRF = 0.5
bg = 10
Dt,Lambda,s, mu, sigma = gen_data(NPix, NPulse, TInterP, Lifetimes, SpecInd, ExcProbs, TauIRF, SigIRF, bg)
# M = np.size(Lifetimes)
# NIter = 1000
# t0 = datetime.now()
# pi, photon_int, eta = run_sflim_sampler(Dt, Lambda, TInterP, TauIRF, SigIRF, TInterP, NIter, M)

print(dtt[1].shape)
# print(datetime.now()-t0)