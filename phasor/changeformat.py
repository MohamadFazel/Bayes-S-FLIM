import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import itertools

data = sio.loadmat(f"SFLIM_Dataset.mat")

# Access the Results from data1 and TRES from data2
Results = np.squeeze(data["Results"])
TRES_data = Results["TRES"]
treslist = np.ndarray.tolist(TRES_data)
time = np.array(treslist)

x = np.linspace(0.001, 12.8, 255)
dt = []
spec = np.zeros(32)
for it in range(time.shape[1]):
    y = time[:, it]
    for jj in range(len(y)):
        t = np.zeros(y[jj]) + x[jj]
        lt = np.ndarray.tolist(t)
        dt.append(lt)
        spec[it] += y[jj]
        dt.append(t)
flattened_list = list(itertools.chain(*dt))
dt = np.array(flattened_list).astype(np.float16)

np.save("spec.npy", spec)
np.save("dt6.npy", dt)
