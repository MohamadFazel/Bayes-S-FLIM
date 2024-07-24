import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio
import itertools

data = sio.loadmat(f"phasor/SFLIM_Dataset.mat")

# Access the Results from data1 and TRES from data2
Results = np.squeeze(data["Results"])
TRES_data = Results["TRES"]
treslist = np.ndarray.tolist(TRES_data)
tres = np.array(treslist)
# plt.plot(tres[:, 5])
# plt.plot(tres[:, 6])
# plt.plot(tres[:, 7])
# plt.plot(tres[:, 8])

# plt.show()
tres100 = tres / 100
# print(f"tres : {tres.shape}")
# plt.plot(tres100[:, 5])
# plt.plot(tres100[:, 6])
# plt.plot(tres100[:, 7])
# plt.plot(tres100[:, 8])
# plt.show()
# exit()
x = np.linspace(0.001, 12.8, 255)
dt = []
spec = np.zeros(32)
for it in range(tres100.shape[1]):
    y = tres100[:, it]
    for jj in range(len(y)):
        t = np.zeros(int(y[jj])) + x[jj]
        lt = np.ndarray.tolist(t)
        dt.append(lt)
        spec[it] += y[jj]
        dt.append(t)
flattened_list = list(itertools.chain(*dt))
dt = np.array(flattened_list).astype(np.float16)

np.save("phasor/data/spec.npy", spec)
np.save("phasor/data/dt6.npy", dt)

sio.savemat("phasor/data/dt6_spec.mat", {"dt6": dt, "spec": spec})
