import scipy.io as sio
import numpy as np

import scipy.io as sio

data_ = sio.loadmat("phasor/data/data_.mat")
dt_ = data_["Dt"][0]
lambda_ = data_["Lambda"]

j = 0
for i in range(len(dt_)):
    j += np.size(dt_[i])
    print(np.size(dt_[i]))

print(j)
