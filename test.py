import numpy as np
from scipy import io

data = io.loadmat(
    "/home/reza/software/Spectral-FLIM/1color_shiftedIRF_MitoOrange_EntireData.mat"
)
dt = np.squeeze(data["Dt"])
lambda_ = data["Lambda"]

print(lambda_[5000, :])
