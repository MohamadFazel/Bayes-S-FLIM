import scipy.io as sio
import numpy as np

def load_data_and_lambda(file_path):
    data = sio.loadmat(file_path)
    dt_mix = np.squeeze(data["Dt"]).reshape(128, 128)[64:, :64].reshape(-1)
    lam_mix = data["Lambda"].reshape(-1, 128, data["Lambda"].shape[1])[64:, :64]
    lam_mix = lam_mix.reshape(-1, lam_mix.shape[2])
    return dt_mix, lam_mix
