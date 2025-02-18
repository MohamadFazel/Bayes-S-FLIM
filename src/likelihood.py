import cupy as cp
import numpy as np
from cupyx.scipy import special


def calculate_lifetime_likelihood_gpu(photon_int, eta, pi_bg, tau_irf, sig_irf, dt_padded, mask, t_inter_p, num):
    lf_cont = photon_int[:, :, None, None] * (
        (eta[:, None, None, None] / 2)
        * cp.exp(
            (eta[:, None, None, None] / 2)
            * (2 * (tau_irf - dt_padded[None, :, :, None] - num * t_inter_p) + eta[:, None, None, None] * sig_irf**2)
        )
        * special.erfc(
            (tau_irf - dt_padded[None, :, :, None] - num * t_inter_p + eta[:, None, None, None] * sig_irf**2)
            / (sig_irf * cp.sqrt(2))
        )
    )
    lf_cont *= mask
    masked_arr = cp.sum(lf_cont, axis=(0, 3))
    masked = masked_arr.copy()
    masked_arr = masked_arr[masked != 0]
    return float(cp.sum(cp.log(masked_arr)))


def calculate_lifetime_likelihood_gpu_int(photon_int, eta, pi_bg, tau_irf, sig_irf, dt_padded, mask, t_inter_p, num):
    lf_cont = photon_int[:, :, None, None] * (
        (eta[:, None, None, None] / 2)
        * cp.exp(
            (eta[:, None, None, None] / 2)
            * (2 * (tau_irf - dt_padded[None, :, :, None] - num * t_inter_p) + eta[:, None, None, None] * sig_irf**2)
        )
        * special.erfc(
            (tau_irf - dt_padded[None, :, :, None] - num * t_inter_p + eta[:, None, None, None] * sig_irf**2)
            / (sig_irf * cp.sqrt(2))
        )
    )
    lf_cont *= mask
    masked_arr = cp.sum(lf_cont, axis=(0, 3))
    masked = masked_arr.copy()
    log_masked_arr = cp.log(masked_arr)
    log_masked_arr[masked == 0] = 0
    return cp.asnumpy(cp.sum(log_masked_arr, axis=1))
