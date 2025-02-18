import cupy as cp
import numpy as np
from cupyx.scipy import special
from scipy import special


def calculate_lifetime_likelihood_gpu(photon_int, eta, pi_bg, tau_irf, sig_irf, dt_padded, mask, t_inter_p, num):
    # Ensure float32 precision
    photon_int = photon_int.astype(cp.float32)
    eta = eta.astype(cp.float32)
    pi_bg = pi_bg.astype(cp.float32)
    tau_irf = tau_irf.astype(cp.float32)
    sig_irf = sig_irf.astype(cp.float32)
    dt_padded = dt_padded.astype(cp.float32)
    mask = mask.astype(cp.float32)
    t_inter_p = t_inter_p.astype(cp.float32)

    # Reshape to prevent redundant broadcasting
    eta = eta[:, None, None, None]
    dt_padded = dt_padded[None, :, :, None]

    # Precompute constants
    half_eta = eta / 2
    sig_irf_sq = sig_irf**2
    sqrt_2 = cp.sqrt(2)
    inv_log2 = 1 / cp.log(2)  # Used for exp2 optimization

    # Compute exponent
    exp_component = cp.multiply(half_eta, 2 * (tau_irf - dt_padded - num * t_inter_p) + cp.multiply(eta, sig_irf_sq))

    # Use exp2 instead of exp
    exp_part = cp.exp2(exp_component * inv_log2)

    # Compute erfc argument
    erf_arg = (tau_irf - dt_padded - num * t_inter_p + cp.multiply(eta, sig_irf_sq)) / (sig_irf * sqrt_2)

    # Use erfcx for better numerical stability
    erfc_part = special.erfc(erf_arg)

    # Compute likelihood
    lf_cont = cp.multiply(photon_int[:, :, None, None], cp.multiply(half_eta, exp_part) * erfc_part)

    # Apply mask
    lf_cont = cp.multiply(lf_cont, mask)

    # Reduce along necessary axes
    masked_arr = cp.sum(lf_cont, axis=(0, 3))

    # Compute log safely (avoid log(0) by replacing zeros with 1.0)
    log_sum = cp.sum(cp.log(cp.where(masked_arr > 0, masked_arr, 1.0)))

    return float(log_sum)


def calculate_lifetime_likelihood_gpu_int(photon_int, eta, pi_bg, tau_irf, sig_irf, dt_padded, mask, t_inter_p, num):
    # Ensure float32 precision
    photon_int = photon_int.astype(cp.float32)
    eta = eta.astype(cp.float32)
    pi_bg = pi_bg.astype(cp.float32)
    tau_irf = tau_irf.astype(cp.float32)
    sig_irf = sig_irf.astype(cp.float32)
    dt_padded = dt_padded.astype(cp.float32)
    mask = mask.astype(cp.float32)
    t_inter_p = t_inter_p.astype(cp.float32)

    # Compute the core likelihood function
    exponent = cp.multiply(
        eta[:, None, None, None] / 2,
        2 * (tau_irf - dt_padded[None, :, :, None] - num * t_inter_p)
        + cp.multiply(eta[:, None, None, None], sig_irf**2),
    )

    erf_arg = (
        tau_irf - dt_padded[None, :, :, None] - num * t_inter_p + cp.multiply(eta[:, None, None, None], sig_irf**2)
    ) / (sig_irf * cp.sqrt(2))

    lf_cont = cp.multiply(
        photon_int[:, :, None, None],
        cp.multiply(eta[:, None, None, None] / 2, cp.exp(exponent)) * special.erfc(erf_arg),
    )

    # Apply mask
    lf_cont = cp.multiply(lf_cont, mask)

    # Sum over necessary axes
    masked_arr = cp.sum(lf_cont, axis=(0, 3))

    # Compute log and handle zeros safely
    log_masked_arr = cp.where(masked_arr > 0, cp.log(masked_arr), 0)

    # Return final sum
    return cp.asnumpy(cp.sum(log_masked_arr, axis=1))
