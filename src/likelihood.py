import cupy as cp
from cupyx.scipy import special


def calculate_lifetime_likelihood_gpu(photon_int, eta, pi_bg, tau_irf, sig_irf, dt_padded, mask, t_inter_p, num):
    # Convert everything to float32 for GPU efficiency
    photon_int = photon_int.astype(cp.float32, copy=False)
    eta = eta.astype(cp.float32, copy=False)
    pi_bg = pi_bg.astype(cp.float32, copy=False)
    tau_irf = tau_irf.astype(cp.float32, copy=False)
    sig_irf = sig_irf.astype(cp.float32, copy=False)
    dt_padded = dt_padded.astype(cp.float32, copy=False)
    mask = mask.astype(cp.float32, copy=False)
    t_inter_p = cp.float32(t_inter_p)
    num = cp.float32(num)

    # Precompute common terms to reduce redundant calculations
    eta_half = cp.multiply(eta[:, None, None, None], 0.5)
    sig_irf_sq = cp.multiply(sig_irf, sig_irf)
    eta_sig_sq = cp.multiply(eta_half, sig_irf_sq)
    dt_offset = tau_irf - dt_padded[None, :, :, None] - num * t_inter_p

    # Compute exponent term
    exp_term = cp.exp(cp.multiply(eta_half, 2 * dt_offset + eta_sig_sq))

    # Compute erfc term
    erfc_term = special.erfc((dt_offset + eta_sig_sq) / (sig_irf * cp.sqrt(cp.float32(2))))

    # Compute the final likelihood contribution
    lf_cont = cp.multiply(photon_int[:, :, None, None], cp.multiply(eta_half, cp.multiply(exp_term, erfc_term)))

    # Apply mask efficiently
    lf_cont = cp.multiply(lf_cont, mask)

    # Sum over the required axes
    masked_arr = cp.sum(lf_cont, axis=(0, 3))

    # Avoid log(0) issues by ensuring all values are positive
    masked_arr = cp.maximum(masked_arr, 1e-10)

    # Compute the final log-sum
    return float(cp.sum(cp.log(masked_arr)))


def calculate_lifetime_likelihood_gpu_int(photon_int, eta, pi_bg, tau_irf, sig_irf, dt_padded, mask, t_inter_p, num):
    # Convert everything to float32 for better GPU performance
    photon_int = photon_int.astype(cp.float32, copy=False)
    eta = eta.astype(cp.float32, copy=False)
    pi_bg = pi_bg.astype(cp.float32, copy=False)
    tau_irf = tau_irf.astype(cp.float32, copy=False)
    sig_irf = sig_irf.astype(cp.float32, copy=False)
    dt_padded = dt_padded.astype(cp.float32, copy=False)
    mask = mask.astype(cp.float32, copy=False)
    t_inter_p = cp.float32(t_inter_p)
    num = cp.float32(num)

    # Precompute shared terms to avoid redundant calculations
    eta_half = cp.multiply(eta[:, None, None, None], 0.5)
    sig_irf_sq = cp.multiply(sig_irf, sig_irf)
    eta_sig_sq = cp.multiply(eta_half, sig_irf_sq)
    dt_offset = tau_irf - dt_padded[None, :, :, None] - num * t_inter_p

    # Compute exponent term
    exp_term = cp.exp(cp.multiply(eta_half, 2 * dt_offset + eta_sig_sq))

    # Compute erfc term using CuPy's GPU-accelerated function
    erfc_term = special.erfc((dt_offset + eta_sig_sq) / (sig_irf * cp.sqrt(cp.float32(2))))

    # Compute the final likelihood contribution
    lf_cont = cp.multiply(photon_int[:, :, None, None], cp.multiply(eta_half, cp.multiply(exp_term, erfc_term)))

    # Apply mask efficiently
    lf_cont = cp.multiply(lf_cont, mask)

    # Sum over the required axes
    masked_arr = cp.sum(lf_cont, axis=(0, 3))

    # Avoid log(0) issues by ensuring all values are positive
    masked_arr = cp.maximum(masked_arr, 1e-10)

    # Compute the final log-sum
    log_masked_arr = cp.log(masked_arr)

    # Convert result to NumPy for CPU processing if needed
    return cp.asnumpy(cp.sum(log_masked_arr, axis=1))
