import cupy as cp
from cupyx.scipy import special
from modules.likelihood import calculate_lifetime_likelihood_gpu
import numpy as np
import scipy.stats as sc


def sample_bg(pi_bg_old, photon_int, eta, tau_irf,  sig_irf, dt_padded, tiled_mask, t_inter_p, num, accept_bg):
    '''
    sampling the lifetime of species
    Args:
        lambd: number of photons per spectral band
        photon_int(numpy.ndarray): Array of shape (NSpecies, NPix) representing photon intensities per pixel per species.
        n_pix: total number of pixels
        eta_old: current lifetime.
        tau_irf (float): Mean of the impulse response function (IRF) in nanoseconds.
        sig_irf (float): Standard deviation of the IRF in nanoseconds.
        dt: time aravals.
        t_interp (float): Interpulse time in nanoseconds.
        num:
        accept_eta: sampling acceptance rate
    Returns:
        tuple: Tuple containing the updated eta array and the updated accept_eta value.
    '''
    alpha_prop = 10000
    pi_bg_prop = np.random.gamma(alpha_prop, pi_bg_old/alpha_prop)

    lf_top = calculate_lifetime_likelihood_gpu(cp.asarray(photon_int), cp.asarray(eta), pi_bg_prop, tau_irf,  sig_irf, dt_padded, tiled_mask, t_inter_p, num)
    lf_bot = calculate_lifetime_likelihood_gpu(cp.asarray(photon_int), cp.asarray(eta), pi_bg_old, tau_irf,  sig_irf, dt_padded, tiled_mask, t_inter_p, num)

    lik_ratio = (lf_top - lf_bot)

    a_prop = np.sum(sc.gamma.logpdf(pi_bg_old, alpha_prop, scale= (pi_bg_prop/alpha_prop))) -np.sum(sc.gamma.logpdf(pi_bg_prop, alpha_prop, scale= (pi_bg_old/alpha_prop)))


    acc_ratio = lik_ratio + a_prop 
    if acc_ratio > np.log(np.random.rand()):
        pi_bg_old = pi_bg_prop
        accept_bg += 1


    return pi_bg_old, accept_bg
