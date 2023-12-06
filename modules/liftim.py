import numpy as np
import scipy.stats as sc
import cupy as cp
from modules.likelihood import calculate_lifetime_likelihood_gpu

def sample_lifetime(photon_int, eta_old, pi_bg, tau_irf,  sig_irf, dt_padded, tiled_mask, t_inter_p, num, accept_eta):
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
    alpha_prop = 18000
    alpha_eta = 1
    beta_eta = 20
    eta_prop = np.random.gamma(alpha_prop, eta_old/alpha_prop)


    lf_top = calculate_lifetime_likelihood_gpu(cp.asarray(photon_int), cp.asarray(eta_prop), pi_bg, tau_irf,  sig_irf, dt_padded, tiled_mask, t_inter_p, num)
    lf_bot = calculate_lifetime_likelihood_gpu(cp.asarray(photon_int), cp.asarray(eta_old), pi_bg, tau_irf,  sig_irf, dt_padded, tiled_mask, t_inter_p, num)

    lik_ratio = (lf_top - lf_bot)
    a_prior = np.sum(sc.gamma.logpdf(eta_prop, alpha_eta, scale=beta_eta)) - np.sum(sc.gamma.logpdf(eta_old, alpha_eta, scale=beta_eta))
    a_prop = np.sum(sc.gamma.logpdf(eta_old, alpha_prop, scale= (eta_prop/alpha_prop))) -np.sum(sc.gamma.logpdf(eta_prop, alpha_prop, scale= (eta_old/alpha_prop)))

    acc_ratio = lik_ratio + a_prop + a_prior
    if acc_ratio > np.log(np.random.rand()):
        eta_old = eta_prop
        accept_eta += 1


    return eta_old, accept_eta
