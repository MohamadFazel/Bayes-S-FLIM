import numpy as np
import scipy.stats as sc
from src.likelihood import *

def sample_int(lambd, pi, i_old, pi_bg, n_pix, eta, tau_irf,  sig_irf, dt_padded, tiled_mask, t_inter_p, num, accept_i):
    '''
    sampling the intensities per spectral band per species
    lambd: number of photons per spectral band
    i_old(numpy.ndarray): Array of shape (NSpecies, NPix) representing photon intensities per pixel per species.
    pi: photon ratios per pixel per species
    n_pix: total number of pixels
    '''
    alpha_prop = 1000
    alpha_i = 1
    beta_i = 2100
    i_new = i_old.copy()
    i_new[:,:] = np.random.gamma(alpha_prop, i_old[:,:]/alpha_prop)
    lf_top = calculate_lifetime_likelihood_gpu_int(cp.asarray(i_new), cp.asarray(eta), cp.asarray(pi_bg), tau_irf, sig_irf, dt_padded, tiled_mask, t_inter_p, num)
    lf_bot = calculate_lifetime_likelihood_gpu_int(cp.asarray(i_old), cp.asarray(eta), cp.asarray(pi_bg), tau_irf, sig_irf, dt_padded, tiled_mask, t_inter_p, num)

    tmp_top = np.sum((i_new[:, :, None] * pi[:, None, :]), axis=0)
    a_top = np.sum(sc.poisson.logpmf(lambd, tmp_top), axis=1)
    a_top[np.isnan(a_top)]=0
    a_top[np.abs(a_top)==np.inf]=0

    tmp_bot = np.sum((i_old[:, :, None] * pi[:, None, :]), axis=0)
    a_bottom =  np.sum(sc.poisson.logpmf(lambd, tmp_bot), axis=1)
    a_bottom[np.isnan(a_bottom)]=0
    a_bottom[np.abs(a_bottom)==np.inf]=0

    a_prior = np.sum(sc.gamma.logpdf(i_new[:,:], alpha_i, scale=beta_i), axis=0) - np.sum(sc.gamma.logpdf(i_old[:,:], alpha_i, scale=beta_i), axis=0)
    a_prop = np.sum(sc.gamma.logpdf(i_old[:,:], alpha_prop, scale= (i_new[:,:]/alpha_prop)), axis=0) -np.sum(sc.gamma.logpdf(i_new[:,:], alpha_prop, scale= (i_old[:,:]/alpha_prop)), axis=0)

    a = (a_top - a_bottom) + (lf_top - lf_bot) + a_prop + a_prior
    cond = a > np.log(np.random.rand(n_pix))[None,:]
    i_old = np.where(cond, i_new, i_old)
    accept_i += np.sum(cond.astype(int))*(1/n_pix)

    return i_old, accept_i