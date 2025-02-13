from datetime import datetime

import cupy as cp
import numpy as np

from .background import *
from .intensity import *
from .liftim import *
from .ratio import *


def run_sflim_sampler(dt, lambda_, tau_irf, sig_irf, t_inter_p, n_iter, m):
    """
    Runs the Gibbs sampling procedure for spectral fluorescence lifetime imaging microscopy (SFLIM).

    Args:
        dt (array): Set of arrival times.
        lambda_ (array): Number of photons within a set of 32 spectral bands.
        t_interp (float): Interpulse time in nanoseconds.
        tau_irf (float): Mean of the impulse response function (IRF) in nanoseconds.
        sig_irf (float): Standard deviation of the IRF in nanoseconds.
        n_iter (int): Number of iterations (samples).
        m (int): Number of species.

    Returns:
        pi (array): Sampled values for the photon probability parameters.
        photon_int (array): Sampled values for the photon intensity parameters.
    """
    nsb = 32
    npix = np.shape(lambda_)[0]
    numeric = 4
    num_itr = 50000  # Number of iterations keeping the chain
    num = cp.arange(numeric)[None, None, None, :]

    # Find the maximum length
    max_len = max(len(x) for x in dt)
    dt_padded = np.zeros((len(dt), max_len))
    mask = np.zeros((len(dt), max_len))

    for i, x in enumerate(dt):
        dt_padded[i, : len(x)] = np.squeeze(x)
        mask[i, : len(x)] = 1

    tiled_mask = cp.asarray(np.tile(mask[None:, :, None], (m, 1, 1, numeric)))
    dt_padded = cp.asarray(dt_padded)
    # Allocating the chains
    eta = np.zeros((num_itr, m))
    pi = np.random.rand(num_itr, m, nsb)
    photon_int = np.zeros((num_itr, m, npix))
    print(max_len)
    # Initializing the chains
    eta[0, :] = np.random.rand(m)
    print(1 / eta[0, :])
    for mm in range(m):
        pi[0, mm, :] /= np.sum(pi[0, mm, :])

    for ii in range(npix):
        photon_int[0, 0:m, ii] = np.random.gamma(1, 1500, size=m)

    pi_bg = np.zeros((num_itr, npix)) + 0.001
    ################################################################

    # Sampling the parameters
    accept_i = 0
    accept_pi = 0
    accept_eta = 0
    accept_bg = 0
    t0 = datetime.now()
    for jj in range(1, n_iter):
        numerator = n_iter - num_itr
        if jj // 10000 == jj / 10000:
            print("Iteration:", jj)
            print(f"Time interval: {datetime.now()-t0}")
            print("Pi acceptance ratio:", 100 * accept_pi / jj)
            print("I acceptance ratio:", 100 * accept_i / jj)
            print("Eta acceptance ratio:", 100 * accept_eta / jj)
            print("Background acceptance ratio:", 100 * accept_bg / jj)
            if jj < (numerator + 1):
                print(f"Eta: {np.sort(1/eta[0])}\n Background: {np.mean(pi_bg[0])} \n")

            else:
                print(f"Eta: {np.sort(1/eta[jj-numerator-1])}\n Background: {np.mean(pi_bg[jj-numerator-1])} \n")

        if jj < (numerator + 1):
            pi[0, :, :], accept_pi = sample_photon_probability(lambda_, pi[0, :, :], photon_int[0, :, :], accept_pi)
            photon_int[0, :, :], accept_i = sample_int(
                lambda_,
                pi[0, :, :],
                photon_int[0, :, :],
                pi_bg[0, :],
                npix,
                eta[0, :],
                tau_irf,
                sig_irf,
                dt_padded,
                tiled_mask,
                t_inter_p,
                num,
                accept_i,
            )
            eta[0, :], accept_eta = sample_lifetime(
                photon_int[0, :, :],
                eta[0, :],
                pi_bg[0, :],
                tau_irf,
                sig_irf,
                dt_padded,
                tiled_mask,
                t_inter_p,
                num,
                accept_eta,
            )
            pi_bg[0, :], accept_bg = (
                0,
                0,
            )  # sample_bg(pi_bg[0,:], photon_int[0, :, :], eta[0,:], tau_irf,  sig_irf, dt_padded, tiled_mask, t_inter_p, num, accept_bg)

        else:
            pi[jj - numerator, :, :], accept_pi = sample_photon_probability(
                lambda_,
                pi[jj - numerator - 1, :, :],
                photon_int[jj - numerator - 1, :, :],
                accept_pi,
            )
            photon_int[jj - numerator, :, :], accept_i = sample_int(
                lambda_,
                pi[jj - numerator - 1, :, :],
                photon_int[jj - numerator - 1, :, :],
                pi_bg[jj - numerator - 1, :],
                npix,
                eta[jj - numerator - 1, :],
                tau_irf,
                sig_irf,
                dt_padded,
                tiled_mask,
                t_inter_p,
                num,
                accept_i,
            )
            eta[jj - numerator, :], accept_eta = sample_lifetime(
                photon_int[jj - numerator - 1, :, :],
                eta[jj - numerator - 1, :],
                pi_bg[jj - numerator - 1, :],
                tau_irf,
                sig_irf,
                dt_padded,
                tiled_mask,
                t_inter_p,
                num,
                accept_eta,
            )
            pi_bg[jj - numerator, :], accept_bg = (
                0,
                0,
            )  # sample_bg(pi_bg[jj-numerator-1,:], photon_int[jj-numerator-1, :, :], eta[jj-numerator-1,:], tau_irf,  sig_irf, dt_padded, tiled_mask, t_inter_p, num, accept_bg)

    print("Pi acceptance ratio:", 100 * accept_pi / n_iter)
    print("I acceptance ratio:", 100 * accept_i / n_iter)
    print("Eta acceptance ratio:", 100 * accept_eta / n_iter)
    print("Backgroud acceptance ratio:", 100 * accept_bg / n_iter)

    return pi, photon_int, eta, pi_bg
