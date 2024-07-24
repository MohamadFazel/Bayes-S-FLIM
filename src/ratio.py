import numpy as np
import scipy.stats as sc


def sample_photon_probability(lambd, pi_old, photon_int, accept_pi):
    """
    Sample the probability of a photon from the species to be detected in the lth spectral band.

    Args:
        lambd (numpy.ndarray): representing the number of photons per spectral band.
        pi_old (numpy.ndarray): representing current photon probabilities.
        photon_int (numpy.ndarray): Array of shape (NSpecies, NPix) representing photon intensities per pixel per species.
        n_pix (int): Total number of pixels.
        accept_pi (float): Accumulated acceptance rate.

    Returns:
        tuple: Tuple containing the updated pi_old array and the updated accept_pi value.
    """

    n_species, n_channel = pi_old.shape
    alpha = np.ones(n_channel) / n_channel
    alpha_prop = 15000

    pi_new = np.copy(pi_old)
    m = np.random.choice(n_species)

    pi_new[m, :] = np.random.gamma(alpha_prop, pi_old[m, :] / alpha_prop)
    pi_new[m] /= np.sum(pi_new[m])

    tmp_top = (photon_int[:, :, None] * pi_new[:, None, :]).sum(axis=0)
    tmp_t = tmp_top[tmp_top > 0.001]
    lam = lambd[tmp_top > 0.001]
    a_top = sc.poisson.logpmf(lam, tmp_t).sum()

    tmp_bot = (photon_int[:, :, None] * pi_old[:, None, :]).sum(axis=0)
    tmp_b = tmp_bot[tmp_bot > 0.001]
    lam = lambd[tmp_bot > 0.001]
    a_bottom = sc.poisson.logpmf(lam, tmp_b).sum()

    # Calculating priors and proposal distributions
    a_prop = 0
    a_prior = 0
    for mm in range(pi_old.shape[0]):
        a_prop += np.sum(sc.dirichlet.logpdf(pi_old[mm, :], pi_new[mm, :])) - np.sum(
            sc.dirichlet.logpdf(pi_new[mm, :], pi_old[mm, :])
        )
        a_prior += np.sum(
            sc.dirichlet.logpdf(pi_new[mm, :], alpha / n_channel)
        ) - np.sum(sc.dirichlet.logpdf(pi_old[mm, :], alpha / n_channel))

    a = (a_top - a_bottom) + a_prop + a_prior
    if a > np.log(np.random.rand()):
        pi_old = pi_new
        accept_pi += 1

    return pi_old, accept_pi
