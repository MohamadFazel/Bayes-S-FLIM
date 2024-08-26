import numpy as np
import cupy as cp
from scipy import special
import scipy.stats as sc
from utils import prepare_data, create_tiled_mask, calculate_acceptance_rate


class Analysis:
    def __init__(self):
        self.setup_parameters()

    def setup_parameters(self):
        self.num_species = 3
        self.num_iter = 200000
        self.n_iter = 10
        self.t_inter_p = 12.85
        self.tau_irf = 2.506
        self.sig_irf = 0.51
        self.img_size = (128, 128)
        self.alpha_prop_life = 1000
        self.alpha_prior_life = 1
        self.beta_prior_life = 10
        self.alpha_prop_int = 5000
        self.alpha_prior_int = 1
        self.beta_prior_int = 2100
        self.alpha_prop_pi = 5000
        self.dt_ = None
        self.lambda_ = None
        self.eta = None
        self.pi = None
        self.photon_int = None

    def update_parameters(self, params):
        for key, value in params.items():
            setattr(self, key, value)

    def init_chain(self):
        self.n_pix = self.lambda_.shape[0]
        self.nsb = self.lambda_.shape[-1]
        self.eta = np.zeros((self.num_iter, self.num_species))
        self.pi = np.random.rand(self.num_iter, self.num_species, self.nsb)
        self.photon_int = np.zeros((self.num_iter, self.num_species, self.n_pix))
        self.eta[0, :] = np.random.rand(self.num_species)
        for mm in range(self.num_species):
            self.pi[0, mm, :] /= np.sum(self.pi[0, mm, :])
        for ii in range(self.n_pix):
            self.photon_int[0, 0 : self.num_species, ii] = np.random.gamma(
                1, 1500, size=self.num_species
            )

    def sample_int(self, it_, numerator):
        if it_ < (numerator + 1):
            i_old = self.photon_int[0, :, :].copy()
            pi = self.pi[0, :, :].copy()
            eta = self.eta[0, :].copy()
        else:
            i_old = self.photon_int[it_ - numerator - 1, :, :].copy()
            pi = self.pi[it_ - numerator - 1, :, :].copy()
            eta = self.eta[it_ - numerator - 1, :].copy()

        i_new = np.random.gamma(self.alpha_prop_int, i_old / self.alpha_prop_int)
        lf_top = self.calculate_lifetime_likelihood_gpu_int(i_new, eta)
        lf_bot = self.calculate_lifetime_likelihood_gpu_int(i_old, eta)

        tmp_top = np.sum((i_new[:, :, None] * pi[:, None, :]), axis=0)
        a_top = np.sum(sc.poisson.logpmf(self.lambda_, tmp_top), axis=1)
        a_top[np.isnan(a_top)] = 0
        a_top[np.abs(a_top) == np.inf] = 0

        tmp_bot = np.sum((i_old[:, :, None] * pi[:, None, :]), axis=0)
        a_bottom = np.sum(sc.poisson.logpmf(self.lambda_, tmp_bot), axis=1)
        a_bottom[np.isnan(a_bottom)] = 0
        a_bottom[np.abs(a_bottom) == np.inf] = 0

        a_prior = np.sum(
            sc.gamma.logpdf(
                i_new[:, :], self.alpha_prior_int, scale=self.beta_prior_int
            ),
            axis=0,
        ) - np.sum(
            sc.gamma.logpdf(
                i_old[:, :], self.alpha_prior_int, scale=self.beta_prior_int
            ),
            axis=0,
        )
        a_prop = np.sum(
            sc.gamma.logpdf(
                i_old[:, :],
                self.alpha_prop_int,
                scale=(i_new[:, :] / self.alpha_prop_int),
            ),
            axis=0,
        ) - np.sum(
            sc.gamma.logpdf(
                i_new[:, :],
                self.alpha_prop_int,
                scale=(i_old[:, :] / self.alpha_prop_int),
            ),
            axis=0,
        )

        a = (a_top - a_bottom) + (lf_top - lf_bot) + a_prop + a_prior

        cond = a > np.log(np.random.rand(self.n_pix))[None, :]
        i_old = np.where(cond, i_new, i_old)
        self.accept_i += np.sum(cond.astype(int)) * (1 / self.n_pix)

        if it_ < (numerator + 1):
            self.photon_int[0, :, :] = i_old.copy()
        else:
            self.photon_int[it_ - numerator, :, :] = i_old.copy()

    def sample_photon_spectra(self, it_, numerator):
        if it_ < (numerator + 1):
            photon_int = self.photon_int[0, :, :].copy()
            pi_old = self.pi[0, :, :].copy()
        else:
            photon_int = self.photon_int[it_ - numerator, :, :].copy()
            pi_old = self.pi[it_ - numerator - 1, :, :].copy()

        n_species, n_channel = pi_old.shape
        alpha = np.ones(n_channel) / n_channel

        pi_new = pi_old.copy()

        m = np.random.choice(n_species)

        pi_new[m, :] = np.random.gamma(
            self.alpha_prop_pi, pi_old[m, :] / self.alpha_prop_pi
        )
        pi_new[m] /= np.sum(pi_new[m])
        tmp_top = (photon_int[:, :, None] * pi_new[:, None, :]).sum(axis=0)
        tmp_t = tmp_top[tmp_top > 0.001]
        lam = self.lambda_[tmp_top > 0.001]
        a_top = sc.poisson.logpmf(lam, tmp_t).sum()

        tmp_bot = (photon_int[:, :, None] * pi_old[:, None, :]).sum(axis=0)
        tmp_b = tmp_bot[tmp_bot > 0.001]
        lam = self.lambda_[tmp_bot > 0.001]
        a_bottom = sc.poisson.logpmf(lam, tmp_b).sum()

        a_prop = 0
        a_prior = 0
        for mm in range(pi_old.shape[0]):
            a_prop += np.sum(
                sc.dirichlet.logpdf(pi_old[mm, :], pi_new[mm, :])
            ) - np.sum(sc.dirichlet.logpdf(pi_new[mm, :], pi_old[mm, :]))
            a_prior += np.sum(
                sc.dirichlet.logpdf(pi_new[mm, :], alpha / n_channel)
            ) - np.sum(sc.dirichlet.logpdf(pi_old[mm, :], alpha / n_channel))

        a = (a_top - a_bottom) + a_prop + a_prior
        if a > np.log(np.random.rand()):
            pi_old = pi_new
            self.accept_pi += 1

        if it_ < (numerator + 1):
            self.pi[0, :, :] = pi_old.copy()
        else:
            self.pi[it_ - numerator, :, :] = pi_old.copy()

    def sample_lifetime(self, it_, numerator):
        if it_ < (numerator + 1):
            photon_int = self.photon_int[0, :, :].copy()
            eta_old = self.eta[0, :].copy()
        else:
            photon_int = self.photon_int[it_ - numerator, :, :].copy()
            eta_old = self.eta[it_ - numerator - 1, :].copy()

        eta_prop = np.random.gamma(
            self.alpha_prior_life, eta_old / self.alpha_prior_life
        )

        lf_top = self.calculate_lifetime_likelihood_gpu(photon_int, eta_prop)
        lf_bot = self.calculate_lifetime_likelihood_gpu(photon_int, eta_old)
        lik_ratio = lf_top - lf_bot

        a_prior = np.sum(
            sc.gamma.logpdf(eta_prop, self.alpha_prior_life, scale=self.beta_prior_life)
        ) - np.sum(
            sc.gamma.logpdf(eta_old, self.alpha_prior_life, scale=self.beta_prior_life)
        )
        a_prop = np.sum(
            sc.gamma.logpdf(
                eta_old, self.alpha_prior_life, scale=(eta_prop / self.alpha_prior_life)
            )
        ) - np.sum(
            sc.gamma.logpdf(
                eta_prop, self.alpha_prior_life, scale=(eta_old / self.alpha_prior_life)
            )
        )

        acc_ratio = lik_ratio + a_prop + a_prior
        if acc_ratio > np.log(np.random.rand()):
            eta_old = eta_prop
            self.accept_eta += 1

        if it_ < (numerator + 1):
            self.eta[0, :] = eta_old.copy()
        else:
            self.eta[it_ - numerator, :] = eta_old.copy()

    def calculate_lifetime_likelihood_gpu_int(self, photon_int_, eta_):
        lf_cont = photon_int_[:, :, None, None] * (
            (eta_[:, None, None, None] / 2)
            * cp.exp(
                (eta_[:, None, None, None] / 2)
                * (
                    2
                    * (
                        self.tau_irf
                        - self.dt_padded[None, :, :, None]
                        - self.num * self.t_inter_p
                    )
                    + eta_[:, None, None, None] * self.sig_irf**2
                )
            )
            * special.erfc(
                (
                    self.tau_irf
                    - self.dt_padded[None, :, :, None]
                    - self.num * self.t_inter_p
                    + eta_[:, None, None, None] * self.sig_irf**2
                )
                / (self.sig_irf * cp.sqrt(2))
            )
        )
        lf_cont *= self.tiled_mask
        masked_arr = cp.sum(lf_cont, axis=(0, 3))
        masked = masked_arr.copy()
        log_masked_arr = cp.log(masked_arr)
        log_masked_arr[masked == 0] = 0
        return cp.asnumpy(cp.sum(log_masked_arr, axis=1))

    def calculate_lifetime_likelihood_gpu(self, photon_int_, eta_):
        lf_cont = photon_int_[:, :, None, None] * (
            (eta_[:, None, None, None] / 2)
            * cp.exp(
                (eta_[:, None, None, None] / 2)
                * (
                    2
                    * (
                        self.tau_irf
                        - self.dt_padded[None, :, :, None]
                        - self.num * self.t_inter_p
                    )
                    + eta_[:, None, None, None] * self.sig_irf**2
                )
            )
            * special.erfc(
                (
                    self.tau_irf
                    - self.dt_padded[None, :, :, None]
                    - self.num * self.t_inter_p
                    + eta_[:, None, None, None] * self.sig_irf**2
                )
                / (self.sig_irf * cp.sqrt(2))
            )
        )
        lf_cont *= self.tiled_mask
        masked_arr = cp.sum(lf_cont, axis=(0, 3))
        masked = masked_arr.copy()
        masked_arr = masked_arr[masked != 0]
        return float(cp.sum(cp.log(masked_arr)))

    def run_analysis(self):
        self.dt_padded, mask = prepare_data(self.dt_)
        self.dt_padded = cp.asarray(self.dt_padded)
        numeric = 4
        self.num = cp.arange(numeric)[None, None, None, :]
        self.tiled_mask = create_tiled_mask(mask, self.num_species, numeric)

        self.accept_i = 0
        self.accept_pi = 0
        self.accept_eta = 0

        self.init_chain()
        numerator = self.num_iter - self.n_iter

        for jj in range(1, self.num_iter):
            self.sample_int(jj, numerator)
            self.sample_photon_spectra(jj, numerator)
            self.sample_lifetime(jj, numerator)

            if jj % 1000 == 0:  # Print progress every 1000 iterations
                print(f"Iteration {jj}/{self.num_iter}")

        # Calculate acceptance rates
        accept_rate_i = calculate_acceptance_rate(self.accept_i, self.num_iter)
        accept_rate_pi = calculate_acceptance_rate(self.accept_pi, self.num_iter)
        accept_rate_eta = calculate_acceptance_rate(self.accept_eta, self.num_iter)

        print(f"Acceptance rate for intensity: {accept_rate_i:.2f}")
        print(f"Acceptance rate for spectra: {accept_rate_pi:.2f}")
        print(f"Acceptance rate for lifetime: {accept_rate_eta:.2f}")

        return self.eta, self.pi, self.photon_int

    def set_data(self, dt, lambda_):
        self.dt_ = dt
        self.lambda_ = lambda_

    def get_results(self):
        return self.eta, self.pi, self.photon_int
