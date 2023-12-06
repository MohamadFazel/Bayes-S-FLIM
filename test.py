# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.stats as sc
# import scipy.special as ss
# import sys
# import datetime
# import hashlib
# import os

# from cupyx.scipy import special
# import cupy as cp
# from modules.likelihood import *
# # def calculate_lifetime_likelihood_gpu(photon_int, eta, tau_irf, sig_irf, dt_padded, mask, t_inter_p, num):

# #     lf_cont = photon_int[:, :, None, None] * (
# #             (eta[:, None, None, None] / 2) *
# #             cp.exp(
# #                 (eta[:, None, None, None] / 2) *
# #                 (
# #                     2 * (
# #                         tau_irf - dt_padded[None, :, :, None] - num * t_inter_p
# #                     ) +
# #                     eta[:, None, None, None] * sig_irf**2
# #                 )
# #             ) *
# #             special.erfc(
# #                 (tau_irf - dt_padded[None, :, :, None] - num * t_inter_p +
# #                 eta[:, None, None, None] * sig_irf**2) /
# #                 (sig_irf * np.sqrt(2))
# #             )
# #         )
# #     lf_cont *= mask
# #     masked_arr = cp.sum(lf_cont , axis=(0,3))
# #     masked_arr = masked_arr[masked_arr!=0]
# #     return float(cp.sum(cp.log(masked_arr)))

# # def calculate_lifetime_likelihood(photon_int, eta, tau_irf,  sig_irf,dt, t_inter_p, num):

# #     a = 0
# #     for i in range(photon_int.shape[1]):
# #         dti = dt[i]
# #         lf_cont = np.sum(
# #             photon_int[:, i, np.newaxis, np.newaxis] *
# #             (
# #                 (eta[:, np.newaxis, np.newaxis] / 2) *
# #                 np.exp(
# #                     (eta[:, np.newaxis, np.newaxis] / 2) *
# #                     (
# #                         2 * (
# #                             tau_irf - dti[np.newaxis, :, np.newaxis] - np.arange(num)[np.newaxis, np.newaxis, :] * t_inter_p
# #                         ) +
# #                         eta[:, np.newaxis, np.newaxis] * sig_irf**2
# #                     )
# #                 ) *
# #                 ss.erfc(
# #                     (tau_irf - dti[np.newaxis, :, np.newaxis] - np.arange(num)[np.newaxis, np.newaxis, :] * t_inter_p +
# #                     eta[:, np.newaxis, np.newaxis] * sig_irf**2) /
# #                     (sig_irf * np.sqrt(2))
# #                 )
# #             ),
# #             axis=(0, 2)
# #         )

# #         a+= np.sum(np.log(lf_cont))
# #     return a

# # def calculate_lifetime_likelihood_gpu_int(photon_int, eta, tau_irf, sig_irf, dt_padded, mask, t_inter_p, num):

# #     lf_cont = photon_int[:, :, None, None] * (
# #             (eta[:, None, None, None] / 2) *
# #             cp.exp(
# #                 (eta[:, None, None, None] / 2) *
# #                 (
# #                     2 * (
# #                         tau_irf - dt_padded[None, :, :, None] - num * t_inter_p
# #                     ) +
# #                     eta[:, None, None, None] * sig_irf**2
# #                 )
# #             ) *
# #             special.erfc(
# #                 (tau_irf - dt_padded[None, :, :, None] - num * t_inter_p +
# #                 eta[:, None, None, None] * sig_irf**2) /
# #                 (sig_irf * np.sqrt(2))
# #             )
# #         )
# #     lf_cont *= mask
# #     masked_arr = cp.sum(lf_cont , axis=(0,3))
# #     log_masked_arr = cp.log(masked_arr)
# #     log_masked_arr[masked_arr==0] =0
# #     return cp.asnumpy(cp.sum(log_masked_arr, axis=1))

# photon_int = np.random.rand(10, 6)
# eta = np.random.rand(10)
# tau_irf = np.random.rand()
# sig_irf = np.random.rand()
# dt = [30*np.random.rand(6), 30*np.random.rand(8), 30*np.random.rand(10), 30*np.random.rand(2), 30*np.random.rand(4), 30*np.random.rand(8)]
# t_inter_p = np.random.rand()
# num = 4
# numeric = 4
# m = 10
# # Find the maximum length
# max_len = max(len(x) for x in dt)
# dt_padded = np.zeros((len(dt), max_len))
# mask = np.zeros((len(dt), max_len))
# num = cp.arange(numeric)[None, None, None, :]

# for i, x in enumerate(dt):
#     dt_padded[i, :len(x)] = x
#     mask[i, :len(x)] = 1

# tiled_mask = cp.asarray(np.tile(mask[None,:, :, None],
#                                 (m, 1, 1, numeric)))

# dt_padded = cp.asarray(dt_padded)
# # out1 = calculate_lifetime_likelihood(photon_int, eta, tau_irf, sig_irf, dt, t_inter_p, numeric)
# out2 =  calculate_lifetime_likelihood_gpu_int(cp.asarray(photon_int), cp.asarray(eta), tau_irf, sig_irf, dt_padded, tiled_mask, t_inter_p, num)
# # print(out1)
# print(out2)
# # print(out1 == out2)
# import time
# timestr = time.strftime("%m%d%H%M%S")
import cupy as cp


cp.random.gamma(2)