# Import necessary libraries
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def atan2_2pi(y, x):
    """Custom atan2 function to ensure the result is in the range [0, 2π]."""
    return np.mod(np.arctan2(y, x), 2 * np.pi)


def PhasorTransform(A, dim=1, Harmonic=1, gs_shift=0):
    # Perform FFT
    gf = np.fft.fft(A, axis=dim - 1)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-10

    # Adjust indexing based on the number of dimensions of gf
    if gf.ndim == 1:
        gf = np.conj(gf[Harmonic] / (gf[0] + epsilon)) + gs_shift
    elif gf.ndim == 2:
        if dim == 1:
            gf = np.conj(gf[Harmonic, :] / (gf[0, :] + epsilon)) + gs_shift
        elif dim == 2:
            gf = np.conj(gf[:, Harmonic] / (gf[:, 0] + epsilon)) + gs_shift
    elif gf.ndim == 3:
        if dim == 1:
            gf = np.conj(gf[Harmonic, :, :] / (gf[0, :, :] + epsilon)) + gs_shift
        elif dim == 2:
            gf = np.conj(gf[:, Harmonic, :] / (gf[:, 0, :] + epsilon)) + gs_shift
        elif dim == 3:
            gf = np.conj(gf[:, :, Harmonic] / (gf[:, :, 0] + epsilon)) + gs_shift

    # Compute G, S, Ph, M
    G = np.real(gf)
    S = np.imag(gf)
    Ph = np.arctan2(S, G)
    M = np.sqrt(S**2 + G**2)

    # Handle NaNs and Infinities
    G = np.nan_to_num(G)
    S = np.nan_to_num(S)
    Ph = np.nan_to_num(Ph)
    M = np.nan_to_num(M)

    return G, S, Ph, M


def PhasorTransform_Correction(G, S, dP, xM):
    # Compute phase and magnitude
    P = np.arctan2(S, G)
    M = np.sqrt(S**2 + G**2)

    # Apply corrections
    Pc = P - dP
    Mc = M / xM
    Sc = Mc * np.sin(Pc)
    Gc = Mc * np.cos(Pc)

    return Gc, Sc, Pc, Mc


def PCA_gs(G, S):
    x = G
    y = S
    mx = np.mean(x)
    my = np.mean(y)

    vect_pca = np.array([x - mx, y - my]).T
    pca = PCA(n_components=2)
    pca.fit(vect_pca)

    rotM = np.array(
        [
            [np.sin(np.pi / 4), np.cos(np.pi / 4)],
            [-np.cos(np.pi / 4), np.sin(np.pi / 4)],
        ]
    )

    gs_new = np.dot(rotM, np.dot(vect_pca, pca.components_.T))
    PCA_param = {
        "rotM": rotM,
        "mx": mx,
        "my": my,
        "coeff": pca.components_.T,
        "function": "PCA_param.rotM*([x-PCA_param.mx,y-PCA_param.my]*PCA_param.coeff)",
    }

    g_new = gs_new[:, 0] + mx
    s_new = gs_new[:, 1] + my

    return g_new, s_new, PCA_param


def Phasor_SFLIM_Unmixing2(
    TRES, G_tau, S_tau, n_comp, Ch_vect, Ch_vect_spectra=None, Display=0, param0=None
):
    if param0 is None:
        param0 = {}

    if Ch_vect_spectra is None:
        Ch_vect_spectra = Ch_vect

    th = np.pi / 4
    rotM = np.array([[np.sin(th), np.cos(th)], [-np.cos(th), np.sin(th)]])

    g_new, s_new, PCA_param = PCA_gs(G_tau[Ch_vect], S_tau[Ch_vect])

    if n_comp == 2:
        if "g_plusInf1" in param0:
            tmp = PCA_param.rotM @ np.dot(
                [
                    param0["g_plusInf1"] - PCA_param.mx,
                    param0["s_plusInf1"] - PCA_param.my,
                ],
                PCA_param.coeff,
            )
            param0["g_plusInf1"] = tmp[0]
            param0["s_plusInf1"] = tmp[1]
        else:
            param0["g_plusInf1"] = g_new[-1]
            param0["s_plusInf1"] = s_new[-1]

        if "g_minusInf1" in param0:
            tmp = PCA_param.rotM @ np.dot(
                [
                    param0["g_minusInf1"] - PCA_param.mx,
                    param0["s_minusInf1"] - PCA_param.my,
                ],
                PCA_param.coeff,
            )
            param0["g_minusInf1"] = tmp[0]
            param0["s_minusInf1"] = tmp[1]
        else:
            param0["g_minusInf1"] = g_new[0]
            param0["s_minusInf1"] = s_new[0]

        if "center1" not in param0:
            param0["center1"] = round(len(g_new) / 3)

        if "steepness1" not in param0:
            param0["steepness1"] = 2

        if "n1" not in param0:
            param0["n1"] = 0.5

        Param = fit_GenLogistic_GS(
            np.arange(1, len(Ch_vect) + 1), g_new, s_new, 0, param0
        )

    elif n_comp == 3:
        if "g_minusInf1" in param0:
            tmp = PCA_param.rotM @ np.dot(
                [
                    param0["g_minusInf1"] - PCA_param.mx,
                    param0["s_minusInf1"] - PCA_param.my,
                ],
                PCA_param.coeff,
            )
            param0["g_minusInf1"] = tmp[0]
            param0["s_minusInf1"] = tmp[1]
        else:
            param0["g_minusInf1"] = g_new[0]
            param0["s_minusInf1"] = s_new[0]

        if "g_plusInf1" in param0:
            tmp = PCA_param.rotM @ np.dot(
                [
                    param0["g_plusInf1"] - PCA_param.mx,
                    param0["s_plusInf1"] - PCA_param.my,
                ],
                PCA_param.coeff,
            )
            param0["g_plusInf1"] = tmp[0]
            param0["s_plusInf1"] = tmp[1]
        else:
            param0["g_plusInf1"] = g_new[len(g_new) // 2]
            param0["s_plusInf1"] = s_new[len(g_new) // 2]

        if "g_plusInf2" in param0:
            tmp = PCA_param.rotM @ np.dot(
                [
                    param0["g_plusInf2"] - PCA_param.mx,
                    param0["s_plusInf2"] - PCA_param.my,
                ],
                PCA_param.coeff,
            )
            param0["g_plusInf2"] = tmp[0]
            param0["s_plusInf2"] = tmp[1]
        else:
            param0["g_plusInf2"] = g_new[-1]
            param0["s_plusInf2"] = s_new[-1]

        if "center1" not in param0:
            param0["center1"] = round(len(g_new) / 3)

        if "steepness1" not in param0:
            param0["steepness1"] = 1 / len(g_new) * 30

        if "n1" not in param0:
            param0["n1"] = 1

        if "center2" not in param0:
            param0["center2"] = round(len(g_new) / 3 * 2)

        if "steepness2" not in param0:
            param0["steepness2"] = 1 / len(g_new) * 30

        if "n2" not in param0:
            param0["n2"] = 1

        Param = fit_DoubleGenLogistic_GS_fx(
            np.arange(1, len(Ch_vect) + 1), g_new, s_new, Display, param0
        )

    gs_unmixed_new = np.array(
        [
            [Param["g_minusInf1"], Param["g_plusInf1"], Param["g_plusInf2"]],
            [Param["s_minusInf1"], Param["s_plusInf1"], Param["s_plusInf2"]],
        ]
    )

    gs_unmix = np.dot(np.linalg.inv(rotM), gs_unmixed_new.T).T @ np.linalg.inv(
        PCA_param["coeff"]
    )
    G_unmix0 = gs_unmix[:, 0] + np.mean(G_tau[Ch_vect])
    S_unmix0 = gs_unmix[:, 1] + np.mean(S_tau[Ch_vect])

    if n_comp == 2:
        U1_tau, U2_tau = Phasor_Unmixing2comp_distance(
            G_tau[Ch_vect_spectra] + 1j * S_tau[Ch_vect_spectra],
            G_unmix0 + 1j * S_unmix0,
        )
        U_tau = np.concatenate((U1_tau, U2_tau), axis=0)
    else:
        U1_tau, U2_tau, U3_tau = Phasor_Unmixing3comp_distance(
            G_tau[Ch_vect_spectra] + 1j * S_tau[Ch_vect_spectra],
            G_unmix0 + 1j * S_unmix0,
        )
        U_tau = np.concatenate((U1_tau, U2_tau, U3_tau), axis=0)

        U_tau[U_tau < 0] = 0
        U_tau[U_tau > 1] = 1
        U_tau = U_tau / np.tile(np.sum(U_tau, axis=0), (U_tau.shape[0], 1))

    S = np.zeros_like(U_tau)
    for i in range(n_comp):
        S[i, :] = np.sum(TRES[:, Ch_vect_spectra], axis=1) * U_tau[i, :]

    G_lambda, S_lambda = PhasorTransform(TRES[:, Ch_vect_spectra], 2)
    G_lambda_pure, S_lambda_pure = PhasorTransform(S, 2)

    if n_comp == 2:
        U1_lambda, U2_lambda = Phasor_Unmixing2comp_distance(
            G_lambda.T + 1j * S_lambda.T, G_lambda_pure + 1j * S_lambda_pure
        )
        U_lambda = np.concatenate((U1_lambda.T, U2_lambda.T), axis=1)
    else:
        U1_lambda, U2_lambda, U3_lambda = Phasor_Unmixing3comp_distance(
            G_lambda.T + 1j * S_lambda.T, G_lambda_pure + 1j * S_lambda_pure
        )
        U_lambda = np.concatenate((U1_lambda.T, U2_lambda.T, U3_lambda.T), axis=1)

    L = np.zeros_like(U_lambda)
    for i in range(n_comp):
        L[:, i] = np.sum(TRES[:, Ch_vect_spectra], axis=0) * U_lambda[:, i]

    if Display == 1:

        fig, axs = plt.subplots(2, 2)

        axs[0, 1].plot(G_tau[Ch_vect], S_tau[Ch_vect], "-o")
        axs[0, 1].plot(G_unmix0, S_unmix0, "-*k")
        axs[0, 1].set_xlim([0, 1])
        axs[0, 1].set_ylim([0, 1])

        axs[1, 0].plot(S.T)
        axs[1, 0].set_yscale("linear")

        axs[1, 1].plot(L)
        axs[1, 1].set_yscale("log")

        axs[0, 0].imshow(np.real(np.log(TRES)))

        plt.show()

    return G_unmix0, S_unmix0, S, L, U_tau, U_lambda, Param


# Load the .mat files
PathName = ""
data1 = sio.loadmat(f"{PathName}SFLIM_Dataset.mat")
data2 = sio.loadmat(f"{PathName}20200319.mat")
print(data1.keys())
print(data2.keys())
# Access the Results from data1 and TRES from data2
Results = np.squeeze(data1["Results"])
Calibration = np.squeeze(data2["Calibration"])
TRES_data = Results["TRES"]
treslist = np.ndarray.tolist(TRES_data)
TRES = np.array(treslist)

xm = Calibration["xM"]
xm_list = np.ndarray.tolist(xm)
xM = np.array(xm_list)
G_tau, S_tau, Ph_tau, M_tau = PhasorTransform(TRES, dim=1, Harmonic=1)
G_tau, S_tau, _, _ = PhasorTransform_Correction(G_tau, S_tau, 0, xM)


Ch_vect = np.arange(3, 33)  # Creates an array from 3 to 32 (inclusive)
N_comp = 3  # Assigns the value 3 to N_comp

plt.plot(TRES[:, 8])
plt.plot(TRES[:, 9])
plt.plot(TRES[:, 10])
plt.plot(TRES[:, 11])
plt.plot(TRES[:, 12])
plt.plot(TRES[:, 13])
plt.show()
