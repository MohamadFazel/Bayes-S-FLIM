import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
import scipy.stats as sc

drc = "pics"
eta_mi = np.load(
    "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/Eta_green_0209104931.npy"
)
eta_vi = np.load(
    "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/Eta_1color_viaFluor_Lifetime_4point9ns_shiftedIRF_Offset_2point5_Sigma_point6_Bg_10percent_0204155656.npy"
)
eta_ly = np.load(
    "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/Eta_1color_LysoRed_Lifetime_3point3ns_shiftedIRF_Offset_2point5_Sigma_point54_Bg_23percent_0205161217.npy"
)

pi_mi = np.load(
    "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/Pi_green_0209104931.npy"
)
pi_vi = np.load(
    "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/Pi_1color_viaFluor_Lifetime_4point9ns_shiftedIRF_Offset_2point5_Sigma_point6_Bg_10percent_0204155656.npy"
)
pi_ly = np.load(
    "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/Pi_1color_LysoRed_Lifetime_3point3ns_shiftedIRF_Offset_2point5_Sigma_point54_Bg_23percent_0205161217.npy"
)

eta = np.load(
    "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/Eta_mixl31_0326090758.npy"
)
pi = np.load(
    "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/Pi_mixl31_0326090758.npy"
)
phi = np.load(
    "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/int_mixl31_0326090758.npy"
)

plt.figure(figsize=(16, 9))
plt.axvline(
    1 / np.mean(eta_ly),
    linestyle="--",
    color="b",
    label="MitoOrange ground truth",
    alpha=0.5,
)
plt.axvline(
    1 / np.mean(eta_vi),
    linestyle="--",
    color="r",
    label="ViaFlour ground truth",
    alpha=0.5,
)
plt.axvline(
    1 / np.mean(eta_mi[:, 1]),
    linestyle="--",
    color="g",
    label="LysoRed ground truth",
    alpha=0.5,
)

plt.hist(
    1 / eta[-20000:, 1],
    bins=100,
    color="r",
    alpha=1,
    label="ViaFlour learned",
    density=True,
)
plt.hist(
    1 / eta[-20000:, 0],
    bins=100,
    color="g",
    alpha=1,
    label="LysoRed learned",
    density=True,
)
plt.hist(
    1 / eta[-20000:, 2],
    bins=100,
    color="b",
    alpha=1,
    label="MitoOrange learned",
    density=True,
)
plt.legend(loc="upper right", fontsize=22)
plt.xlabel("lifetime (ns)", fontsize=28)
plt.ylabel("distribution", fontsize=28)
plt.title("Lifetimes Histogram", fontsize=40)
plt.savefig(f"{drc}/liftimes.png")
plt.show()

x = np.linspace(375, 760, 32)
pin_m = np.mean(pi_mi[-20000:, :, :], axis=0)
pin_v = np.mean(pi_vi[-20000:, :, :], axis=0)
pin_l = np.mean(pi_ly[-20000:, :, :], axis=0)
pin = np.mean(pi[-20000:, :, :], axis=0)

plt.figure(figsize=(16, 9))
plt.plot(x, pin_m[1] / np.sum(pin_m[1]), "--", color="g", label="LysoRed ground truth")
plt.plot(x, pin_v[0] / np.sum(pin_v[0]), "--", color="r", label="ViaFlour ground truth")
plt.plot(
    x, pin_l[0] / np.sum(pin_l[0]), "--", color="b", label="MitoOrange ground truth"
)

plt.plot(x, pin[0] / np.sum(pin[0]), color="g", label="LysoRed learned")
plt.plot(x, pin[1] / np.sum(pin[1]), color="r", label="ViaFlour learned")
plt.plot(x, pin[2] / np.sum(pin[2]), color="b", label="MitoOrange learned")
plt.legend(loc="upper right", fontsize=22)
plt.xlabel("wavelength (nm)", fontsize=28)
plt.ylabel("distribution", fontsize=28)
plt.title("Species Spectra", fontsize=40)
plt.savefig(f"{drc}/spec.png")
plt.show()


phi = np.mean(phi[-20000:, :, :], axis=0)
phi = phi.reshape(phi.shape[0], -1, 32)


# # nam = 3
plt.imshow(phi[0], cmap="Greens")
plt.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)
plt.title("MitoOrange")
plt.savefig(f"{drc}/MitoOrange.png")
plt.show()

plt.imshow(phi[1], cmap="Reds")
plt.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)
plt.title("ViaFlour")
plt.savefig(f"{drc}/ViaFlour.png")
plt.show()
plt.imshow(phi[2], cmap="Blues")
plt.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)
plt.title("LysoRed")
plt.savefig(f"{drc}/LysoRed.png")
plt.show()
# ________________________________________________________________

# eta = np.load("/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/Eta_3color_shiftedIRF_Data#4_0202093931.npy")
# pi = np.load("/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/Pi_3color_shiftedIRF_Data#4_0202093931.npy")
# phi = np.load("/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/int_3color_shiftedIRF_Data#4_0202093931.npy")

# print(eta.shape)

# plt.figure(figsize=(16,9))
# plt.axvline(1/np.mean(eta_ly), linestyle='--', color='b', label='MitoOrange ground truth', alpha=0.5)
# plt.axvline(1/np.mean(eta_vi), linestyle='--', color='r', label='ViaFlour ground truth', alpha=0.5)
# plt.axvline(1/np.mean(eta_mi[:,1]), linestyle='--', color='g', label='MitoOrange ground truth', alpha=0.5)

# plt.hist(1/eta[-20000:,0], bins=100, color='b', alpha=1, label='MitoOrange learned', density=True)
# plt.hist(1/eta[-20000:,1], bins=100, color='g', alpha=1, label='ViaFlour learned', density=True)
# plt.hist(1/eta[-20000:,2], bins=100, color='r', alpha=1, label='MitoOrange learned', density=True)
# plt.legend(loc='upper right', fontsize=22)
# plt.xlabel('lifetime (ns)', fontsize=28)
# plt.ylabel('distribution', fontsize=28)
# plt.title("Lifetimes Histogram", fontsize=40)
# plt.savefig(f"{drc}/liftime2s.png")
# plt.show()

# pin = np.mean(pi[-20000:,:,:], axis=0)

# plt.figure(figsize=(16,9))
# plt.plot(x, pin_m[1]/np.sum(pin_m[1]), '--', color='g', label="MitoOrange ground truth")
# plt.plot(x, pin_v[0]/np.sum(pin_v[0]), '--', color='r', label="ViaFlour ground truth")
# plt.plot(x, pin_l[0]/np.sum(pin_l[0]), '--', color='b', label="MitoOrange ground truth")

# plt.plot(x, pin[0]/np.sum(pin[0]), color='b', label="MitoOrange ground truth")
# plt.plot(x, pin[1]/np.sum(pin[1]), color='g', label="ViaFlour ground truth")
# plt.plot(x, pin[2]/np.sum(pin[2]), color='r', label="MitoOrange ground truth")
# plt.legend(loc='upper right', fontsize=22)
# plt.xlabel('wavelength (nm)', fontsize=28)
# plt.ylabel('distribution', fontsize=28)
# plt.title("Species Spectra", fontsize=40)
# plt.savefig(f"{drc}/spe2c.png")
# plt.show()


# phi = np.mean(phi[-20000:, :,:], axis=0)
# print(phi.shape)
# phi = phi.reshape(phi.shape[0], -1, 100)


# # nam = 3
# plt.imshow(phi[0], cmap="Greens")
# plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# plt.title("MitoOrange")
# plt.savefig(f"{drc}/MitoOrange.png")
# plt.show()

# plt.imshow(phi[1], cmap='Reds')
# plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# plt.title("ViaFlour")
# plt.savefig(f"{drc}/ViaFlour.png")
# plt.show()
# plt.imshow(phi[2], cmap='Blues')
# plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# plt.title("LysoRed")
# plt.savefig(f"{drc}/LysoRed.png")
# plt.show()


exit()

# ________________________________________________________________


phi = np.mean(phi[-18000:, :, :], axis=0)
phi = phi.reshape(phi.shape[0], -1, 64)


# # nam = 3
plt.imshow(phi[0], cmap="Greens")
plt.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)
plt.title("MitoOrange")
# plt.savefig(f"{drc}/MitoOrange.png")
plt.show()
exit()
x = np.linspace(375, 760, 32)
pin = np.mean(pi[:, :, :], axis=0)

plt.hist(1 / eta[:, 1], bins=100, color="r", alpha=1, label="species 1", density=True)
plt.hist(1 / eta[:, 0], bins=100, color="c", alpha=1, label="species 2", density=True)
plt.show()


plt.figure(figsize=(16, 9))
plt.plot(x, pin[1] / np.sum(pin[1]), color="r", label="species 1")
plt.plot(x, pin[0] / np.sum(pin[0]), color="c", label="species 2")
plt.show()

exit()
sigma = np.load("/media/reza/AHMAD/sigma_02030939.npy")
mu = np.load("/media/reza/AHMAD/mu_02030939.npy")
x = np.linspace(375, 760, 32)

drc = "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/poster/9 prt"
pin = np.mean(pi[-20000:, :, :], axis=0)
pin_l = np.mean(pi_ly[-20000:, :, :], axis=0)
pin_m = np.mean(pi_mi[-20000:, :, :], axis=0)
pin_v = np.mean(pi_vi[-20000:, :, :], axis=0)


dist = np.zeros((pin.shape[0], 32))
for i in range(pin.shape[0]):
    dist[i, :] = 0.75 * sc.norm(mu[i, 0], sigma[i, 0]).pdf(x) + 0.25 * sc.norm(
        mu[i, 1], sigma[i, 1]
    ).pdf(x)

plt.figure(figsize=(16, 9))

plt.plot(x, dist[2] / np.sum(dist[2]), "--", color="b", label="ground truth 1")
plt.plot(x, dist[8] / np.sum(dist[8]), "--", color="k", label="ground truth 2")
plt.plot(x, dist[3] / np.sum(dist[3]), "--", color="g", label="ground truth 3")
plt.plot(x, dist[1] / np.sum(dist[1]), "--", color="r", label="ground truth 4")
plt.plot(x, dist[0] / np.sum(dist[0]), "--", color="c", label="ground truth 5")
plt.plot(x, dist[7] / np.sum(dist[7]), "--", color="orange", label="ground truth 6")
plt.plot(x, dist[6] / np.sum(dist[6]), "--", color="purple", label="ground truth 7")
plt.plot(x, dist[5] / np.sum(dist[5]), "--", color="y", label="ground truth 8")
plt.plot(x, dist[4] / np.sum(dist[4]), "--", color="#00FF00", label="ground truth 9")

plt.plot(x, pin[3] / np.sum(pin[3]), color="b", label="species 1")
plt.plot(x, pin[4] / np.sum(pin[4]), color="k", label="species 2")
plt.plot(x, pin[1] / np.sum(pin[1]), color="g", label="species 3")
plt.plot(x, pin[8] / np.sum(pin[8]), color="r", label="species 4")
plt.plot(x, pin[2] / np.sum(pin[2]), color="c", label="species 5")
plt.plot(x, pin[5] / np.sum(pin[5]), color="orange", label="species 6")
plt.plot(x, pin[6] / np.sum(pin[6]), color="purple", label="species 7")
plt.plot(x, pin[7] / np.sum(pin[7]), color="y", label="species 8")
plt.plot(x, pin[0] / np.sum(pin[0]), color="#00FF00", label="species 9")

# plt.plot(x, dist[0]/np.sum(dist[0]), '--', color='b', label="ground truth 1")

# # # plt.plot(x, pin[7]/np.sum(pin[7]), color='g', label="Species 2")
# plt.plot(x, dist[1]/np.sum(dist[1]), '--', color='g', label="ground truth 2")

# plt.plot(x, dist[2]/np.sum(dist[2]), '--', color='orange', label="ground truth 3")

# plt.plot(x, dist[3]/np.sum(dist[3]), '--', color='purple', label="ground truth 4")

# # plt.plot(x, pin[6]/np.sum(pin[6]), color='m', label="Species 5")
# # plt.plot(x, dist[4]/np.sum(dist[4]), '--', color='g', label="Ground Truth 4")

# # plt.plot(x, pin[5]/np.sum(pin[5]), color='y', label="Species 6")
# # plt.plot(x, dist[5]/np.sum(dist[5]), '--', color='y', label="Ground Truth 6")
# plt.plot(x, pin_m[0]/np.sum(pin_m[0]), color='g', linestyle='--', label="ground truth (MitoOrange)")
# plt.plot(x, pin_l[0]/np.sum(pin_l[0]), color='blue', linestyle='--', label="ground truth (LysoRed)")
# plt.plot(x, pin_v[0]/np.sum(pin_v[0]), color='r', linestyle='--', label="ground truth (ViaFlour)")

# plt.plot(x, pin[0]/np.sum(pin[0]), color='g', label="learned (MitoOrange)")


# # plt.plot(x, pin[3]/np.sum(pin[3]), color='g', label="species 3")
# plt.plot(x, pin[1]/np.sum(pin[1]), color='r', label="learned (LysoRed)")

# # # plt.plot(x, dist[6]/np.sum(dist[6]), '--', color='k', label="Ground Truth 7")
# plt.plot(x, pin[2]/np.sum(pin[2]), color='b', label="learned (ViaFlour)")

# # plt.plot(x, dist[7]/np.sum(dist[7]), '--', color='#FFA07A', label="Ground Truth 8")

# # plt.plot(x, pin[4]/np.sum(pin[4]), color='#00FF00', label="Species 9")
# # plt.plot(x, dist[8]/np.sum(dist[8]), '--', color='#00FF00', label="Ground Truth 9")

plt.legend(loc="upper right", fontsize=14)
plt.xlabel("wavelength (nm)", fontsize=28)
plt.ylabel("distribution", fontsize=28)
plt.title("Species Spectra", fontsize=40)
plt.savefig(f"{drc}/spec.png")
plt.show()
# # plt.plot(np.linspace(375, 760, 32), pin[0]/np.sum(pin[0]),'red', label="Species 1")
# plt.plot(np.linspace(375, 760, 32), pin[1]/np.sum(pin[1]),'green', label="Species 2")
# plt.plot(np.linspace(375, 760, 32), pin[2]/np.sum(pin[2]),'blue', label="Species 3")
# plt.legend()
# plt.xlabel('Wavelength(nm)')
# plt.ylabel('Distribution')
# plt.title("Species Wavelength Distribution")
# plt.savefig(f"{drc}/spec.png")
# exit()
# plt.show()
# nam = "mix"
# # plt.plot(pin1[0]/np.sum(pin1[0]),'b--', label="Ground_Truth-LysoRed")
# plt.plot(pin[1]/np.sum(pin[1]),'b', label="2_Learned")
# # plt.plot(pin2[0]/np.sum(pin2[0]),'g--', label="Ground_Truth-MitoOrange")
# plt.plot(pin[0]/np.sum(pin[0]),'g', label="1_Learned")
# # plt.plot(pin3[0]/np.sum(pin3[0]),'r--', label="Ground_Truth-Viafluor")
# plt.plot(pin[2]/np.sum(pin[2]),'r', label="1_Learned")
# plt.title("spectrum")
# plt.legend()
# plt.savefig(f"{drc}/spec_{nam}.png")
# plt.show()
# #----------------------------------------------------------------
phi = np.mean(phi[-18000:, :, :], axis=0)
phi = phi.reshape(phi.shape[0], -1, 64)


# # nam = 3
plt.imshow(phi[0], cmap="Greens")
plt.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)
plt.title("MitoOrange")
# plt.savefig(f"{drc}/MitoOrange.png")
plt.show()

plt.imshow(phi[2], cmap="Blues")
plt.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)
plt.title("ViaFlour")
# plt.savefig(f"{drc}/ViaFlour.png")
plt.show()

plt.imshow(phi[1], cmap="Reds")
plt.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)
plt.title("LysoRed")
# plt.savefig(f"{drc}/LysoRed.png")
plt.show()
# plt.imshow(phi[3], cmap='Greens')
# plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# # plt.title("Species 4")
# plt.savefig(f"{drc}/species_4.png")
# plt.show()

# #----------------------------------------------------------------
eta_g9 = np.array([0.6, 0.9, 1.3, 1.6, 2, 2.4, 3.1, 4, 5.5])
plt.figure(figsize=(16, 9))
plt.axvline(0.6, linestyle="--", color="b", label="ground truth 1", alpha=0.4)
plt.axvline(0.9, linestyle="--", color="k", label="ground truth 2", alpha=0.5)
plt.axvline(1.3, linestyle="--", color="g", label="ground truth 3", alpha=0.5)
plt.axvline(1.6, linestyle="--", color="r", label="ground truth 4", alpha=0.5)
plt.axvline(2.1, linestyle="--", color="c", label="ground truth 5", alpha=0.5)
plt.axvline(2.4, linestyle="--", color="orange", label="ground truth 6", alpha=0.5)
plt.axvline(3.3, linestyle="--", color="purple", label="ground truth 7", alpha=0.5)
plt.axvline(4.1, linestyle="--", color="y", label="ground truth 8", alpha=0.5)
plt.axvline(5.3, linestyle="--", color="#00FF00", label="ground truth 9", alpha=0.5)

# plt.axvline(np.mean(1/eta_mi), linestyle='--', color='g', label='ground truth 1', alpha=0.4)
# plt.axvline(np.mean(1/eta_ly), linestyle='--', color='b', label='ground truth 1', alpha=0.4)
# plt.axvline(np.mean(1/eta_vi), linestyle='--', color='r', label='ground truth 1', alpha=0.4)
plt.hist(
    1 / eta[-20000:, 2], bins=100, color="b", alpha=1, label="species 1", density=True
)
plt.hist(
    1 / eta[-10000:, 8], bins=100, color="k", alpha=1, label="Species 2", density=True
)
plt.hist(
    1 / eta[-20000:, 3], bins=100, color="g", alpha=1, label="species 3", density=True
)
plt.hist(
    1 / eta[-20000:, 1], bins=100, color="r", alpha=1, label="species 4", density=True
)
plt.hist(
    1 / eta[-20000:, 0], bins=100, color="c", alpha=1, label="species 5", density=True
)
plt.hist(
    1 / eta[-10000:, 7],
    bins=100,
    color="orange",
    alpha=1,
    label="Species 6",
    density=True,
)
plt.hist(
    1 / eta[-10000:, 6],
    bins=100,
    color="purple",
    alpha=1,
    label="Species 7",
    density=True,
)
plt.hist(
    1 / eta[-10000:, 5], bins=100, color="y", alpha=1, label="Species 8", density=True
)
plt.hist(
    1 / eta[-10000:, 4],
    bins=100,
    color="#00FF00",
    alpha=1,
    label="Species 9",
    density=True,
)


# plt.title("Lifetimes Histogram")


# eta = np.load("/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/exp3/Eta_3color_shiftedIRF_Data#5_0130185744.npy")
# pi = np.load("/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/exp3/Pi_3color_shiftedIRF_Data#5_0130185744.npy")
# phi = np.load("/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/exp3/int_3color_shiftedIRF_Data#5_0130185744.npy")

# x = np.linspace(375, 760, 32)


# drc = "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/4_particle/after"
# pin = np.mean(pi[-20000:,:,:], axis=0)

# plt.figure(figsize=(16,9))

# plt.plot(x, pin[0]/np.sum(pin[0]), color='orange', label="species 1")

# plt.plot(x, pin[1]/np.sum(pin[1]), color='blue', label="species 3")

# plt.plot(x, pin[2]/np.sum(pin[2]), color='purple', label="species 4")

# plt.legend(loc='upper right', fontsize=15)
# plt.xlabel('wavelength (nm)', fontsize=30)
# plt.ylabel('distribution', fontsize=30)
# plt.title("Species Spectrum", fontsize=42)
# plt.savefig(f"{drc}/spec.png")
# plt.show()

# plt.figure(figsize=(16,9))
# plt.hist(1/eta[-20000:,1], bins=100, color='b', alpha=1, label='species 1', density=True)
# plt.hist(1/eta[-20000:,0], bins=100, color='orange', alpha=1, label='species 3', density=True)
# plt.hist(1/eta[-20000:,2], bins=100, color='purple', alpha=1, label='species 4', density=True)


plt.legend(loc="upper right", fontsize=14)
plt.xlabel("lifetime (ns)", fontsize=28)
plt.ylabel("distribution", fontsize=28)
plt.title("Lifetimes Histogram", fontsize=40)
plt.savefig(f"{drc}/liftimes.png")
plt.show()
