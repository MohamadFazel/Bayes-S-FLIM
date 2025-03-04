from datetime import datetime

import matplotlib.pylab as plt
import numpy as np

# put location of data here
eta_dir = ""
pi_dir = ""
phi_dir = "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/SpectralFlim/NewResult/lmv_Phot_1218155633.npy"

save_path = "results"

image_size = 128
# __________________________________________________________
eta = np.load(eta_dir)
pi = np.load(pi_dir)
phi = np.load(phi_dir)

imshow_color = ["Blues", "Reds", "Greens", "Purples", "Oranges", "Greys"]
plot_color = ["blue", "red", "green", "purple", "orange", "gray"]

num = phi.shape[0] // 2
# __________________________________________________________
pin = np.mean(pi[-num:, :, :], axis=0)

plt.figure(figsize=(16, 16))
for it in range(pin.shape[0]):
    plt.plot(pin[it] / np.sum(pin[it]), plot_color[it], label=f"spectra {it}")

plt.legend()
plt.title("Species Spectra")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Distribution")
plt.savefig(f"{save_path}/spectra.png")
plt.show()

# ___________________________________________________________

plt.figure(figsize=(16, 16))
for eta_it in range(eta.shape[1]):
    plt.hist(1 / eta[-num:, eta_it], bins=100, color=plot_color[eta_it], label=f"eta {eta_it}")
plt.legend()
plt.title("Lifetimes Histogram")
plt.xlabel("Lifetime (ns)")
plt.ylabel("Distribution")
plt.savefig(f"{save_path}/eta.png")
plt.show()

# __________________________________________________________
phi = np.mean(phi[-num:, :, :], axis=0)
phi = phi.reshape(phi.shape[0], -1, image_size)
for phi_it in range(phi.shape[0]):
    plt.figure(figsize=(16, 16))
    plt.imshow(phi[phi_it], cmap=imshow_color[phi_it])
    plt.colorbar()
    plt.title(f"map {phi_it}")
    plt.savefig(f"{save_path}/map_{phi_it}.png")
    plt.show()
