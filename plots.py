from datetime import datetime

import matplotlib.pylab as plt
import numpy as np

# put location of data here
eta_dir = ""
pi_dir = ""
phi_dir = ""

save_path = "results"
# __________________________________________________________
eta = np.load(eta_dir)
pi = np.load(pi_dir)
phi = np.load(phi_dir)

imshow_color = ["Blues", "Reds", "Greens", "Purples", "Oranges", "Greys"]
plot_color = ["blue", "red", "green", "purple", "orange", "gray"]

num = eta.shape[0]
# __________________________________________________________
pin = np.mean(pi[-num:, :, :], axis=0)

plt.figure(figsize=(16, 16))
for it in range(pin.shape[0]):
    plt.plot(pin[it] / np.sum(pin[it]), plot_color[it], label=f"spectra {it}")

plt.legend()
plt.title("spectra")
plt.savefig(f"{save_path}/spectra.png")
plt.show()
# __________________________________________________________
phi = np.mean(phi[-num:, :, :], axis=0)
for phi_it in range(phi.shape[0]):
    plt.figure(figsize=(16, 16))
    plt.imshow(phi[phi_it], cmap=imshow_color[phi_it])
    plt.colorbar()
    plt.title(f"map {phi_it}")
    plt.savefig(f"{save_path}/map_{phi_it}.png")
    plt.show()
# ___________________________________________________________

plt.figure(figsize=(16, 16))
for eta_it in range(eta.shape[0]):
    plt.hist(1 / eta[-num:, eta_it], bins=100, color=plot_color[eta_it], label=f"eta {eta_it}")
plt.legend()
plt.title("lifetime histogram")
plt.savefig(f"{save_path}/eta.png")
plt.show()
