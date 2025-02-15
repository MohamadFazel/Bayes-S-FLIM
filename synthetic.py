import os
import time

from scipy.io import savemat

from src.forward import *
from src.gen_shape import *
from src.sflim import run_sflim_sampler

####################################
########## Params ##################
####################################
b = 32
num_species = 4
num_pixels = b * b
num_pulses = 10**5
inter_pulse_time = 12.8
lifetimes = np.array([0.5, 0.9, 2.0, 5.0])  # np.array([0.6, 0.9, 1.3, 1.6, 2, 2.4, 3.1, 4, 5])
irf_offset = 2.5
irf_sigma = 0.5
num_iterations = 160000
background = 0

####################################

img1 = generate_map_1().reshape(-1)
img2 = generate_map_2().reshape(-1)
img3 = generate_map_3().reshape(-1)
img4 = generate_map_6().reshape(-1)
b = 32
excitation_probs = np.random.random((b, b, 4)) * 0.000008


excitation_probs = excitation_probs.reshape(-1, 4)

for it in range(excitation_probs.shape[0]):
    if img1[it] > 0.1:
        excitation_probs[it, 0] = img1[it] * 0.000004  # np.random.randint(6,10) * 0.0009
    if img2[it] > 0.1:
        excitation_probs[it, 1] = img2[it] * 0.000016  # np.random.randint(6,10) * 0.0009
    if np.abs(img3[it]) > 0:
        excitation_probs[it, 2] = img3[it] * 0.000008  # np.random.randint(6,10)*0.0009
    if np.abs(img4[it]) > 0:
        excitation_probs[it, 3] = img4[it] * 0.00003  # np.random.randint(6,10)*0.0009

save_dir = ""

img = excitation_probs[:, 0].reshape(-1, b)
print(img.max())
plt.imshow(img, cmap="Blues")
plt.savefig("results/img1.png")
plt.show()
np.save("results/img1.npy", img)

img = excitation_probs[:, 1].reshape(-1, b)
print(img.max())
plt.imshow(img, cmap="Greens")
plt.savefig("results/img2.png")
plt.show()
np.save("results/img2.npy", img)

img = excitation_probs[:, 2].reshape(-1, b)
print(img.max())
plt.imshow(img, cmap="Oranges")
plt.savefig("results/img3.png")
plt.show()
np.save("results/img3.npy", img)

img = excitation_probs[:, 3].reshape(-1, b)
print(img.max())
plt.imshow(img, cmap="Purples")
plt.savefig("results/img4.png")
plt.show()
np.save("results/img4.npy", img)


img = np.sum(excitation_probs[:, :], axis=1).reshape(-1, b)
print(img.max())
plt.imshow(img, cmap="gray")
plt.savefig("results/img.png")
plt.show()
np.save("results/img.npy", img)

# Set parameters for data generation

spec_indices = np.arange(lifetimes.shape[0]) + 1
# Generate synthetic data
dt, lambda_, s, mu, sigma, _ = gen_data(
    num_pixels,
    num_pulses,
    inter_pulse_time,
    lifetimes,
    spec_indices,
    excitation_probs,
    irf_offset,
    irf_sigma,
    background,
)
print(lambda_.shape)
print(len(dt))
matlab_structure = {"Dt": dt, "Lambda": lambda_, "S": s, "Mu": mu, "Sigma": sigma}
# Save the structure as a .mat file
savemat(os.path.abspath(f"sample_data/sample_{time.time()}.mat"), matlab_structure)
