import time
import os

from scipy.io import savemat

from src.forward import *
from src.gen_shape import *
from src.sflim import run_sflim_sampler

img1 = generate_map_1().reshape(-1)
img2 = generate_map_2().reshape(-1)
img3 = generate_map_3().reshape(-1)
img4 = generate_map_6().reshape(-1)
plt.imshow(img1.reshape(-1, 32))
plt.show()
plt.imshow(img2.reshape(-1, 32))
plt.show()
plt.imshow(img3.reshape(-1, 32))
plt.show()
plt.imshow(img4.reshape(-1, 32))
plt.show()

b = 32
excitation_probs = np.random.random((b, b, 4)) * 0.000008

excitation_probs = excitation_probs.reshape(-1, 4)

for it in range(excitation_probs.shape[0]):
    if img1[it] > 0.1:
        excitation_probs[it, 0] = (
            img1[it] * 0.000004
        )  # np.random.randint(6,10) * 0.0009
    if img2[it] > 0.1:
        excitation_probs[it, 1] = (
            img2[it] * 0.000016
        )  # np.random.randint(6,10) * 0.0009
    if np.abs(img3[it]) > 0:
        excitation_probs[it, 2] = img3[it] * 0.000008  # np.random.randint(6,10)*0.0009
    if np.abs(img4[it]) > 0:
        excitation_probs[it, 3] = img4[it] * 0.00003  # np.random.randint(6,10)*0.0009

save_dir = ""

if not os.path.exists("cache"):
    os.makedirs("cache")

img = excitation_probs[:, 0].reshape(-1, b)
print(img.max())
plt.imshow(img, cmap="Blues")
plt.savefig("cache/img1.png")
plt.show()
np.save("cache/img1.npy", img)

img = excitation_probs[:, 1].reshape(-1, b)
print(img.max())
plt.imshow(img, cmap="Greens")
plt.savefig("cache/img2.png")
plt.show()
np.save("cache/img2.npy", img)

img = excitation_probs[:, 2].reshape(-1, b)
print(img.max())
plt.imshow(img, cmap="Oranges")
plt.savefig("cache/img3.png")
plt.show()
np.save("cache/img3.npy", img)

img = excitation_probs[:, 3].reshape(-1, b)
print(img.max())
plt.imshow(img, cmap="Purples")
plt.savefig("cache/img4.png")
plt.show()
np.save("cache/img4.npy", img)

# img = excitation_probs[:,3].reshape(-1,b)
# print(img.max())
# plt.imshow(img, cmap='Oranges')
# plt.savefig("simulation/9img9.png")
# plt.show()
# np.save("simulation/9img9.npy",img)

# img = excitation_probs[:,4].reshape(-1,b)
# print(img.max())
# plt.imshow(img, cmap='Purples')
# plt.savefig("simulation/9img4.png")
# plt.show()
# np.save("simulation/9img4.npy",img)

# img = excitation_probs[:,5].reshape(-1,b)
# print(img.max())
# plt.imshow(img, cmap='Reds')
# plt.savefig("simulation/9img5.png")
# plt.show()
# np.save("simulation/9img5.npy",img)

# img = excitation_probs[:,6].reshape(-1,b)
# print(img.max())
# plt.imshow(img, cmap='Reds')
# plt.savefig("simulation/9img6.png")
# plt.show()
# np.save("simulation/9img6.npy",img)

# img = excitation_probs[:,7].reshape(-1,b)
# print(img.max())
# plt.imshow(img, cmap='Reds')
# plt.savefig("simulation/9img7.png")
# plt.show()
# np.save("simulation/9img7.npy",img)

# img = excitation_probs[:,8].reshape(-1,b)
# print(img.max())
# plt.imshow(img, cmap='Reds')
# plt.savefig("simulation/9img8.png")
# plt.show()
# np.save("simulation/9img8.npy",img)


img = np.sum(excitation_probs[:, :], axis=1).reshape(-1, b)
print(img.max())
plt.imshow(img, cmap="gray")
plt.savefig("cache/img.png")
plt.show()
np.save("cache/img.npy", img)

# Set parameters for data generation
num_species = 4
num_pixels = b * b
num_pulses = 10**5
inter_pulse_time = 12.8
lifetimes = np.array(
    [0.5, 0.9, 2.0, 5.0]
)  # np.array([0.6, 0.9, 1.3, 1.6, 2, 2.4, 3.1, 4, 5])
spec_indices = np.arange(lifetimes.shape[0]) + 1
# excitation_probs = np.random.random((num_pixels, num_species)) * 0.008
print(excitation_probs.shape)
print(excitation_probs.dtype)
print(excitation_probs.max())
# exit()
irf_offset = 2.5
irf_sigma = 0.5
num_iterations = 160000
background = 0


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

np.save(f"cache/lambda_.npy", lambda_)
# np.save(f"cache/dt.npy", dt)
matlab_structure = {}

# Assign each nested list to a field in the structure
for i, sublist in enumerate(dt):
    field_name = f"field_{i}"
    matlab_structure[field_name] = np.array(sublist)

# Save the structure as a .mat file
savemat("cache/output_file.mat", matlab_structure)
savemat("sample_data/sample_syntetic.mat", {"Dt": dt, "Lambda": lambda_})

# Run SpectralFlim sampler
pi, photon_int, eta, pi_bg = run_sflim_sampler(
    dt, lambda_, irf_offset, irf_sigma, inter_pulse_time, num_iterations, num_species
)


timestr = time.strftime("%m%d%H%M")
np.save(f"cache/Pi_{timestr}.npy", pi)
np.save(f"cache/Phot_{timestr}.npy", photon_int)
np.save(f"cache/Eta_{timestr}.npy", eta)

np.save(f"cache/sigma_{timestr}.npy", sigma)
np.save(f"cache/mu_{timestr}.npy", mu)
