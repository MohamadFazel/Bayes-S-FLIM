import time
from src.sflim import run_sflim_sampler
from src.forward import *

# Get the current time in a formatted string
timestr = time.strftime("%m%d%H%M%S")

# Set parameters for data generation
num_species = 9
num_pixels = 32 * 32
num_pulses = 10**5
inter_pulse_time = 12.8
lifetimes = np.array([0.6, 1.1, 1.7, 2.4, 3.1, 3.7, 4.5, 5.4, 6.5])
spec_indices = np.arange(lifetimes.shape[0]) + 1
excitation_probs = np.random.random((num_pixels, num_species)) * 0.008
irf_offset = 8
irf_sigma = 0.5
num_iterations = 250000
background = 0


# Generate synthetic data
dt, lambda_, s, mu, sigma = gen_data(num_species, num_pulses, inter_pulse_time, lifetimes, spec_indices, excitation_probs, irf_offset, irf_sigma, background)

# Run SpectralFlim sampler
pi, photon_int, eta, pi_bg = run_sflim_sampler(dt, lambda_, inter_pulse_time, irf_offset, irf_sigma, inter_pulse_time, num_iterations, num_species)

# Save results with timestamp in filenames
np.save(f"/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/SpectralFlim/NewResult/123_Pi_{timestr}.npy", pi)
np.save(f"/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/SpectralFlim/NewResult/123_Phot_{timestr}.npy", photon_int)
np.save(f"/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/SpectralFlim/NewResult/123_Eta_{timestr}.npy", eta)
