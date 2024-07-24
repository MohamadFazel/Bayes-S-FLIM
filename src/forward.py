import numpy as np
import matplotlib.pyplot as plt


def gen_data(
    n_pix, n_pulse, t_inter_p, lifetimes, spec_ind, exc_probs, tau_irf, sig_irf, bg
):
    """
    genData() generates a set of arrival times from input lifetimes
    and excitation probability.

    Args:
        npix (int): Number of pixels.
        npulse (int): Number of pulses per pixel.
        t_inter_p (float): Interpulse time (ns).
        lifetimes (array-like): Array of species lifetimes (ns).
        spec_ind (array-like): Index to assign spectrum from the list to species.
        exc_probs (array-like): Species excitation probabilities.
        tau_irf (float): Mean of the instrument response function (IRF) (ns).
        sig_irf (float): Standard deviation of the IRF (ns).
        bg (float): Background parameter for Poisson noise.

    Returns:
        tuple: A tuple containing the following:
            dt (list): Simulated arrival times per pixel.
            lambda_ (ndarray): Number of simulated photons per spectral band per pixel.
            spec_inten (ndarray): Number of simulated photons per spectral band per pixel per species.
    """
    n_pulse = int(n_pulse)
    n_spec = lifetimes.size
    mu = np.array(
        [
            [555, 537],
            [400, 415],
            [655, 640],
            [483, 502],
            [620, 635],
            [585, 603],
            [698, 678],
            [432, 454],
            [515, 531],
        ]
    )
    sigma = np.random.randint(10, 20, size=(n_spec, 2))
    sigma[:, 0] = np.random.randint(10, 30, size=n_spec)
    spec_bands = np.linspace(375, 760, 32)

    n_channel = spec_bands.size
    lambda_ = np.zeros((n_pix, n_channel))
    nn = spec_ind.size
    spec_inten = np.zeros((n_pix, nn, n_channel))
    dt = []
    s = []
    total_exc_prob = np.sum(exc_probs, axis=1)
    exc_probs = exc_probs / total_exc_prob[:, None]
    spec_dt = []
    for pp in range(n_pix):

        pulse_excitation = 1 - np.exp(-2 * sig_irf * total_exc_prob[pp])
        pulse_excitation = pulse_excitation > np.random.rand(n_pulse)

        exc_pulse_indices = np.where(pulse_excitation)[0]
        exc_species_indices = np.random.choice(
            n_spec, size=len(exc_pulse_indices), p=exc_probs[pp]
        )
        exc_species_lifetimes = lifetimes[exc_species_indices]
        exc_times = np.random.normal(
            tau_irf, sig_irf, len(exc_pulse_indices)
        ) + np.random.exponential(exc_species_lifetimes)
        arrival_times = exc_times - t_inter_p * np.floor(exc_times / t_inter_p)

        index_lambda = np.random.choice(
            2, size=exc_species_indices.size, p=[0.75, 0.25]
        )
        tmp_lamds = np.random.normal(
            mu[exc_species_indices, index_lambda],
            sigma[exc_species_indices, index_lambda],
        )
        tmp_id = np.digitize(tmp_lamds, spec_bands) - 1
        tmp_id = np.clip(tmp_id, 0, 31)
        lambda_[pp] += np.bincount(tmp_id, minlength=32)
        s.append(exc_species_indices)
        dt.append(arrival_times)
        spec_dt.append((arrival_times, tmp_id))
    return dt, lambda_, s, mu, sigma, spec_dt


dt, lambda_, s, mu, sigma, spec_dt = gen_data(
    1,
    10e5,
    12.8,
    np.array([1.7, 3.5, 5]),
    np.array([1, 2, 3]),
    np.random.rand(1, 3),
    2.53,
    0.51,
    0,
)

x, y = spec_dt[0][0], spec_dt[0][1]

spectral_channels = [[] for _ in range(32)]

for i in range(len(y)):
    it = int(y[i])
    spectral_channels[it].append(x[i])


# Define the number of bins and the range
num_bins = 255
range_bins = (0, 12.8)
# Initialize a list to hold the histograms for each channel
histograms = []

for channel_lifetimes in spectral_channels:
    histogram, bin_edges = np.histogram(
        channel_lifetimes, bins=num_bins, range=range_bins
    )
    histograms.append(histogram)
# Combine all lifetimes into one list
all_lifetimes = [lifetime for sublist in spectral_channels for lifetime in sublist]

# Calculate the histogram for all lifetimes combined
combined_histogram, combined_bin_edges = np.histogram(
    all_lifetimes, bins=num_bins, range=range_bins
)

# Plot the combined histogram
plt.figure()
plt.title("Combined Histogram")
plt.bar(np.linspace(0, 12.8, num_bins), combined_histogram, width=0.05)
plt.xlabel("Lifetime")
plt.ylabel("Frequency")
plt.show()
# Optionally, plot the histograms for visualization
for i, histogram in enumerate(histograms):
    plt.figure()
    plt.title(f"Channel {i} Histogram")
    plt.plot(np.linspace(0, 12.8, num_bins), histogram)
    plt.xlabel("Lifetime")
    plt.ylabel("Frequency")
    plt.show()
