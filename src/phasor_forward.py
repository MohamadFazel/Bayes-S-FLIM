import matplotlib.pyplot as plt
import numpy as np
from src.forward import gen_data

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
