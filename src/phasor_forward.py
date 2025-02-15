import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from forward import gen_data

# _______________________________________________________________________________-
n_pix = 10
n_pulse = 0.5 * 1e4
t_inter_p = 12.8
lifetimes = np.array([0.5, 2.5, 4.5])
spec_ind = np.array([1, 2, 3])
exc_probs = np.random.rand(10, 3)
tau_irf = 12.8
sig_irf = 0.52
# _______________________________________________________________________________-


dt, lambda_, s, mu, sigma, spectral_dt = gen_data(
    n_pix,
    n_pulse,
    t_inter_p,
    lifetimes,
    spec_ind,
    exc_probs,
    tau_irf,
    sig_irf,
    0,
)

with open(
    "/media/reza/6ef102d2-1499-441c-853f-c29aa6cb40ac/reza/reza/Spectral-FLIM/phasor/data/params.txt", "w"
) as file:
    file.write(
        f"n_pix = {n_pix},\n n_pulse = {n_pulse},\n t_inter_p = {t_inter_p},\n lifetimes = {lifetimes},\n tau_irf = {tau_irf},\n sig_irf = {sig_irf},\n mu = {mu},\n sigma = {sigma}"
    )

# np.save("dt_.npy", dt)
# np.save("lambda_", lambda_)

sio.savemat(
    "dataforphasor/data_5k.mat",
    {"Dt": dt, "Lambda": lambda_},
)
spectral_pix_dt = []
print(len(spectral_dt))
for spec_dt in spectral_dt:
    x, y = spec_dt[0][0], spec_dt[0][1]

    spectral_channels = [[] for _ in range(32)]

    for i in range(len(y)):
        it = int(y[i])
        spectral_channels[it].append(x[i])
    spectral_pix_dt.append(spectral_channels)
sio.savemat(
    "dataforphasor/arrival_time_in_spectral_channels_5k.mat",
    {"spectral_channels": spectral_pix_dt},
)
# exit()
# Define the number of bins and the range
num_bins = 255
range_bins = (0, t_inter_p)
# Initialize a list to hold the histograms for each channel
histograms = []

for channel_lifetimes in spectral_channels:
    histogram, bin_edges = np.histogram(channel_lifetimes, bins=num_bins, range=range_bins)
    histograms.append(histogram)
# Combine all lifetimes into one list
all_lifetimes = [lifetime for sublist in spectral_channels for lifetime in sublist]

# Calculate the histogram for all lifetimes combined
combined_histogram, combined_bin_edges = np.histogram(all_lifetimes, bins=num_bins, range=range_bins)
data_to_save = {
    "channel_histograms": histograms,
    "combined_histogram": combined_histogram,
    "bin_edges": combined_bin_edges,  # assuming all histograms share the same bin edges
}
print(np.sum(histograms))
sio.savemat("dataforphasor/histograms_5k.mat", data_to_save)

# Plot the combined histogram
# plt.figure()
# plt.title("Combined Histogram")
# plt.plot(np.linspace(0, 12.8, num_bins), combined_histogram, "r")
# plt.xlabel("Lifetime")
# plt.ylabel("Frequency")
# plt.show()
# # exit()
# plt.figure()

# Optionally, plot the histograms for visualization
# for i, histogram in enumerate(histograms):
#     # plt.title(f"Channel {i} Histogram")
#     plt.plot(np.linspace(0, 12.8, num_bins), histogram)
#     # plt.xlabel("Lifetime")
#     # plt.ylabel("Frequency")
#     if i == 10:
#         break
# plt.show()
