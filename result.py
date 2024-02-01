import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
import scipy.stats as sc

eta = np.load("/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/exp3/Eta_3color_shiftedIRF_Data#5_0130185744.npy")
pi = np.load("/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/exp3/Pi_3color_shiftedIRF_Data#5_0130185744.npy")
phi = np.load("/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/exp3/int_3color_shiftedIRF_Data#5_0130185744.npy")

# sigma = np.load("/media/reza/EF8A-9812/pics/4Particle/fsd/sigma_01221748.npy")
# mu = np.load("/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/4_particle/mu_01292119.npy")
x = np.linspace(375, 760, 32)

drc = "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/exp3/after"
pin = np.mean(pi[-20000:,:,:], axis=0)
# dist = np.zeros((pin.shape[0], 32))
# for i in range(pin.shape[0]):
#     dist[i, :] = .75*sc.norm(mu[i,0], sigma[i,0]).pdf(x) + .25*sc.norm(mu[i,1], sigma[i,1]).pdf(x)
    
# plt.figure(figsize=(16,9))
# # plt.plot(x, pin[8]/np.sum(pin[8]),color='b', label="Species 1")
# plt.plot(x, dist[0]/np.sum(dist[0]), '--', color='b', label="ground truth 1")

# # # plt.plot(x, pin[7]/np.sum(pin[7]), color='g', label="Species 2")
# plt.plot(x, dist[1]/np.sum(dist[1]), '--', color='g', label="ground truth 2")

# plt.plot(x, dist[2]/np.sum(dist[2]), '--', color='orange', label="ground truth 3")

# plt.plot(x, dist[3]/np.sum(dist[3]), '--', color='purple', label="ground truth 4")

# # plt.plot(x, pin[6]/np.sum(pin[6]), color='m', label="Species 5")
# # plt.plot(x, dist[4]/np.sum(dist[4]), '--', color='g', label="Ground Truth 4")

# # plt.plot(x, pin[5]/np.sum(pin[5]), color='y', label="Species 6")
# # plt.plot(x, dist[5]/np.sum(dist[5]), '--', color='y', label="Ground Truth 6")
plt.plot(x, pin[0]/np.sum(pin[0]), color='orange', label="species 1")

# plt.plot(x, pin[3]/np.sum(pin[3]), color='g', label="species 3")
plt.plot(x, pin[1]/np.sum(pin[1]), color='blue', label="species 2")

# # plt.plot(x, dist[6]/np.sum(dist[6]), '--', color='k', label="Ground Truth 7")
plt.plot(x, pin[2]/np.sum(pin[2]), color='purple', label="species 4")

# # plt.plot(x, dist[7]/np.sum(dist[7]), '--', color='#FFA07A', label="Ground Truth 8")

# # plt.plot(x, pin[4]/np.sum(pin[4]), color='#00FF00', label="Species 9")
# # plt.plot(x, dist[8]/np.sum(dist[8]), '--', color='#00FF00', label="Ground Truth 9")

plt.legend(loc='upper right', fontsize='large')
plt.xlabel('wavelength (nm)', fontsize='x-large')
plt.ylabel('distribution', fontsize='x-large')
plt.title("Species Spectrum", fontsize='xx-large')
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
# phi = np.mean(phi[-18000:, :,:], axis=0)
# phi = phi.reshape(phi.shape[0], -1, 32)

        
# # nam = 3
# plt.imshow(phi[0], cmap="Oranges")
# plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# # plt.title("Species 1")
# plt.savefig(f"{drc}/species_1.png")
# plt.show()

# plt.imshow(phi[2], cmap='Purples')
# plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# # plt.title("Species 2")
# plt.savefig(f"{drc}/species_2.png")
# plt.show()

# plt.imshow(phi[1], cmap='Blues')
# plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# # plt.title("Species 3")
# plt.savefig(f"{drc}/species_3.png")
# plt.show()
# plt.imshow(phi[3], cmap='Greens')
# plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# # plt.title("Species 4")
# plt.savefig(f"{drc}/species_4.png")
# plt.show()

# #----------------------------------------------------------------
plt.figure(figsize=(16,9))
# plt.axvline(0.5, linestyle='--', color='b', label='ground truth 1', alpha=0.4)
# plt.axvline(0.9, linestyle='--', color='g', label='ground truth 2', alpha=0.5)
# # plt.axvline(1.3, linestyle='--', color='red', label='ground truth 3', alpha=0.5)
# # plt.axvline(1.6, linestyle='--', color='c', label='ground truth 4', alpha=0.5)
# plt.axvline(2, linestyle='--', color='orange', label='ground truth 3', alpha=0.5)
# # plt.axvline(2.4, linestyle='--', color='y', label='ground truth 6', alpha=0.5)
# # plt.axvline(3.1, linestyle='--', color='k', label='ground truth 7', alpha=0.5)
# # plt.axvline(4, linestyle='--', color='#FFA07A', label='ground truth 8', alpha=0.5)
# plt.axvline(5, linestyle='--', color='purple', label='ground truth 4', alpha=0.5)

plt.hist(1/eta[-20000:,1], bins=100, color='b', alpha=1, label='species 1', density=True)
# plt.hist(1/eta[-20000:,3], bins=100, color='g', alpha=1, label='species 3', density=True)
plt.hist(1/eta[-20000:,0], bins=100, color='orange', alpha=1, label='species 2', density=True)
plt.hist(1/eta[-20000:,2], bins=100, color='purple', alpha=1, label='species 4', density=True)

# # plt.hist(1/eta[-10000:,8], bins=100, color='b', alpha=1, label='Species 1')
# # plt.hist(1/eta[-10000:,7], bins=100, color='g', alpha=1, label='Species 2')
# # plt.hist(1/eta[-10000:,6], bins=100, color='m', alpha=1, label='Species 5')
# # plt.hist(1/eta[-10000:,5], bins=100, color='y', alpha=1, label='Species 6')

# # plt.hist(1/eta[-10000:,4], bins=100, color='#00FF00', alpha=1, label='Species 9')

# plt.title("Lifetimes Histogram")

plt.legend(loc='upper right', fontsize='large')
plt.xlabel('lifetime (ns)', fontsize='x-large')
plt.ylabel('distribution', fontsize='x-large')
plt.title("Lifetimes Histogram", fontsize='xx-large')
plt.savefig(f"{drc}/liftime_{3}.png")
plt.show()


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


# plt.legend(loc='upper right', fontsize=15)
# plt.xlabel('lifetime (ns)', fontsize=30)
# plt.ylabel('distribution', fontsize=30)
# plt.title("Lifetimes Histogram", fontsize=45)
# plt.savefig(f"{drc}/liftime_{3}.png")
# plt.show()
