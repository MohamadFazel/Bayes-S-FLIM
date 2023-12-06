import numpy as np
import matplotlib.pylab as plt
from datetime import datetime


eta = np.load("/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/Mix_Eta_1111090334.npy")
pi = np.load("/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/mix_Pi_1111090334.npy")
phi = np.load("/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/mix_Phot_1111090334.npy")

# pi1 = np.load("/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/LysoRed_Pi_1103165407.npy")
# pi2 = np.load("/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/MitoOrange_Pi_1104082304.npy")
# pi3 = np.load("/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/Viafluor_Pi_1104174450.npy")


# pin = np.mean(pi[-4000:,:,:], axis=0)
# # pin1 = np.mean(pi1[-4000:,:,:], axis=0)
# # pin2 = np.mean(pi2[-4000:,:,:], axis=0)
# # pin3 = np.mean(pi3[-4000:,:,:], axis=0)


# nam = datetime.now()

# # plt.plot(pin2[0]/np.sum(pin2[0]),'r--', label="Ground_Truth")
# plt.plot(pin[1]/np.sum(pin[1]),'r', label="2_Learned")
# plt.legend()
# plt.title("Viafluor")
# plt.savefig(f"/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/spec_Viafluor2e_{nam}.png")
# plt.show()

# # plt.plot(pin3[0]/np.sum(pin3[0]),'b--', label="Ground_Truth")
# plt.plot(pin[0]/np.sum(pin[0]),'b', label="1_Learned")
# plt.title("Viafluor")
# plt.legend()
# plt.savefig(f"/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/spec_Viafluor_{nam}.png")
# plt.show()

# plt.plot(pin1[0]/np.sum(pin1[0]),'g--', label="Ground_Truth")
# plt.plot(pin[2]/np.sum(pin[2]),'g', label="1_Learned")
# plt.title("LysoRed")
# plt.legend()
# plt.savefig(f"/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/spec_LysoRed2_{nam}.png")
# plt.show()
#----------------------------------------------------------------
phi = np.mean(phi[-18000:, :,:], axis=0)

phi = phi.reshape(phi.shape[0], -1, 64)
plt.imshow(phi[0])
# plt.title("Viafluor")
# plt.savefig(f"/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/Viafluor_{nam}.png")
plt.show()

plt.imshow(phi[1])
# plt.title("Viafluor")
# plt.savefig(f"/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/Viafluor2_{nam}.png")
plt.show()

plt.imshow(phi[2])
# plt.title("LysoRed")
# plt.savefig(f"/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/lyso_{nam}.png")
plt.show()

# plt.imshow(phi[3])
# plt.title("LysoRed")
# plt.savefig(f"/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/lyso_{nam}.png")
# plt.show()
#----------------------------------------------------------------
plt.hist(1/eta[-20000:,0], bins=100, color="red", label="Viafluor")
plt.hist(1/eta[-20000:,1], bins=100, color="b", label="Viafluor")
plt.hist(1/eta[-20000:,2], bins=100, color="g", label="Viafluor")
# plt.hist(1/eta[-20000:,3], bins=100, color="y", label="Viafluor")

# plt.plot(1/eta[:,0])
# plt.hist(1/eta[-10000:,2], bins=100, color="g", label="LysoRed")

plt.title("Lifetimes Histogram")
plt.legend()
# plt.savefig(f"/mnt/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/flim/single/liftime_Viafluor{nam}.png")
plt.show()