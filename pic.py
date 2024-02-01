import numpy as np
import matplotlib.pylab as plt

drc = "/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/4_particle/after"

img = np.load("/media/reza/44ec9f87-1051-4bdf-8f53-fcf9d10c68a5/FLIM_results/4_particle/img4.npy")

plt.imshow(img, cmap='Purples')
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# plt.title("Species 2")
plt.savefig(f"{drc}/img4.png")
plt.show()

