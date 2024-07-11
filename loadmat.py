import scipy.io as sio
import matplotlib.pyplot as plt

a = sio.loadmat("/home/reza/Downloads/Results.mat")
ims = a['Ims']

plt.imshow(ims[:,:,1], cmap='Greens')
plt.show()