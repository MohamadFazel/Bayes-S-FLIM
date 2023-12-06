import numpy as np
import matplotlib.pylab as plt


lyso = 0.01* np.array([37, 32, 1, 0, 1, 2, 2, 5, 5, 4, 4, 4, 11, 19, 22, 23, 22, 21, 23, 22, 23, 29, 79, 128, 128, 90, 68, 58, 54, 50, 48, 46])
lyso/=np.sum(lyso)
lyso*= (760-375)/32
print(np.sum(lyso))

plt.plot(lyso)
plt.show()