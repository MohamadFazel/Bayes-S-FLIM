import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def load_mat(fname):
    mat = loadmat(fname)
    return {key: np.array(value) for key, value in mat.items()}


a = load_mat("/home/reza/Downloads/Images.mat")


plt.imshow(a["Im1"], cmap="Greens")
plt.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)
plt.savefig("mohammad/Im1.png")
plt.show()

plt.imshow(a["Im2"], cmap="Oranges")
plt.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)
plt.savefig("mohammad/Im2.png")
plt.show()

plt.imshow(a["Im3"], cmap="Blues")
plt.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)
plt.savefig("mohammad/Im3.png")
plt.show()

plt.imshow(a["Im4"], cmap="Purples")
plt.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)
plt.savefig("mohammad/Im4.png")
plt.show()
