import matplotlib.pyplot as plt

def plot_image(data, save_path):
    plt.imshow(data, cmap='gray')
    plt.savefig(save_path)
    plt.show()
