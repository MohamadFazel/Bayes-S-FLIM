import numpy as np
from scipy.ndimage import convolve
from scipy.stats import norm
from matplotlib.path import Path
import matplotlib.pylab as plt
def generate_map_1():
    # Generating the grid and Gaussian filter
    txg, tyg = np.meshgrid(np.arange(0.5, 5, 1), np.arange(0.5, 5, 1))
    filt = np.exp(-((txg - 2.5) ** 2 + (tyg - 2.5) ** 2) / (2 * 1.3 ** 2))

    # Generating map #1
    im1 = np.zeros((32, 32))
    indices = [
        (1, 6, 7), (2, 14, 15), (26, 1, 2),
        # ... (list of indices continued)
        (15, 26, 28), (14, 27, 28),
        (13, 28, 29), (12, 29, 30),
        (10, 30, 31),
    ]

    for i, j, k in indices:
        im1[i - 1, j - 1:k] = np.random.randint(80, 120)

    im1 = 3 * im1
    im1 = convolve(1.1 * im1, filt, mode='constant', cval=0)

    return im1
def generate_map_2():
    # Generating map #2
    x_range = np.arange(0.5, 32, 1)
    y_range = np.arange(0.5, 32, 1)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    im_2 = (
        (np.random.randint(18000, 22000)) * norm.pdf(X_grid, loc=27, scale=3.6) * norm.pdf(Y_grid, loc=20, scale=4.8)
    )
    im_2[im_2 > 150] = 0
    im_2[im_2 < 80] = 0

    # ... (similar block for tmp_im with different parameters)

    im_2 = 3 * im_2

    # Assuming Filt is a Gaussian filter kernel
    sigma_filt = 1.1
    size_filt = int(6 * sigma_filt + 1)
    filt = norm.pdf(np.arange(size_filt), loc=size_filt // 2, scale=sigma_filt).reshape(-1, 1)

    # Normalize the filter kernel
    filt /= np.sum(filt)

    # Convolve im_2 with the Gaussian filter
    im_2 = convolve(im_2, filt, mode='constant', cval=0.0)

    return im_2
def generate_map_3():
    # Generating map #3
    x_rnd = np.random.randint(3, 30, size=11)
    y_rnd = np.random.randint(3, 30, size=11)
    psf_rnd = 0.9 + np.random.rand(11)
    X_grid, Y_grid = np.meshgrid(np.arange(0.5, 32, 1), np.arange(0.5, 32, 1))
    im_3 = np.zeros((32, 32))

    for ii in range(len(x_rnd)):
        im_3 += (
            1000 *
            norm.pdf(X_grid, loc=x_rnd[ii], scale=psf_rnd[ii])
            * norm.pdf(Y_grid, loc=y_rnd[ii], scale=psf_rnd[ii])
        )

    im_3 = 4 * im_3

    # Assuming Filt is a Gaussian filter kernel
    sigma_filt = 1.15
    size_filt = int(6 * sigma_filt + 1)
    filt = norm.pdf(np.arange(size_filt), loc=size_filt // 2, scale=sigma_filt).reshape(-1, 1)

    # Normalize the filter kernel
    filt /= np.sum(filt)

    # Convolve im_3 with the Gaussian filter
    im_3 = convolve(2 * im_3, filt, mode='constant', cval=0.0)

    return im_3

def generate_map_4():
    # Generating map #10 (Smaller Stars)
    im_10 = np.zeros((32, 32))

    # Define the coordinates for the first smaller star-like shape
    star1_coords = [
        (16, 5), (18, 10), (22, 10), (20, 13), (21, 18),
        (16, 15), (11, 18), (12, 13), (8, 10), (14, 10)
    ]

    # Define the coordinates for the second smaller star-like shape
    star2_coords = [
        (26, 23), (28, 28), (32, 28), (30, 31), (31, 36),
        (26, 33), (21, 36), (22, 31), (18, 28), (24, 28)
    ]

    # Define the coordinates for the third smaller star-like shape with adjusted y-position
    star3_coords = [
        (28, -2), (30, 3), (34, 3), (32, 6), (33, 11),
        (28, 8), (23, 11), (24, 6), (20, 3), (26, 3)
    ]

    # Draw the smaller stars on the map
    draw_polygon(im_10, star1_coords, intensity=np.random.randint(80, 120))
    draw_polygon(im_10, star2_coords, intensity=np.random.randint(80, 120))
    draw_polygon(im_10, star3_coords, intensity=np.random.randint(80, 120))

    im_10 = 2 * im_10  # Adjust intensity if needed

    # Assuming Filt is a Gaussian filter kernel
    sigma_filt = 1.5
    size_filt = int(6 * sigma_filt + 1)
    filt = norm.pdf(np.arange(size_filt), loc=size_filt // 2, scale=sigma_filt).reshape(-1, 1)

    # Normalize the filter kernel
    filt /= np.sum(filt)

    # Convolve im_10 with the Gaussian filter
    im_10 = convolve(im_10, filt, mode='constant', cval=0.0)

    return im_10
def draw_polygon(image, vertices, intensity):
    # Draw a filled polygon on the image
    x, y = zip(*vertices)
    
    path = Path(np.column_stack((x, y)))
    x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    points = np.column_stack((x.flatten(), y.flatten()))
    
    mask = path.contains_points(points).reshape(image.shape)
    image[mask] = intensity

