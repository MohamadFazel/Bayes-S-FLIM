import numpy as np
import cupy as cp
from scipy.io import loadmat, savemat
import pickle


def load_data(file_path):
    """
    Load data from a .mat file.

    Args:
    file_path (str): Path to the .mat file.

    Returns:
    tuple: (dt, lambda_) arrays from the file.
    """
    if file_path.endswith(".mat"):
        file_data = loadmat(file_path)
        dt = file_data["Dt"]
        lambda_ = file_data["Lambda"]

        if dt.ndim > 1:
            dt = np.squeeze(dt)

        return dt, lambda_
    else:
        raise ValueError("Unsupported file format. Please use .mat files.")


def prepare_data(dt):
    """
    Prepare data for analysis by padding and creating a mask.

    Args:
    dt (np.array): Input data array.

    Returns:
    tuple: (dt_padded, mask)
    """
    max_len = max(len(np.squeeze(x)) for x in dt)
    dt_padded = np.zeros((len(dt), max_len))
    mask = np.zeros((len(dt), max_len))

    for i, x in enumerate(dt):
        x = np.squeeze(x)
        dt_padded[i, : len(x)] = x
        mask[i, : len(x)] = 1

    return dt_padded, mask


def create_tiled_mask(mask, num_species, numeric):
    """
    Create a tiled mask for GPU computations.

    Args:
    mask (np.array): Input mask.
    num_species (int): Number of species.
    numeric (int): Numeric value for tiling.

    Returns:
    cp.array: Tiled mask.
    """
    return cp.asarray(np.tile(mask[None:, :, None], (num_species, 1, 1, numeric)))


def save_results(file_name, eta, pi, photon_int):
    """
    Save analysis results to a file.

    Args:
    file_name (str): Name of the file to save results.
    eta (np.array): Eta values.
    pi (np.array): Pi values.
    photon_int (np.array): Photon intensity values.
    """
    data = {"eta": eta, "pi": pi, "photon_int": photon_int}

    if file_name.endswith(".mat"):
        savemat(file_name, data)
    elif file_name.endswith(".pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(data, f)
    else:
        raise ValueError("Unsupported file format. Please use .mat or .pkl extension.")


def calculate_acceptance_rate(accepted, total):
    """
    Calculate the acceptance rate.

    Args:
    accepted (int): Number of accepted proposals.
    total (int): Total number of proposals.

    Returns:
    float: Acceptance rate.
    """
    return accepted / total if total > 0 else 0


def create_colormap():
    """
    Create a colormap for plotting.

    Returns:
    tuple: (colors, cmaps) Lists of colors and colormaps.
    """
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "black",
        "magenta",
        "yellow",
        "brown",
        "pink",
    ]
    cmaps = [
        "Reds",
        "Blues",
        "Greens",
        "Oranges",
        "Purples",
        "Greys",
        "spring",
        "Wistia",
        "copper",
        "cool",
    ]
    return colors, cmaps


def linspace_wavelength(start, end, num):
    """
    Create a linear space for wavelength.

    Args:
    start (int): Start wavelength.
    end (int): End wavelength.
    num (int): Number of points.

    Returns:
    np.array: Linear space of wavelengths.
    """
    return np.linspace(start, end, num)


def calculate_mean(data, start_index=-20000):
    """
    Calculate mean of the data from a specific start index.

    Args:
    data (np.array): Input data.
    start_index (int): Start index for calculation.

    Returns:
    np.array: Mean of the data.
    """
    return np.mean(data[start_index:], axis=0)


def reshape_data(data, img_size):
    """
    Reshape data to image size.

    Args:
    data (np.array): Input data.
    img_size (int): Size of the image.

    Returns:
    np.array: Reshaped data.
    """
    return data.reshape(data.shape[0], -1, img_size)


# Add any other utility functions as needed
