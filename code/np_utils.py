import numpy as np

def expand_images_to_3_channels(images):
    """
    Takes an images array of data format NHWC with either 1 or 0 channels
    and returns it with 3 channels

    Args:
    images (np.ndarray): with format (N, H, W, 1) or (N, H, W)

    Returns:
        (np.ndarray): with format (N, H, W, 3)
    """
    return np.stack((images.squeeze(),)*3, -1)
