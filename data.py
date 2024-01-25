import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.stats import truncnorm


def generate_mixture_params(size=100):
    """Generates a single-channel 2d gradient image that represents a mixture parameter in [0,1]."""
    x = np.linspace(0, 1, size + 1)
    y = np.linspace(0, 1, size + 1)
    x, y = np.meshgrid(x, y)
    # compute the mixture parameter as a slowly chaning image
    # z = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)  # four blobs
    z = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.3**2)  # one blob
    # normalize to [0,1]
    z = (z - z.min()) / (z.max() - z.min())
    return x, y, z


def plot_mixture_param(x, y, z):
    """Plots a single-channel 2d gradient image that represents a mixture parameter in [0,1]."""
    fig, ax = plt.subplots()
    ax.imshow(z, cmap="gray", interpolation="nearest")
    ax.set_title("Mixture Parameter")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # add colorbar
    ax.figure.colorbar(ax.imshow(z, cmap="gray"))
    plt.show()


def sample_point_from_mixture(mixture_factor, g1_mean, g1_std, g2_mean, g2_std):
    """Samples a point from a mixture of two truncated Gaussians."""
    if np.random.rand() < mixture_factor:
        loc, scale = g1_mean, g1_std
    else:
        loc, scale = g2_mean, g2_std
    a, b = (0 - loc) / scale, (1 - loc) / scale
    return truncnorm.rvs(a, b, loc=loc, scale=scale, size=1)


def generate_data(N, x, y, z, g1_mean, g1_std, g2_mean, g2_std):
    """Generates a dataset of points sampled from a mixture of two truncated Gaussians."""
    X, targets = [], []
    for _ in tqdm.tqdm(range(N), desc="generating data"):
        i = np.random.randint(x.shape[0])
        j = np.random.randint(x.shape[1])
        X.append((x[i, j], y[i, j]))
        mixture_factor = z[i, j]
        targets.append(
            sample_point_from_mixture(mixture_factor, g1_mean, g1_std, g2_mean, g2_std)
        )
    return np.array(X), np.array(targets)


def plot_data_3d(X, targets):
    """Creates a scatter 3D plot that shows the data points and their targets."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], targets, c="r", marker="o")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def plot_data_2d(X, targets):
    """Creates 2d plots by averaging over the z-axis."""
    assert targets.shape[0] == X.shape[0]
    # create the image
    image = np.zeros((101, 101))
    counts = np.zeros((101, 101))
    for n in tqdm.tqdm(range(len(X))):
        j = int(np.round(X[n, 0] * 100))
        i = int(np.round(X[n, 1] * 100))
        image[i, j] += targets[n]
        counts[i, j] += 1
    image = np.nan_to_num(image / counts)
    # plot the image
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="viridis", interpolation="nearest")
    ax.set_title("Data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # add colorbar
    ax.figure.colorbar(ax.imshow(image, cmap="viridis"))
    plt.show()
