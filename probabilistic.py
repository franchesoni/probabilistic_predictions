from functools import partial
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from scipy.stats import truncnorm
import torch
import cProfile
from base_methods import MLP, fit_torch, predict_torch

from methods import methods
from metrics import crps_single_prediction


def generate_mixture_params():
    """Generates a single-channel 2d gradient image that represents a mixture parameter in [0,1]."""
    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)
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
    for _ in range(N):
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


def main(
    # set parameters
    SEED=0,
    N=10000,
    g1_mean=0.3,
    g1_std=0.4,
    g2_mean=0.6,
    g2_std=0.1,
    visualize=False,
):
    # get data
    np.random.seed(SEED)
    x, y, z = generate_mixture_params()
    X, targets = generate_data(N, x, y, z, g1_mean, g1_std, g2_mean, g2_std)
    X, targets = torch.from_numpy(X).float(), torch.from_numpy(targets).float()
    if visualize:
        plot_mixture_param(x, y, z)
        plot_data_3d(X, targets)
        plot_data_2d(X, targets)

    X_train, y_train, X_cal, y_cal, X_test, y_test = (
        X[: N // 2],
        targets[: N // 2],
        X[N // 2 : 3 * N // 4],
        targets[N // 2 : 3 * N // 4],
        X[3 * N // 4 :],
        targets[3 * N // 4 :],
    )
    # run methods
    for method_name in methods:
        method_class = methods[method_name]
        print('running method {}'.format(method_name))
        method = method_class()
        # get model
        model = MLP(output_dim=method.get_mlp_output_dim())
        # get predictions
        fit_torch(model, X_train, y_train)
        # calibrate
        predictions_cal = predict_torch(model, X_cal)
        method.compute_extra_params(predictions_cal, y_cal)
        lower_bound, upper_bound = y_cal.min(), y_cal.max()
        # compute error
        avg_crps, counter = 0, 0
        for ind, x in enumerate(X_test):
            prediction = predict_torch(model, x[None])
            predicted_cdf = partial(method.__class__.predict_cdf, method, prediction.squeeze())
            avg_crps = avg_crps * (counter / (counter+1)) + crps_single_prediction(predicted_cdf, y_test[ind], lower_bound, upper_bound) / (counter+1)
            counter += 1
            print(f'sample {ind}/{len(X_test)}, avg_crps={avg_crps}', end='\r')
        print('Final CRPS:', avg_crps)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
    # cProfile.run("main()", sort="cumtime", filename="probabilistic.profile")
