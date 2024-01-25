from functools import partial
import cProfile
import os
from pathlib import Path
import sys

from fire import Fire
import numpy as np
import torch

from base_methods import MLP, fit_torch, predict_torch
from methods import methods
from metrics import crps_batch_prediction, crps_single_prediction
from data import generate_mixture_params, generate_data, plot_mixture_param, plot_data_3d, plot_data_2d



def main(
    SEED=0,
    N=100000,
    g1_mean=0.3,
    g1_std=0.4,
    g2_mean=0.6,
    g2_std=0.1,
    visualize=False,
    size=10,
    epochs=6,
    batch_size=1024,
    lr=1,
    extra_metrics_every=1,
):
    # get data
    np.random.seed(SEED)
    x, y, z = generate_mixture_params(size)
    X, targets = generate_data(N, x, y, z, g1_mean, g1_std, g2_mean, g2_std)
    X, targets = torch.from_numpy(X).float(), torch.from_numpy(targets).float()
    if visualize:  # visualize data (optional)
        plot_mixture_param(x, y, z)
        plot_data_3d(X, targets)
        plot_data_2d(X, targets)

    # split data
    X_train, y_train, X_cal, y_cal, X_test, y_test = (
        X[:-2000],
        targets[:-2000],
        X[-2000:-1000],
        targets[-2000:-1000],
        X[-1000:],
        targets[-1000:],
    )
    # run methods
    for method_name in methods:
        print(">" * 80)
        print("running method {}".format(method_name))
        # get method
        method_class = methods[method_name]
        method = method_class()  # defines the output_dim, loss_fn, distribution prediction, etc.
        # get model
        model = MLP(output_dim=method.get_mlp_output_dim())
        # train model
        fit_torch(
            model,
            X_train,
            y_train,
            extra_metrics={  # extra metrics computed at every epoch
                "crps": partial(
                    crps_batch_prediction,
                    lambda *args, **kwargs: method.get_cdf_func(*args, **kwargs),
                    lambda *args, **kwargs: method.get_bounds(*args, **kwargs),
                )
            },
            loss_fn=method.loss_fn,
            n_epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            extra_metrics_every=extra_metrics_every,
        )
        # calibrate (compute other necessary params using the calibration set)
        predictions_cal = predict_torch(model, X_cal)
        method.compute_extra_params(predictions_cal, y_cal)
        lower_bound, upper_bound = y_cal.min(), y_cal.max()  # bounds are obtained from this set
        # compute CRPS
        avg_crps, counter = 0, 0
        for ind, x in enumerate(X_test):  # one element at a time
            prediction = predict_torch(model, x[None])
            predicted_cdf = method.get_cdf_func(prediction.squeeze())
            avg_crps = avg_crps * (counter / (counter + 1)) + crps_single_prediction(
                predicted_cdf, y_test[ind], lower_bound, upper_bound
            ) / (counter + 1)
            counter += 1
            print(f"sample {ind}/{len(X_test)}, avg_crps={avg_crps}", end="\r")
        print("Final CRPS:", avg_crps)
        print("<" * 80)


# manage command line interface
if __name__ == "__main__":
    # if --profile is passed then launch cprofile
    if "--profile" in sys.argv:
        sys.argv.remove("--profile")
        if Path("probabilistic.profile").exists():
            os.remove("main.profile")
        cProfile.run("main()", sort="cumtime", filename="main.profile")
    else:
        # fire will make the command line interface based on the function signature
        Fire(main)
