import torch
from itertools import product
from numpy import nanmean as mean, ceil, isnan
from time import strftime, time
from pathlib import Path
import schedulefree
import matplotlib.pyplot as plt

from datasets import get_dataset
from methods import get_method, get_crps_PL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ae_loss(pred, target, **kwargs):
    return torch.abs(pred - target)


def hparams_iterator(**kwargs):
    # Convert values to lists if they aren't already
    for key, value in kwargs.items():
        if not isinstance(value, list):
            kwargs[key] = [value]

    # Get all combinations
    keys = list(kwargs.keys())
    values = list(kwargs.values())

    # Compute the product of all parameter lists
    for combination in product(*values):
        yield dict(zip(keys, combination))


def main(
    method_name,
    method_kwargss=dict(),
    dataset_name="bishop_toy",
    steps=5000,
    max_time_per_run=180,
    batch_size=128,
    model_sizes=["base", "small", "large"],
    lrs=[1e-2, 1e-3, 1e-4],
    betas=[0.9, 0.95],
    seeds=0,
    tag="",
    select_by="crps",
):
    # logging
    tag = "_" + tag if tag else ""
    fulltag = f"{method_name}_{dataset_name}{tag}_{strftime('%Y%m%d_%H%M%S')}"
    dstdir = Path(f"runs/{fulltag}")
    str_to_log = str(locals())
    # data
    trainds = get_dataset(dataset_name, split="train")
    valds = get_dataset(dataset_name, split="val")
    testds = get_dataset(dataset_name, split="test")
    traindl = torch.utils.data.DataLoader(trainds, batch_size=batch_size, shuffle=True)
    val_batch_size = batch_size
    while not (len(valds) % val_batch_size == 0):
        val_batch_size -= 1
    valdl = torch.utils.data.DataLoader(valds, batch_size=val_batch_size, shuffle=False)
    test_batch_size = batch_size
    while not (len(testds) % test_batch_size == 0):
        test_batch_size -= 1
    testdl = torch.utils.data.DataLoader(
        testds, batch_size=test_batch_size, shuffle=False
    )

    torch.autograd.set_detect_anomaly(True)
    best_score, best_hparams = 1e9, None
    for hparams in hparams_iterator(
        seed=seeds,
        model_size=model_sizes,
        beta=betas,
        lr=lrs,
        method_kwargs=method_kwargss,
    ):
        try:
            (
                hidden_dim,
                hidden_layers,
                optimizer,
                loss_curve,
                best_hparams,
                str_to_log,
                best_score,
            ) = train_and_validate(
                method_name,
                dataset_name,
                steps,
                max_time_per_run,
                select_by,
                dstdir,
                str_to_log,
                trainds,
                traindl,
                valdl,
                best_score,
                hparams,
                best_hparams,
            )
        except Exception as e:
            print(f"Error with hparams {hparams}: {e}")
            str_to_log += f"\nError with hparams {hparams}: {e}"

    print("*" * 40)
    # now load the best model
    seed = best_hparams["seed"]
    model_size = best_hparams["model_size"]
    beta = best_hparams["beta"]
    lr = best_hparams["lr"]
    method_kwargs = best_hparams["method_kwargs"]
    model = get_method(method_name)(
        [trainds.get_feature_dim(), *([hidden_dim] * hidden_layers)], **method_kwargs
    ).to(DEVICE)
    model.load_state_dict(torch.load(dstdir / "best_model.pth"), strict=True)

    # evaluate on test
    test_alphas, test_alphas_ranks, test_ece, test_loss, test_logscore, test_crps = (
        evaluate(testdl, model, optimizer)
    )

    print(f"Test Loss: {test_loss}")
    print(f"Test Logscore: {test_logscore}")
    print(f"Test CRPS: {test_crps}")
    print(f"Test ECE: {test_ece}")
    str_to_log += (
        f"\nTest Loss: {test_loss}"
        + f"\nTest Logscore: {test_logscore}"
        + f"\nTest CRPS: {test_crps}"
        + f"\nTest ECE: {test_ece}"
    )

    st = time()
    # figures and log
    dstdir.mkdir(parents=True, exist_ok=True)

    # increase font size
    plt.rcParams.update({"font.size": 16})
    plt.figure()
    plt.hist(test_alphas.cpu().numpy().flatten(), bins=100, density=True)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Frequency")
    plt.savefig(dstdir / "PIT_hist.png")
    print("Time to plot PIT hist:", time() - st)
    st = time()

    plt.figure()
    plt.plot(
        test_alphas.cpu().numpy(),
        test_alphas_ranks.cpu().numpy() / len(test_alphas),
        ".",
    )
    plt.plot(test_alphas.cpu().numpy(), test_alphas.cpu().numpy(), "-", color="black")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\alpha_{(i)}/N$")
    plt.savefig(dstdir / "reliability.png")
    print("Time to plot reliability:", time() - st)
    st = time()

    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(dstdir / "loss_curve.png")
    print("Time to plot loss curve:", time() - st)
    st = time()

    if dataset_name == "bishop_toy":
        # Generate a grid of y values for the PDF
        y_min = 0
        y_max = 1
        y_grid = torch.linspace(y_min, y_max, 1000).reshape(1, -1).to(DEVICE)
        x_vis = torch.linspace(-0.1, 1.1, 1000).reshape(-1, 1).to(DEVICE)

        # Calculate the PDF values for the grid
        with torch.no_grad():
            params = model(x_vis)
            energy = model.get_logscore_at_y(
                y_grid.expand(x_vis.shape[0], -1).contiguous(), params
            )
        # mu = y_vis.reshape(-1, 1)  # Median from the predictions
        # pdf_values = (1 / (2 * b)) * torch.exp(-torch.abs(y_grid - mu) / b)
        plt.figure()
        plt.scatter(
            trainds.X.cpu().numpy(),
            trainds.Y.cpu().numpy(),
            s=1,
            label="Data",
            color="black",
        )
        for i in range(params.shape[1]):
            plt.plot(
                x_vis.cpu().numpy().reshape(-1),
                params[:, i].cpu().numpy(),
                label=f"Param {i}",
            )
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Params")
        plt.savefig(dstdir / "Parameters.png")

        # Convert to numpy for plotting
        x_vis_np = x_vis.reshape(-1).cpu().numpy()
        y_grid_np = y_grid.reshape(-1).cpu().numpy()
        pdf_values_np = torch.exp(-energy).cpu().numpy()
        for cbar_values, cbar_name in [
            (pdf_values_np, "PDF"),
            (energy.cpu().numpy(), "Energy"),
        ]:
            plt.figure(figsize=(10, 6))
            plt.scatter(
                trainds.X.cpu().numpy(), trainds.Y.cpu().numpy(), s=1, label="Data"
            )
            plt.pcolormesh(
                x_vis_np,
                y_grid_np,
                cbar_values.T,
                shading="auto",
                cmap="viridis",
                alpha=0.9,
            )

            # Add labels and legend
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.colorbar(label=cbar_name)
            plt.savefig(dstdir / f"{cbar_name}.png")

    # log
    with open(dstdir / "log.txt", "w") as f:
        f.write(str_to_log)
    print("Time to save log (and plot bishop):", time() - st)
    print("*" * 40)


def train_and_validate(
    method_name,
    dataset_name,
    steps,
    max_time_per_run,
    select_by,
    dstdir,
    str_to_log,
    trainds,
    traindl,
    valdl,
    best_score,
    hparams,
    best_hparams,
):
    seed = hparams["seed"]
    model_size = hparams["model_size"]
    beta = hparams["beta"]
    lr = hparams["lr"]
    method_kwargs = hparams["method_kwargs"]

    if model_size == "base":
        hidden_dim = 128
        hidden_layers = 2
    elif model_size == "small":
        hidden_dim = 64
        hidden_layers = 1
    elif model_size == "large":
        hidden_dim = 512
        hidden_layers = 4
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    torch.manual_seed(seed)
    model = get_method(method_name)(
        [trainds.get_feature_dim(), *([hidden_dim] * hidden_layers)],
        **method_kwargs,
    ).to(DEVICE)
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=lr,
        warmup_steps=int(steps * 0.05),
        betas=(beta, 0.999),
    )

    print("=" * 30)
    str_to_log += f"\n\nTraining {method_name} on {dataset_name} with {steps} steps and hparams {hparams}"
    print(
        f"Training {method_name} on {dataset_name} with {steps} steps and hparams {hparams}"
    )
    loss_curve = []
    model.train()
    optimizer.train()
    step = 0
    st = time()
    while True:
        for x, y in traindl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.reshape(x.shape[0], -1).float()
            y = y.reshape(y.shape[0], -1)
            optimizer.zero_grad()
            pred = model(x)
            loss = model.loss(y, pred)
            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())

            step += 1
            print(f"Step {step}/{steps}, Loss {loss}", end="\r")
            end_training = step >= steps or (time() - st) > max_time_per_run
            if end_training:
                break
        if end_training:
            break

    if hasattr(model, "global_width") and not model.train_width:
        model.global_width.data = torch.tensor([mean(loss_curve[-steps // 20 :])]).to(
            DEVICE
        )
    (
        val_alphas,
        val_alphas_ranks,
        ece,
        meanvalloss,
        meanvallogscores,
        meanvalcrpss,
    ) = evaluate(valdl, model, optimizer)

    print(f"Validation Loss: {meanvalloss}")
    print(f"Validation Logscore: {meanvallogscores}")
    print(f"Validation CRPS: {meanvalcrpss}")
    print(f"Validation ECE: {ece}")
    str_to_log += (
        f"\nValidation Loss: {meanvalloss}"
        + f"\nValidation Logscore: {meanvallogscores}"
        + f"\nValidation CRPS: {meanvalcrpss}"
        + f"\nValidation ECE: {ece}"
    )

    # now check if the model is better, if it is, save it
    if select_by == "crps":
        score = meanvalcrpss
    elif select_by == "logscore":
        score = meanvallogscores
    else:
        raise ValueError(f"Unknown select_by: {select_by}")
    if score < best_score:
        best_hparams = hparams
        best_state_dict = model.state_dict()
        dstdir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state_dict, dstdir / "best_model.pth")
        with open(dstdir / "best_hparams.txt", "w") as f:
            f.write(str(best_hparams))
        best_score = score
    return (
        hidden_dim,
        hidden_layers,
        optimizer,
        loss_curve,
        best_hparams,
        str_to_log,
        best_score,
    )


def evaluate(dataloader, model, optimizer):
    model.eval()
    optimizer.eval()
    losses, logscores, crpss, alphas_targets = [], [], [], []
    with torch.no_grad():
        for feat_vec, target in dataloader:
            feat_vec, target = feat_vec.to(DEVICE), target.to(DEVICE)
            feat_vec = feat_vec.reshape(feat_vec.shape[0], -1).float()
            target = target.reshape(target.shape[0], -1)
            pred = model(feat_vec)
            loss = model.loss(target, pred)
            losses.append(loss.item())
            logscores.append(model.get_logscore_at_y(target, pred).cpu())
            crpss.append(
                model.get_numerical_CRPS(
                    target, pred, lower=0.0, upper=1.0, count=1000
                ).cpu()
            )
            alphas_targets.append(
                torch.concatenate((model.get_F_at_y(target, pred), target), dim=1).cpu()
            )
        alphas_targets = torch.cat(alphas_targets, dim=0)  # (N, 2)
        alphas = alphas_targets[:, 0].float()
        alphas_ranks = torch.empty_like(alphas)
        alphas_ranks[alphas.argsort()] = torch.arange(
            len(alphas), device=alphas.device
        ).float()
        ece = mean((torch.abs(alphas - alphas_ranks / len(alphas))).cpu().numpy())
        meanvalloss = mean(losses)
        logscoresf = torch.stack(logscores).view(-1)
        assert (
            torch.isnan(logscoresf).sum() / len(logscoresf) < 0.1
        ), f"{torch.isnan(logscoresf).sum() / len(logscoresf)} of logscores are nan"
        meanvallogscores = mean(logscores)
        meanvalcrpss = mean(crpss)
    return alphas, alphas_ranks, ece, meanvalloss, meanvallogscores, meanvalcrpss


from fire import Fire

Fire(main)
