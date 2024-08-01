import torch
from numpy import mean, ceil
import time
from pathlib import Path
import schedulefree
import matplotlib.pyplot as plt

from datasets import get_dataset
from methods import get_method, get_crps_PL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ae_loss(pred, target, **kwargs):
    return torch.abs(pred - target)


def main(
    method_name,
    method_kwargs=dict(),
    dataset_name="bishop_toy",
    steps=100,
    batch_size=128,
    hidden_dim=128,
    hidden_layers=2,
    lr=1e-2,
    seed=0,
    tag="",
):
    # logging
    tag = "_" + tag if tag else ""
    fulltag = f"{method_name}_{dataset_name}{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    dstdir = Path(f"runs/{fulltag}")
    str_to_log = str(locals())
    # data
    trainds = get_dataset(dataset_name, n_samples=10000, seed=0)
    valds = get_dataset(dataset_name, n_samples=1000, seed=1)
    traindl = torch.utils.data.DataLoader(trainds, batch_size=batch_size, shuffle=True)
    val_batch_size = batch_size
    while not (len(valds) % val_batch_size == 0):
        val_batch_size -= 1
    valdl = torch.utils.data.DataLoader(valds, batch_size=val_batch_size, shuffle=False)

    torch.manual_seed(seed)
    model = get_method(method_name)(
        [trainds.get_feature_dim(), *([hidden_dim] * hidden_layers)], **method_kwargs
    ).to(DEVICE)
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=lr, warmup_steps=int(steps * 0.05)
    )

    print("=" * 30)
    print(f"Training {method_name} on {dataset_name} with {steps} steps")
    loss_curve = []
    model.train()
    optimizer.train()
    step = 0
    while True:
        for x, y in traindl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.reshape(x.shape[0], -1)
            y = y.reshape(y.shape[0], -1)
            optimizer.zero_grad()
            pred = model(x)
            loss = model.loss(y, pred)
            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())

            step += 1
            print(f"Step {step}/{steps}, Loss {loss}", end="\r")
            end_training = step >= steps
            if end_training:
                break
        if end_training:
            break

    if hasattr(model, "global_width") and not model.train_width:
        model.global_width.data = torch.tensor([mean(loss_curve[-steps // 20 :])]).to(
            DEVICE
        )
    model.eval()
    optimizer.eval()
    val_losses, val_logscores, val_crpss, val_alphas_targets = [], [], [], []
    with torch.no_grad():
        for feat_vec, target in valdl:
            feat_vec, target = feat_vec.to(DEVICE), target.to(DEVICE)
            feat_vec = feat_vec.reshape(feat_vec.shape[0], -1)
            target = target.reshape(target.shape[0], -1)
            pred = model(feat_vec)
            val_losses.append(model.loss(target, pred))
            val_logscores.append(model.get_logscore_at_y(target, pred))
            val_crpss.append(
                model.get_numerical_CRPS(
                    target, pred, lower=0., upper=1., count=1000
                )
            )
            val_alphas_targets.append(torch.concatenate((model.get_F_at_y(target, pred), target), dim=1))
            if model.is_PL():
                closed_crps = get_crps_PL(target, **model.prepare_params(pred)) 
        val_alphas_targets = torch.cat(val_alphas_targets, dim=0)  # (N, 2)
        val_alphas = val_alphas_targets[:, 0]
        val_alphas_ranks = torch.empty_like(val_alphas)
        val_alphas_ranks[val_alphas.argsort()] = torch.arange(len(val_alphas)).float()
        ece = mean((torch.abs(val_alphas - val_alphas_ranks / len(val_alphas))).cpu().numpy())
        meanvalloss = mean(val_losses)
        meanvallogscores = mean(val_logscores)
        meanvalcrpss = mean(val_crpss)


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

    # figures and log
    dstdir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(val_alphas.cpu().numpy(), bins=100, density=True)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Frequency")
    plt.savefig(dstdir / "PIT_hist.png")

    plt.figure()
    plt.plot(val_alphas.cpu().numpy(), val_alphas_ranks.cpu().numpy() / len(val_alphas), ".")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\alpha_{(i)}/N$")
    plt.savefig(dstdir / "reliability.png")

    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(dstdir / "loss_curve.png")

    # Generate a grid of y values for the PDF
    y_min = 0
    y_max = 1
    y_grid = torch.linspace(y_min, y_max, 1000).reshape(1, -1).to(DEVICE)
    x_vis = torch.linspace(-0.1, 1.1, 1000).reshape(-1, 1).to(DEVICE)

    # Calculate the Laplace PDF values for the grid
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
        plt.scatter(trainds.X.cpu().numpy(), trainds.Y.cpu().numpy(), s=1, label="Data")
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


from fire import Fire

Fire(main)
