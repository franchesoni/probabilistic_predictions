from pathlib import Path
import itertools
from time import time
from pathlib import Path

import torch
from numpy import nanmean as mean
import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from datasets import get_dataset
from methods import get_method


# debug util
def bp():
    import ipdb

    ipdb.set_trace()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)


def main(
    method_name,
    method_kwargs=dict(),
    max_seconds=60,
    batch_size=128,
    num_workers=32,
    lr=1e-4,
    beta=0.9,
    warmup_steps=500,
    val_every=200,
    weight_decay=0.001,
    model_size="large",
    dataset_name="bishop_toy",
    seed=0,
    tag="",
    device="cuda:0",
):
    assert dataset_name == "bishop_toy", "Only bishop_toy is supported for now"
    # utils
    seed_everything(seed)
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tag = "_" + tag if tag else ""
    writer = SummaryWriter(comment=f"_{method_name}{tag}")
    dstdir = Path(writer.get_logdir())
    # data
    trainds = get_dataset(dataset_name, split="train", n_samples=100000)
    valds = get_dataset(dataset_name, split="val")
    testds = get_dataset(dataset_name, split="test")
    traindl = torch.utils.data.DataLoader(trainds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_batch_size = batch_size
    while not (
        len(valds) % val_batch_size == 0
    ):  # reduce val batch size until it fits nicely
        val_batch_size -= 1
    valdl = torch.utils.data.DataLoader(valds, batch_size=val_batch_size, num_workers=num_workers, shuffle=False)
    test_batch_size = batch_size
    while not (len(testds) % test_batch_size == 0):
        test_batch_size -= 1
    testdl = torch.utils.data.DataLoader(
        testds, batch_size=test_batch_size, shuffle=False
    )
    # model
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

    model = get_method(method_name)(
        [trainds.get_feature_dim(), *([hidden_dim] * hidden_layers)],
        **method_kwargs,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta, 0.999),
        weight_decay=weight_decay,
    )

    print("=" * 30)
    print(f"Training {method_name} on {dataset_name} for {max_seconds} seconds")
    model.train()
    # optimizer.train()

    st = time()
    global_step = 0
    while True:
        for x, y in traindl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, y = x.reshape(x.shape[0], -1).float(), y.reshape(y.shape[0], -1)
            optimizer.zero_grad()
            pred = model(x)
            pred.retain_grad()  # debug
            loss = model.loss(y, pred)
            loss.backward()
            writer.add_scalar("loop/pred_grad_norm", torch.norm(pred.grad).item(), global_step)

            optimizer.step()
            loss_value = loss.item()

            if (global_step + 1) % val_every == 0:
                val_scores = validate(traindl, valdl, model, optimizer, device)
                for score_name, score_value in val_scores.items():
                    if score_name.startswith("_alphas"):
                        continue
                    writer.add_scalar(
                        f"loop/val_{score_name}", score_value, global_step
                    )

            time_so_far = time() - st
            speed = global_step / time_so_far
            writer.add_scalar("loop/speed", speed, global_step)
            writer.add_scalar("loop/train_loss", loss_value, global_step)
            print(
                f"time={int(time_so_far)}(s), steps={global_step}, speed={speed:.2f}(steps/s), loss={loss_value:.3g}",
                end="\r",
            )

            global_step += 1
            end_training = (time() - st) > max_seconds
            if end_training:
                break
        if end_training:
            break

    # evaluate on test
    final_scores = validate(traindl, testdl, model, optimizer, device)
    test_alphas, test_alphas_ranks = (
        final_scores["_alphas"],
        final_scores["_alphas_rank"],
    )

    print(
        "\n"+
        f"Final results:\n"
        +f"Logscore: {final_scores['logscore']}, CRPS: {final_scores['crps']}, ECE: {final_scores['ece']}"
    )

    # figures and log
    dstdir.mkdir(parents=True, exist_ok=True)
    st = time()

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

    if dataset_name == "bishop_toy":
        # Generate a grid of y values for the PDF
        y_min = 0
        y_max = 1
        y_grid = torch.linspace(y_min, y_max, 1000).reshape(1, -1).to(device)
        x_vis = torch.linspace(-0.1, 1.1, 1000).reshape(-1, 1).to(device)

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
            testds.X.cpu().numpy(),
            testds.Y.cpu().numpy(),
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
                testds.X.cpu().numpy(), testds.Y.cpu().numpy(), s=1, label="Data"
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


def validate(train_dl, val_dl, model, optim, device):
    print("validating...", end="/r")
    model.eval()
    with torch.no_grad():
        scores = {"logscore": [], "crps": [], "_alphas": []}
        print("validating...", end="\r")
        seed_everything(0)
        for x, y in tqdm.tqdm(val_dl):
            x, y = x.to(device), y.to(device)
            x, y = x.reshape(x.shape[0], -1).float(), y.reshape(y.shape[0], -1)
            pred = model(x)
            scores["logscore"].append(model.get_logscore_at_y(y, pred).cpu())
            target_range = y.max() - y.min()
            scores["crps"].append(
                mean(
                    model.get_numerical_CRPS(
                        y,
                        pred,
                        lower=y.min() - target_range * 0.05,
                        upper=y.max() + target_range * 0.05,
                        count=100,
                        divide=False,
                    ).cpu()
                )
            )
            scores["_alphas"].append(model.get_F_at_y(y, pred).cpu())  # collect alphas
        scores["logscore"] = mean(scores["logscore"])
        scores["crps"] = mean(scores["crps"])
        scores["_alphas"] = torch.cat(scores["_alphas"], dim=0).flatten() # (N, 1)
        alphas_ranks = torch.empty_like(scores["_alphas"])
        alphas_ranks[scores["_alphas"].argsort()] = torch.arange(
            len(scores["_alphas"]), device=scores["_alphas"].device 
        ).float()
        scores["_alphas_rank"] = alphas_ranks
        scores["ece"] = mean(
            (
                torch.abs(
                    scores["_alphas"] - scores["_alphas_rank"] / len(scores["_alphas"])
                )
            )
            .cpu()
            .numpy()
        )
    print("done validating.", end="/r")
    model.train()
    return scores


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
