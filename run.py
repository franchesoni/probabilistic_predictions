import torch
from numpy import mean
import schedulefree
import matplotlib.pyplot as plt

from datasets import get_dataset
from methods import get_method

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
):
    print(locals())
    trainds = get_dataset(dataset_name, n_samples=10000, seed=0)
    valds = get_dataset(dataset_name, n_samples=1000, seed=1)
    traindl = torch.utils.data.DataLoader(trainds, batch_size=batch_size, shuffle=True)
    valdl = torch.utils.data.DataLoader(valds, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    model = get_method(method_name)([trainds.get_feature_dim(), *([hidden_dim] * hidden_layers)], **method_kwargs).to(DEVICE)
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=lr, warmup_steps=int(steps * 0.05)
    )

    loss_curve = []
    model.train()
    optimizer.train()
    step = 0
    while True:
        for x, y in traindl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.shape[0], -1)
            y = y.view(y.shape[0], -1)
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

    model.eval()
    optimizer.eval()
    val_losses = []
    with torch.no_grad():
        for feat_vec, target in valdl:
            feat_vec, target = feat_vec.to(DEVICE), target.to(DEVICE)
            feat_vec = feat_vec.view(feat_vec.shape[0], -1)
            target = target.view(target.shape[0], -1)
            pred = model(feat_vec)
            val_losses.append(model.loss(target, pred))
        print(f"Validation Loss: {mean(val_losses)}")
    if hasattr(model, "global_width") and not model.train_width:
        model.global_width.data = torch.tensor([mean(val_losses)]).to(DEVICE)


    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel("Step")
    plt.ylabel("Loss")


    # Generate a grid of y values for the PDF
    y_min = 0
    y_max = 1
    y_grid = torch.linspace(y_min, y_max, 1000).view(1, -1).to(DEVICE)
    x_vis = torch.linspace(-0.1, 1.1, 1000).view(-1, 1).to(DEVICE)


    # Calculate the Laplace PDF values for the grid
    with torch.no_grad():
        params = model(x_vis)
        energy = model.get_logscore_at_y(y_grid, params)
    # mu = y_vis.view(-1, 1)  # Median from the predictions
    # pdf_values = (1 / (2 * b)) * torch.exp(-torch.abs(y_grid - mu) / b)

    # Convert to numpy for plotting
    x_vis_np = x_vis.view(-1).cpu().numpy()
    y_grid_np = y_grid.view(-1).cpu().numpy()
    pdf_values_np = torch.exp(-energy).cpu().numpy()

    # Plot the scatter plot and the model predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(trainds.X.cpu().numpy(), trainds.Y.cpu().numpy(), s=1, label="Data")
    # plt.plot(
    #     x_vis_np, y_vis.view(-1).cpu().detach().numpy(), "r", label="Model Prediction"
    # )

    # Plot the PDF as a colored background
    plt.pcolormesh(
        x_vis_np, y_grid_np, pdf_values_np.T, shading="auto", cmap="viridis", alpha=0.5
    )

    # Add labels and legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="PDF")
    plt.legend()
    plt.show()


from fire import Fire

Fire(main)
