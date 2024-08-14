from pathlib import Path
from functools import partial
import pickle
import itertools
from contextlib import suppress
import time

import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
import cv2
from numpy import nanmean as mean
from schedulefree import AdamWScheduleFree

from methods import get_method


# debug util
def bp():
    import ipdb

    ipdb.set_trace()


def fast_collate(batch):
    """A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)"""
    batch_size = len(batch)
    imgs = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
    depths = torch.zeros((batch_size, *batch[0][1].shape), dtype=torch.int64)
    for i in range(batch_size):
        imgs[i] += torch.from_numpy(batch[i][0])
        depths[i] += torch.from_numpy(batch[i][1])
    return imgs, depths


class PrefetchLoader:

    def __init__(
        self,
        loader,
        gpu_transform,
        device=torch.device("cuda"),
    ):

        self.loader = loader
        self.device = device
        self.is_cuda = torch.cuda.is_available() and device.type == "cuda"
        self.transform = gpu_transform

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress

        for next_input, next_target in self.loader:

            with stream_context():
                next_input = next_input.to(device=self.device, non_blocking=True)
                next_target = next_target.to(device=self.device, non_blocking=True)
                next_input, next_target = self.transform(next_input, next_target)

            if not first:
                yield input, target
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class Scenenet(torch.utils.data.Dataset):
    # https://robotvault.bitbucket.io/scenenet-rgbd.html
    # to download: `wget https://www.doc.ic.ac.uk/~bjm113/scenenet_data/train_split/train_0.tar.gz` (and the same with train_1.tar.gz)
    # to extract: `tar -xvf train_0.tar.gz``
    def __init__(self, root, sample_every=1):
        self.root = Path(root)
        print("Loading Scenet from", self.root)

        if Path(f"datacache_{self.root.name}.pkl").exists():
            with open(f"datacache_{self.root.name}.pkl", "rb") as f:
                self.images, self.depths = pickle.load(f)
        else:
            self.images = sorted(list(self.root.glob("**/photo/*.jpg")))
            self.depths = sorted(list(self.root.glob("**/depth/*.png")))
            with open(f"datacache_{self.root.name}.pkl", "wb") as f:
                pickle.dump((self.images, self.depths), f)

        if sample_every > 1:
            self.images = self.images[::sample_every]
            self.depths = self.depths[::sample_every]
        print("Done loading, found", len(self.images), "images.")

    def __getitem__(self, index):
        # Load image
        imgpath, depthpath = self.images[index], self.depths[index]
        img, depth = cv2.imread(str(imgpath)), cv2.imread(
            str(depthpath), cv2.IMREAD_UNCHANGED
        ).astype(
            int
        )  # read in BGR, the network doesn't care
        return img, depth

    def __len__(self):
        return len(self.images)


def gpu_transform(img, depth):
    # we take the images to [-0.5, 0.5] and we use the normalized log depth as target
    img, logdepth = img.float().div(255).sub(0.5), torch.log(
        1 + depth
    )  # [-0.5, 0.5], [0, 10]
    img, logdepth = img.permute(0, 3, 1, 2), logdepth.unsqueeze(1)  # (B, C, H, W)
    img, logdepth = torch.nn.functional.interpolate(
        img, size=(256, 256), mode="bilinear"
    ), torch.nn.functional.interpolate(logdepth, size=(256, 256), mode="bilinear")
    B = img.shape[0]
    logdepth = logdepth - torch.median(logdepth.view(B, -1), dim=1)[0].view(
        B, 1, 1, 1
    )  # center the depth
    logdepth = logdepth / torch.mean(logdepth.abs().view(B, -1), dim=1).view(
        B, 1, 1, 1
    )  # normalize the depth
    return img, logdepth

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)

def train(
    max_seconds=1800,
    batch_size=64,
    lr=1e-4,
    beta=0.9,
    warmup_steps=500,
    val_every=200,
    weight_decay=0.001,
    num_workers=96,
    embedding_dim=512,
    method_name="laplacewb",
    method_kwargs={},
    seed=1,
    dsfactor=8,
    device="cuda",
    tag=""
):
    # utils
    seed_everything(seed)
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tag = f"_{tag}" if tag else ""
    writer = SummaryWriter(comment=f"{method_name}{tag}")
    # model
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.scratch.output_conv[4] = torch.nn.Conv2d(
        32, embedding_dim, kernel_size=(1, 1), stride=(1, 1)
    )  # this last layer is the one that outputs the depth, we will make it output a big vector instead
    method = get_method(method_name)(layer_sizes=[512], **method_kwargs)
    model = model.to(device)
    method = method.to(device)
    # data
    train_ds = Scenenet(root="/export/home/data/monocular/train/0")
    torch_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=fast_collate,
        num_workers=num_workers,
        drop_last=True,
    )
    dl = PrefetchLoader(torch_dl, gpu_transform, device=device)
    val_ds = Scenenet(root="/export/home/data/monocular/train/1", sample_every=1000)
    val_dl = PrefetchLoader(
        torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=fast_collate,
            num_workers=num_workers,
            drop_last=True,
        ),
        gpu_transform,
        device=device,
    )
    # optim
    optim = AdamWScheduleFree(
        list(model.parameters()) + list(method.parameters()),
        lr=lr,
        betas=(beta, 0.999),
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
    )

    model.train()
    method.train()
    optim.train()

    st = time.time()
    global_step = 0
    while not (time.time() - st > max_seconds):
        for imgs, logdepths in dl:
            embeddings = model(imgs)  # (B, E, H, W)
            embeddings, logdepths = reshape_downsample(embeddings, logdepths, dsfactor)
            params = method(embeddings)  # (BHdsWds, P)
            loss = method.loss(logdepths, params)
            loss.backward()
            optim.step()
            loss_value = loss.item()

            if global_step % val_every == 0:  # maybe validate
                del embeddings, params, loss
                torch.cuda.empty_cache()
                val_scores, vis_imgs, vis_target = validate(
                    dl, val_dl, model, method, optim
                )
                cv2.imwrite(f"imgs/vis_imgs.png", vis_imgs)
                cv2.imwrite(f"imgs/vis_target.png", vis_target)
                for score_name, score_value in val_scores.items():
                    writer.add_scalar(f"val/{score_name}", score_value, global_step)
                print(f"step={global_step}, validation scores: {val_scores}")

            time_so_far = time.time() - st
            speed = global_step / time_so_far
            writer.add_scalar("train/speed", speed, global_step)
            writer.add_scalar("train/loss", loss_value, global_step)
            print(
                f"time={int(time_so_far)}(s), steps={global_step}, speed={speed:.2f}(steps/s), loss={loss_value:.3g}",
                end="\r",
            )
            global_step += 1
    # save the model
    logdir = writer.log_dir
    torch.save(model.state_dict(), f"{logdir}/model.pth")


def reshape_downsample(embeddings, logdepths, dsfactor):
    # downsample and reshape
    B, E, H, W = embeddings.shape
    # Reshape embeddings and logdepths for easy indexing
    embeddings = embeddings.reshape(B, E, H * W)  # Shape (B, E, H*W)
    logdepths = logdepths.reshape(B, 1, H * W)  # Shape (B, H*W)
    if dsfactor > 1:
        # Calculate the number of samples to keep after downsampling
        num_samples = H * W // (dsfactor**2)
        # Generate random indices to shuffle the flattened spatial dimensions
        random_indices = (
            torch.rand(B, H * W, device=embeddings.device)
            .argsort(dim=1)
            .unsqueeze(1)[:, :, :num_samples]
        )  # (B, 1, num_samples)

        # Index into the embeddings and logdepths using random_indices
        embeddings = torch.gather(
            embeddings, dim=2, index=random_indices.expand(B, E, num_samples)
        )
        logdepths = torch.gather(logdepths, dim=2, index=random_indices)
    embeddings = embeddings.permute(0, 2, 1).reshape(-1, E)  # (BHdsWds, E)
    logdepths = logdepths.reshape(-1, 1)  # (BHdsWds, 1)
    return embeddings, logdepths


def validate(train_dl, val_dl, model, method, optim, val_dsfactor=8):
    print("validating...", end="/r")
    # schedulefree setup
    model.train()
    method.train()
    optim.eval()
    with torch.no_grad():
        # for imgs, logdepths in itertools.islice(train_dl, 2):  # debug
        for imgs, logdepths in itertools.islice(train_dl, 50):
            embeddings = model(imgs)  # (B, E, H, W)
            embeddings, logdepths = reshape_downsample(embeddings, logdepths, dsfactor=val_dsfactor)
            params = method(embeddings)  # (BHdsWds, P)
    model.eval()
    method.eval()

    # validation
    with torch.no_grad():
        scores = {"logscore": [], "crps": []}
        print("validating...", end="\r")
        seed_everything(0)
        for imgs, logdepths in tqdm.tqdm(val_dl):
            embeddings = model(imgs)  # (B, E, H, W)
            embeddings, targets = reshape_downsample(embeddings, logdepths, dsfactor=val_dsfactor)
            params = method(embeddings)  # (BHdsWds, P)
            scores["logscore"].append(method.get_logscore_at_y(targets, params).cpu())
            target_range = targets.max() - targets.min()
            scores["crps"].append(
                mean(
                    method.get_numerical_CRPS(
                        targets,
                        params,
                        lower=targets.min() - target_range * 0.05,
                        upper=targets.max() + target_range * 0.05,
                        count=100,
                        divide=False,
                    ).cpu()
                )
            )
        scores["logscore"] = mean(scores["logscore"])
        scores["crps"] = mean(scores["crps"])

        # show 8 images
        vis_imgs = (
            torch.concatenate(
                list(imgs[:8].permute(0, 2, 3, 1).add(0.5).mul(255)), dim=1
            )
            .to(torch.uint8)
            .cpu()
            .numpy()[..., ::-1]
        )  # RGB

        def norm_tensor(x):
            # this function is for tensors of shape (B, H, W)
            maxs, mins = (
                x.max(dim=2)[0].max(dim=1)[0][:, None, None],
                x.min(dim=2)[0].min(dim=1)[0][:, None, None],
            )
            return (x - mins) / (maxs - mins)

        vis_target = (
            torch.concatenate(
                list(norm_tensor(logdepths[:8, 0]).mul(255).to(torch.uint8)),
                dim=1,
            )
            .cpu()
            .numpy()
        )
        scores = {k: v / len(val_dl) for k, v in scores.items()}
    print("done validating.", end="/r")
    # schedulefree setup
    model.train()
    method.train()
    optim.train()
    return scores, vis_imgs, vis_target


if __name__ == "__main__":
    from fire import Fire

    Fire(train)

    # commands
    # python monocular.py --method_name=laplacewb
    # python monocular.py --method_name=laplacescore
    # python monocular.py --method_name=mdn --method_kwargs="{n_components: 3}"
    # python monocular.py --method_name=pinball --method_kwargs="{n_quantile_levels: 128, bounds: (-5., 5.)}"
    # python monocular.py --method_name=crpsqr --method_kwargs="{n_quantile_levels: 128, bounds: (-5., 5.)}"
    # python monocular.py --method_name=ce --method_kwargs="{n_bins: 128, bounds: (-5., 5.)}"
    # python monocular.py --method_name=crpshist --method_kwargs="{n_bins: 128, bounds: (-5., 5.)}"

    # python monocular.py --method_name=iqn (DOESN'T WORK YET)
