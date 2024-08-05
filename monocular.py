from pathlib import Path
from functools import partial
import itertools
from contextlib import suppress
import time

import torch
from torch.utils.tensorboard import SummaryWriter
import cv2
from schedulefree import AdamWScheduleFree


def fast_collate(batch):
    """A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)"""
    batch_size = len(batch)
    imgs = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
    for i in range(batch_size):
        imgs[i] += torch.from_numpy(batch[i][0])
    depths = torch.stack([torch.from_numpy(batch[i][1]) for i in range(batch_size)])
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
        self.images = sorted(list(self.root.glob("**/photo/*.jpg")))
        self.depths = sorted(list(self.root.glob("**/depth/*.png")))
        if sample_every > 1:
            self.images = self.images[::sample_every]
            self.depths = self.depths[::sample_every]
        print("Done loading, found", len(self.images), "images.")

    def __getitem__(self, index):
        # Load image
        imgpath, depthpath = self.images[index], self.depths[index]
        img, depth = cv2.imread(str(imgpath)), cv2.imread(
            str(depthpath), cv2.IMREAD_UNCHANGED
        )  # read in BGR, the network doesn't care
        return img, depth

    def __len__(self):
        return len(self.images)


def gpu_transform(img, depth):
    img, invdepth = img.float().div(255).sub(0.5), 1 / (1 + depth.float())
    img, invdepth = img.permute(0, 3, 1, 2), invdepth.unsqueeze(1)  # (B, C, H, W)
    img, invdepth = torch.nn.functional.interpolate(
        img, size=(256, 256), mode="bilinear"
    ), torch.nn.functional.interpolate(invdepth, size=(256, 256), mode="bilinear")
    return img, invdepth


def train(
    max_seconds=3600,
    batch_size=32,
    lr=1e-4,
    beta=0.9,
    warmup_steps=1000,
    val_every=100,
    weight_decay=0.001,
    num_workers=96,
):
    # utils
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir="runs_monocular")
    # model
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model = model.to(device)
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
    val_ds = Scenenet(root="/export/home/data/monocular/train/2", sample_every=1000)
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
    loss_fn = torch.nn.functional.l1_loss
    optim = AdamWScheduleFree(
        model.parameters(),
        lr=lr,
        betas=(beta, 0.999),
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
    )

    model.train()
    optim.train()

    st = time.time()
    global_step = 0
    while not (time.time() - st > max_seconds):
        for imgs, invdepths in dl:
            out = model(imgs).unsqueeze(
                1
            )  # (B, 1, H, W), I don't know why it's not already 4D (has no channel dim)
            loss = loss_fn(out, invdepths)
            loss.backward()
            optim.step()
            global_step += 1

            if global_step % val_every == 0:  # maybe validate
                val_scores, vis_imgs, vis_out, vis_target = validate(dl, val_dl, model, optim)
                cv2.imwrite(f"monocular/vis_imgs.png", vis_imgs)
                cv2.imwrite(f"monocular/vis_target.png", vis_target)
                cv2.imwrite(f"monocular/vis_out_{global_step}.png", vis_out)
                for score_name, score_value in val_scores.items():
                    writer.add_scalar(f"val/{score_name}", score_value, global_step)
                print(f"step={global_step}, validation scores: {val_scores}")

            time_so_far = time.time() - st
            speed = global_step / time_so_far
            writer.add_scalar("train/speed", speed, global_step)
            writer.add_scalar("train/loss", loss.item(), global_step)
            print(
                f"speed={speed:.2f}(steps/s), time={int(time_so_far)}(s), loss={loss.item():.3g}",
                end="\r",
            )


def validate(train_dl, val_dl, model, optim):
    print("validating...", end="/r")
    # schedulefree setup
    model.train()
    optim.eval()
    with torch.no_grad():
        for imgs, invdepths in itertools.islice(train_dl, 50):
            model(imgs)
    model.eval()
    # validation
    with torch.no_grad():
        scores = {"l1": 0}
        for imgs, invdepths in val_dl:
            out = model(imgs).unsqueeze(1)
            scores["l1"] += torch.nn.functional.l1_loss(
                out, invdepths
            )  # add mean over batch
        # show 8 images
        vis_imgs = (
            torch.concatenate(
                list(imgs[:8].permute(0, 2, 3, 1).add(0.5).mul(255)), dim=1
            )
            .to(torch.uint8)
            .cpu()
            .numpy()[..., ::-1]
        )  # RGB
        vis_out = (
            torch.concatenate(
                list(
                    (out[:8, 0] / out[:8, 0].max(dim=2)[0].max(dim=1)[0][:, None, None])
                    .mul(255)
                    .to(torch.uint8)
                ),
                dim=1,
            )
            .cpu()
            .numpy()
        )
        vis_target = (
            torch.concatenate(
                list(
                    (invdepths[:8, 0] / invdepths[:8, 0].max(dim=2)[0].max(dim=1)[0][
                        :, None, None
                    ]
                )
                .mul(255)
                .to(torch.uint8)
            ),
            dim=1,
        ).cpu().numpy())
        scores = {k: v / len(val_dl) for k, v in scores.items()}
    print("done validating.", end="/r")
    # schedulefree setup
    model.train()
    optim.train()
    return scores, vis_imgs, vis_out, vis_target


if __name__ == "__main__":
    from fire import Fire

    Fire(train)
