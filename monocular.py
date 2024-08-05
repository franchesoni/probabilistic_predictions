from pathlib import Path
import time

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from schedulefree import AdamWScheduleFree


class Scenenet(torch.utils.data.Dataset):
    # https://robotvault.bitbucket.io/scenenet-rgbd.html
    def __init__(self, root):
        self.root = Path(root)
        print("Loading Scenet from", self.root)
        self.images = sorted(list(self.root.glob("**/photo/*.jpg")))
        self.depths = sorted(list(self.root.glob("**/depth/*.png")))
        print("Done loading, found", len(self.images), "images.")

    def __getitem__(self, index):
        # Load image
        imgpath, depthpath = self.images[index], self.depths[index]
        img, depth = cv2.imread(str(imgpath)), cv2.imread(str(depthpath), cv2.IMREAD_UNCHANGED)  # read in BGR, the network doesn't care
        img, depth = cv2.resize(img, (256, 256)), cv2.resize(depth, (256, 256))
        img = torch.tensor((img / 255 - 0.5).astype(np.float32)).permute(2, 0, 1)  
        invdepth = torch.tensor((1 / (1+depth)).astype(np.float32))
        return img, invdepth

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    imgs, depths = zip(*batch)
    imgs = torch.stack(imgs)
    depths = torch.stack(depths)
    return imgs, depths

def train(max_seconds=3600, batch_size=32, lr=1e-4, beta=0.9, warmup_steps=10000, weight_decay=0.001, num_workers=12):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    model = model.to(device)
    ds = Scenenet(root='train')
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    loss_fn = torch.nn.functional.l1_loss
    optim = AdamWScheduleFree(model.parameters(), lr=lr, betas=(beta, 0.999), weight_decay=weight_decay, warmup_steps=warmup_steps)
    writer = SummaryWriter(log_dir='monocular')

    model.train()
    optim.train()

    st = time.time()
    global_step = 0
    while not (time.time() - st > max_seconds):
        for imgs, invdepths in dl:
            imgs, invdepths = imgs.to(device, non_blocking=True), invdepths.to(device, non_blocking=True)
            out = model(imgs)
            loss = loss_fn(out, invdepths)
            loss.backward()
            optim.step()
            global_step += 1

            time_so_far = time.time() - st
            speed = global_step/time_so_far
            writer.add_scalar('train/speed', speed, global_step)
            writer.add_scalar('train/loss', loss.item(), global_step)
            print(f"speed={speed:.2f}(steps/s), time={int(time_so_far)}(s), loss={loss.item():.3g}", end='\r')
    

train(max_seconds=360, warmup_steps=1000)

