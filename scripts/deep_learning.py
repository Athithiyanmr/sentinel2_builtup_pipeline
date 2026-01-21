# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# #!/usr/bin/env python3
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ==================================================
# DEVICE
# ==================================================
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ==================================================
# DATASET
# ==================================================
class BuiltupDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir):
        self.imgs = sorted(Path(img_dir).glob("*.npy"))
        self.masks = sorted(Path(mask_dir).glob("*.npy"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = np.load(self.imgs[i]).astype("float32") / 10000.0
        y = np.load(self.masks[i]).astype("float32")
        return torch.tensor(x), torch.tensor(y).unsqueeze(0)

# ==================================================
# TRAIN
# ==================================================
def train_model(patch_img, patch_msk, model_path, epochs=25, batch_size=4):

    ds = BuiltupDataset(patch_img, patch_msk)

    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=5,   # B02,B03,B04,B08,B11
        classes=1
    ).to(DEVICE)
    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss  = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        model.train()
        t_loss = 0

        for x,y in tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}"):
            x,y = x.to(DEVICE), y.to(DEVICE)
            p = model(x)
            loss = dice_loss(p, y) + bce_loss(p, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            t_loss += loss.item()

        print("Train loss:", t_loss/len(train_dl))

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(DEVICE), y.to(DEVICE)
                p = model(x)
                v_loss += (dice_loss(p, y) + bce_loss(p, y)).item()   

        print("Val loss:", v_loss/len(val_dl))

        torch.save(model.state_dict(), model_path)

    return model

# ==================================================
# PREDICT TILE
# ==================================================
def predict_tile(model, stack_path, out_prob, out_mask, patch=256, thresh=0.6):

    model.eval()

    with rasterio.open(stack_path) as src:
        profile = src.profile
        H, W = src.height, src.width

        profile.update(dtype="float32", count=1, compress="DEFLATE")

        with rasterio.open(out_prob, "w", **profile) as dp, \
             rasterio.open(out_mask, "w", **{**profile, "dtype": "uint8"}) as dm:

            for row in range(0, H, patch):
                for col in range(0, W, patch):

                    h = min(patch, H - row)
                    w = min(patch, W - col)

                    win = Window(col, row, w, h)
                    img = src.read(window=win).astype("float32") / 10000.0

                    if img.shape[1] < patch or img.shape[2] < patch:
                        pad = np.zeros((img.shape[0], patch, patch), dtype="float32")
                        pad[:, :h, :w] = img
                        img = pad

                    x = torch.tensor(img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        p = torch.sigmoid(model(x))[0,0].cpu().numpy()

                    p = p[:h, :w]
                    m = (p >= thresh).astype("uint8")

                    dp.write(p, 1, window=win)
                    dm.write(m, 1, window=win)

# ==================================================
# PUBLIC RUN FUNCTION (LIKE YOUR STYLE)
# ==================================================
def run(
    patch_img="data/patches/images",
    patch_msk="data/patches/masks",
    sentinel_root="data/sentinel",
    tiles=("T44PMV",),
    year=2025,
    out_dir="output/predictions",
    model_path="models/builtup_unet_m2.pth",
    epochs=25,
    batch_size=4,
    threshold=0.6,
):

    print("Using device:", DEVICE)

    print("\n--- STEP 1: TRAINING ---")
    model = train_model(patch_img, patch_msk, model_path, epochs, batch_size)

    print("\n--- STEP 2: PREDICTION ---")
    out_dir = Path(out_dir)

    for tile in tiles:
        stack = Path(sentinel_root)/tile/str(year)/"dl_stack"/"S2_DL_STACK.tif"

        out_tile = out_dir/tile/str(year)
        out_tile.mkdir(parents=True, exist_ok=True)

        predict_tile(
            model,
            stack_path=stack,
            out_prob=out_tile/"BUILTUP_PROB_DL.tif",
            out_mask=out_tile/"BUILTUP_MASK_DL.tif",
            patch=256,
            thresh=threshold
        )

    print("\n🎉 TRAINING + PREDICTION COMPLETED")

# ==================================================
# CLI SAFE
# ==================================================
if __name__ == "__main__":
    run()

# %%
