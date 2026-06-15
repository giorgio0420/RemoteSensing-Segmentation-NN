# -*- coding: utf-8 -*-
"""
TASK 2 (secondaria) - Multi-modal / multi-spectral su DFC2020 (Sentinel-1 + Sentinel-2).

Legge i .tif direttamente dallo zip HF (niente libreria `datasets`). Etichette rimappate a 8 classi
(+255 ignore); con --binary -> acqua(1) vs terra(0).

Modelli:
  --model resnet  : U-Net ResNet-34 (smp), in_channels variabile, pretraining ImageNet vs scratch
  --model satmae  : SatMAE++ fMoW-Sentinel (ViT-L group-channel, congelato) + decoder  [richiede --ckpt]

Confronti chiave (consegna):
  con/senza pesi SatMAE++   ->  --model satmae  con/ senza --ckpt
  con/senza radar           ->  --bands msi  vs  msi_sar
  data-scarce / pretraining ->  --scratch (solo resnet)
"""
import argparse, csv, io, os, random, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

S2_MEAN = np.array([1370.19,1184.38,1120.77,1136.26,1263.74,1645.40,1846.87,1762.60,1972.62,582.73,14.77,1732.16,1247.92], dtype=np.float32)
S2_STD  = np.array([633.15,650.28,712.13,965.23,948.98,1108.07,1258.36,1233.15,1364.39,472.38,14.31,1310.37,1087.60], dtype=np.float32)
S1_MEAN = np.array([-12.55,-20.19], dtype=np.float32)
S1_STD  = np.array([5.26,5.91], dtype=np.float32)

RGB_IDX = [3, 2, 1]
MSI_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]       # 10 bande = set SatMAE-Sentinel [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]
DFC_MAP = np.array([255, 0,0,0,0,0, 1,1, 255,255, 2, 3, 4, 5, 4, 255, 6, 7], dtype=np.int64)
IGNORE = 255
SEED = 42


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def prepare_data():
    """Scarica DFC2020.zip ed estrae con 'unzip' di sistema (Python 3.12 rifiuta lo zip per
    'overlapped entries'). Legge i tif da disco."""
    from huggingface_hub import hf_hub_download
    import subprocess
    zp = hf_hub_download("GFM-Bench/DFC2020", "data/DFC2020.zip", repo_type="dataset")
    out = "/content/dfc2020"
    ddir = os.path.join(out, "DFC2020")
    marker = os.path.join(out, ".extracted_ok")
    if not os.path.exists(marker):
        os.makedirs(out, exist_ok=True)
        print("Estraggo con 'unzip' di sistema (bypassa il check zip-bomb di Python 3.12)...")
        subprocess.run(["unzip", "-q", "-o", zp, "-d", out], check=True)
        open(marker, "w").close()
    df = pd.read_csv(os.path.join(ddir, "metadata.csv"))
    return ddir, df


class DFC2020(Dataset):
    def __init__(self, data_dir, df_split, mode="msi", subset=None, seed=SEED, binary=False):
        self.dir = data_dir
        self.df = df_split.reset_index(drop=True)
        self.mode = mode; self.binary = binary
        self.bands = RGB_IDX if mode == "rgb" else MSI_IDX
        idx = list(range(len(self.df)))
        if subset and subset < len(idx):
            random.Random(seed).shuffle(idx); idx = idx[:subset]
        self.idx = idx

    def _read(self, rel):
        import tifffile
        im = tifffile.imread(os.path.join(self.dir, rel))
        if im.ndim == 2:
            im = im[..., None]
        return np.transpose(im, (2, 0, 1))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        row = self.df.iloc[self.idx[i]]
        opt = self._read(row.optical_path).astype(np.float32)
        opt = (opt - S2_MEAN[:, None, None]) / S2_STD[:, None, None]
        x = opt[self.bands]
        if self.mode == "msi_sar":
            rad = self._read(row.radar_path).astype(np.float32)
            rad = (rad - S1_MEAN[:, None, None]) / S1_STD[:, None, None]
            x = np.concatenate([x, rad], axis=0)
        lab = self._read(row.label_path)[0].astype(np.int64)
        lab = DFC_MAP[lab]
        if self.binary:
            lab = np.where(lab == IGNORE, IGNORE, (lab == 7).astype(np.int64))   # 1=acqua, 0=terra
        return torch.from_numpy(np.ascontiguousarray(x)).float(), torch.from_numpy(lab).long()


def in_channels_for(mode):
    return {"rgb": 3, "msi": 10, "msi_sar": 12}[mode]


def build_model(a, n_classes):
    if a.model == "satmae":
        from satmae_sentinel import SatMAESentinelSeg
        return SatMAESentinelSeg(num_classes=n_classes, ckpt_path=a.ckpt,
                                 ft_blocks=a.ft_blocks, use_sar=(a.bands == "msi_sar"))
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name="resnet34",
                    encoder_weights="imagenet" if not a.scratch else None,
                    in_channels=in_channels_for(a.bands), classes=n_classes)


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    inter = torch.zeros(n_classes, device=device); union = torch.zeros(n_classes, device=device)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            pred = model(x).argmax(1)
        valid = y != IGNORE
        correct += ((pred == y) & valid).sum().item(); total += valid.sum().item()
        for c in range(n_classes):
            pc = (pred == c) & valid; yc = (y == c) & valid
            inter[c] += (pc & yc).sum(); union[c] += (pc | yc).sum()
    iou = inter / (union + 1e-6); v = union > 0
    return correct / max(total, 1), (iou[v].mean().item() if v.any() else 0.0), iou.cpu().numpy()


def compute_class_weights(ds, n_classes, sample=400):
    """Median-frequency balancing da un campione di label (per le classi rare)."""
    counts = np.zeros(n_classes, dtype=np.float64)
    for j in ds.idx[:sample]:
        row = ds.df.iloc[j]
        lab = DFC_MAP[ds._read(row.label_path)[0].astype(np.int64)]
        if ds.binary:
            lab = np.where(lab == IGNORE, IGNORE, (lab == 7).astype(np.int64))
        for c in range(n_classes):
            counts[c] += (lab == c).sum()
    freq = counts / max(counts.sum(), 1.0)
    med = np.median(freq[freq > 0]) if (freq > 0).any() else 1.0
    w = np.clip(np.where(freq > 0, med / np.maximum(freq, 1e-6), 1.0), 0.0, 10.0)  # cap per stabilita'
    print(f"class-weights: {np.round(w, 2).tolist()}")
    return torch.tensor(w, dtype=torch.float32)


def main(a):
    set_seed()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nc = 2 if a.binary else 8
    if a.model == "satmae" and a.bands == "rgb":
        a.bands = "msi"          # SatMAE-Sentinel vuole le 10 bande
    ddir, df = prepare_data()
    tr = DFC2020(ddir, df[df.split == "train"], a.bands, subset=a.subset, binary=a.binary)
    va = DFC2020(ddir, df[df.split == "val"], a.bands, subset=a.val_subset, binary=a.binary)
    print(f"model={a.model} | bands={a.bands} | binary={a.binary} | classes={nc} | "
          f"train {len(tr)} | val {len(va)} | dev={dev}")
    tl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(va, batch_size=a.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    import segmentation_models_pytorch as smp
    model = build_model(a, nc).to(dev)
    cw = compute_class_weights(tr, nc).to(dev) if a.class_weights else None
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE, weight=cw)
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE)
    crit = lambda p, t: ce(p, t) + dice(p, t)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=a.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    best = -1.0; best_iou = None; best_state = None
    for ep in range(a.epochs):
        model.train(); run = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        acc, miou, iou = evaluate(model, vl, dev, nc)
        if miou > best:
            best = miou; best_iou = iou
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"ep {ep+1}/{a.epochs} | loss {run/len(tl):.3f} | acc {acc*100:.1f}% | mIoU {miou:.4f}"
              + ("  <- best" if miou == best else ""))
    print(f"   best mIoU={best:.4f} | IoU/classe={np.round(best_iou,3).tolist()}")
    if best_state is not None:
        torch.save(best_state, f"best_{a.tag}.pt")     # per le figure input/output

    new = not os.path.exists("results_task2.csv")
    with open("results_task2.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "model", "bands", "binary", "epochs", "best_mIoU"])
        w.writerow([a.tag, a.model, a.bands, a.binary, a.epochs, f"{best:.4f}"])
    print(f"OK '{a.tag}' -> results_task2.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["resnet", "satmae"], default="resnet")
    p.add_argument("--ckpt", default="", help="checkpoint SatMAE-Sentinel (per --model satmae)")
    p.add_argument("--ft-blocks", dest="ft_blocks", type=int, default=0, help="satmae: scongela ultimi N blocchi ViT")
    p.add_argument("--bands", choices=["rgb", "msi", "msi_sar"], default="msi")
    p.add_argument("--binary", action="store_true", help="acqua vs terra (2 classi)")
    p.add_argument("--class-weights", dest="class_weights", action="store_true", help="median-frequency sulle classi rare")
    p.add_argument("--scratch", action="store_true", help="resnet: encoder random invece di ImageNet")
    p.add_argument("--subset", type=int, default=2000)
    p.add_argument("--val-subset", dest="val_subset", type=int, default=800)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tag", default="t2")
    main(p.parse_args())
