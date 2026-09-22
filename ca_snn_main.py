"""
Training and evaluation script for CorrelationAwareSNN (CA-SNN-HSI)
====================================================================
Drop-in replacement for batch_main.py; uses CorrelationAwareSNN instead
of MS2GCAN.  All dataset loading, preprocessing, and metric computation
are unchanged.
"""

import os
import time

# Ensure all relative paths (data files, checkpoints) resolve against the
# directory that contains this script, regardless of where Python is launched.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from CASNN import CorrelationAwareSNN
from preprocess import PatchDataset, get_location, loadData
from utils import AA_fn, kappa_fn, confusion_matrix, loss_fn

# ── Device ─────────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device)

# ── Dataset selection ───────────────────────────────────────────────────────
dataset_list = ["PU_normal", "PU", "HU", "WHLK"]
dataset_name = dataset_list[0]   # <-- change index to switch dataset

# ── Per-dataset hyperparameters ─────────────────────────────────────────────
CONFIGS = {
    "PU": dict(
        data_size=19, T=3, n_stages=4, gcn_layers=2, K_hop=2,
        hidden=32, batch_size=64,
    ),
    "PU_normal": dict(
        data_size=19, T=3, n_stages=4, gcn_layers=2, K_hop=2,
        hidden=32, batch_size=64,
    ),
    "HU": dict(
        data_size=13, T=3, n_stages=4, gcn_layers=2, K_hop=1,
        hidden=96, batch_size=32,
    ),
    "WHLK": dict(
        data_size=15, T=3, n_stages=4, gcn_layers=2, K_hop=1,
        hidden=48, batch_size=32,
    ),
}

cfg        = CONFIGS[dataset_name]
data_size  = cfg["data_size"]
T          = cfg["T"]
n_stages   = cfg["n_stages"]
gcn_layers = cfg["gcn_layers"]
K_hop      = cfg["K_hop"]
hidden     = cfg["hidden"]
batch_size = cfg["batch_size"]

epochs    = 200
lr        = 1e-4
test_only = False

# ── Data loading ─────────────────────────────────────────────────────────────
data, labels_TE, labels_TR, class_num = loadData(dataset_name)
H, W, S = data.shape
input_dim = S

data = np.reshape(data, [H * W, -1])
scaler = preprocessing.StandardScaler()
data = scaler.fit_transform(data)
data = np.reshape(data, [H, W, -1])

train_loc, train_lbl = get_location(labels_TR)
test_loc,  test_lbl  = get_location(labels_TE)

train_dataset = PatchDataset(data, train_loc, train_lbl, data_size)


def collate_fn(batch):
    patches, labels = zip(*batch)
    patches = torch.stack(patches, 0)          # [B, L², S]
    labels  = torch.tensor(labels, dtype=torch.float32)
    return patches, labels


# ── Model ─────────────────────────────────────────────────────────────────────
seed_list   = [0]
all_results = []

for seed_idx, seed in enumerate(seed_list):
    print(f"\n{'='*60}")
    print(f"Run {seed_idx + 1}/{len(seed_list)}  –  seed {seed}")
    print(f"{'='*60}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn,
    )

    model = CorrelationAwareSNN(
        T=T,
        img_size=data_size,
        num_cls=class_num,
        input_dim=input_dim,
        hidden=hidden,
        n_stages=n_stages,
        gcn_layers=gcn_layers,
        K_hop=K_hop,
        use_cupy=False,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── Training ──────────────────────────────────────────────────────────────
    if not test_only:
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs, eta_min=lr * 0.01
        )
        best_loss = float("inf")

        for epoch in range(epochs):
            model.train()
            t0          = time.time()
            total_loss  = 0.0
            corrects    = np.zeros(class_num)
            totals      = np.zeros(class_num)

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimiser.zero_grad()
                out  = model(batch_x)
                loss = loss_fn(out, batch_y, class_num, device)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
                c, t = AA_fn(out, batch_y)
                corrects += c
                totals   += t

            oa      = corrects.sum() / max(totals.sum(), 1)
            scheduler.step()
            cur_lr  = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1:3d}/{epochs}  "
                f"Loss {total_loss:.4f}  OA {oa:.4f}  "
                f"LR {cur_lr:.6f}  ET {time.time()-t0:.1f}s"
            )

            if total_loss < best_loss:
                best_loss = total_loss
                ckpt = f"./best_{type(model).__name__}_{dataset_name}_weights.pth"
                torch.save(model.state_dict(), ckpt)
                print("  → saved best model")

    # ── Testing ───────────────────────────────────────────────────────────────
    print("\nTesting …")
    test_dataset = PatchDataset(data, test_loc, test_lbl, data_size)
    test_loader  = DataLoader(
        test_dataset, batch_size=batch_size * 16,
        shuffle=False, collate_fn=collate_fn,
    )

    ckpt = f"./best_{type(model).__name__}_{dataset_name}_weights.pth"
    sd   = torch.load(ckpt, map_location=device)
    sd   = {k: v for k, v in sd.items()
            if "total_ops" not in k and "total_params" not in k}
    model.load_state_dict(sd, strict=False)
    model.eval()

    corrects    = np.zeros(class_num)
    totals      = np.zeros(class_num)
    outputs_all = []
    labels_all  = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Test"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out      = model(batch_x)
            c, t     = AA_fn(out, batch_y)
            corrects += c
            totals   += t
            outputs_all.append(out.cpu().numpy())
            labels_all.append(batch_y.cpu().numpy())

    outputs_np = np.concatenate(outputs_all, 0)
    labels_np  = np.concatenate(labels_all,  0)

    OA        = corrects.sum() / totals.sum()
    acc_class = corrects / totals
    AA        = acc_class.mean()
    kappa     = kappa_fn(outputs_np, labels_np)
    cm        = confusion_matrix(labels_np, np.argmax(outputs_np, 1))

    precision = np.diag(cm) / (np.sum(cm, axis=0) + 1e-8)
    recall    = np.diag(cm) / (np.sum(cm, axis=1) + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\nOA {OA:.4f}  AA {AA:.4f}  Kappa {kappa:.4f}")
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print(f"  Class {i+1:2d}  acc {acc_class[i]:.4f}  "
              f"P {p:.4f}  R {r:.4f}  F1 {f:.4f}")

    all_results.append(
        dict(seed=seed, OA=OA, AA=AA, kappa=kappa,
             acc_class=acc_class, cm=cm,
             precision=precision, recall=recall, f1=f1)
    )

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
OA_arr    = np.array([r["OA"]    for r in all_results])
AA_arr    = np.array([r["AA"]    for r in all_results])
kappa_arr = np.array([r["kappa"] for r in all_results])
print(f"OA    : {OA_arr.mean():.6f} ± {OA_arr.std():.6f}")
print(f"AA    : {AA_arr.mean():.6f} ± {AA_arr.std():.6f}")
print(f"Kappa : {kappa_arr.mean():.6f} ± {kappa_arr.std():.6f}")
print(f"{'='*60}")
