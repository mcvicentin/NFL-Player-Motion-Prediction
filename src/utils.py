import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from .dataset import PlaysSeqDataset


def dir_to_v(s, dir_deg):
    """Convert speed and direction in degrees to velocity components (vx, vy)."""
    th = np.deg2rad(dir_deg)
    vx = s * np.cos(th)
    vy = s * np.sin(th)
    return vx, vy


def make_dataloaders(df_in, df_out, n_train=200, n_val=50, seed=42, L=11, k_neighbors=12, dt=0.093):
    """
    Split available plays into train/val and return DataLoaders.
    """
    keys_all = df_in[["game_id", "play_id"]].drop_duplicates().values.tolist()
    random.Random(seed).shuffle(keys_all)

    ds_tr = PlaysSeqDataset(keys_all[:n_train], df_in, df_out, L=L, k_neighbors=k_neighbors, dt=dt)
    ds_va = PlaysSeqDataset(keys_all[n_train:n_train + n_val], df_in, df_out, L=L, k_neighbors=k_neighbors, dt=dt)

    dl_tr = DataLoader(ds_tr, batch_size=1, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False)

    print(f"✅ Dataset ready | train plays: {len(ds_tr)} | val plays: {len(ds_va)}")
    return dl_tr, dl_va
