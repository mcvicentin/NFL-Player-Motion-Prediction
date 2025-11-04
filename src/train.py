import gc
import json
import torch
import optuna
import random
import numpy as np
from .utils import make_dataloaders
from .model import SeqInterModel
from .losses import huber_loss, smoothness_loss, direction_loss, rmse_metric


# ============================================================
# 🔁 TRAINING LOOP
# ============================================================

def train_seq_model(df_in, df_out, n_train=700, n_val=150, L=11, k_neighbors=12, dt=0.093,
                    hid=256, attn_dim=192, epochs=100, lr=5.8e-4, w_smooth=0.09,
                    v_max=10.14, seed=42):
    """Main training loop for the sequence interaction model."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dl_tr, dl_va = make_dataloaders(df_in, df_out, n_train, n_val, seed, L, k_neighbors, dt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SeqInterModel(in_dim=14, hid=hid, attn_dim=attn_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=False)

    best_va = (1e9, None)
    best_ep = 0
    patience, wait = 10, 0

    for ep in range(1, epochs + 1):
        tr_loss, tr_rmse = _run_epoch(model, dl_tr, opt, device, dt, v_max, w_smooth, epoch=ep, train=True, epochs=epochs)
        va_loss, va_rmse = _run_epoch(model, dl_va, opt, device, dt, v_max, w_smooth, epoch=ep, train=False, epochs=epochs)
        sched.step(va_loss)

        print(f"Epoch {ep:02d} | train loss={tr_loss:.3f} rmse={tr_rmse:.3f} | val loss={va_loss:.3f} rmse={va_rmse:.3f}")

        if va_rmse < best_va[0] - 1e-3:
            best_va = (va_rmse, model.state_dict())
            best_ep = ep
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            print(f"⏹️ Early stopping at epoch {ep} (best={best_va[0]:.3f} @ {best_ep})")
            break

    return model, best_va[0]


def _run_epoch(model, loader, opt, device, dt, v_max, w_smooth, epoch, train=True, epochs=100):
    """Run one epoch of training or validation."""
    model.train() if train else model.eval()
    total_loss, total_rmse, cnt = 0, 0, 0

    scheduled_sampling_max = 0.5
    ss_p = min(scheduled_sampling_max, epoch / float(max(1, epochs // 2)) * scheduled_sampling_max) if train else 0.0

    for s in loader:
        past = s["past"].squeeze(0).to(device)
        if train:
            past = past + torch.randn_like(past) * 0.05  # data augmentation
        nbr_idx = s["nbr_idx"].squeeze(0).to(device)
        ball_ctx = s["ball_land"].squeeze(0).to(device)
        target = s["target"].squeeze(0).to(device)
        valid = s["valid"].squeeze(0).to(device)
        x0, y0 = past[:, -1, 0], past[:, -1, 1]
        T = target.size(1)

        roles = past[:, -1, 4:8].argmax(dim=1)
        weights = torch.tensor([1.5 if r in [0, 1] else 1.0 for r in roles], device=device)

        with torch.set_grad_enabled(train):
            pred = model(past, nbr_idx, ball_ctx, T, x0, y0,
                         scheduled_sampling_p=ss_p, teacher=target,
                         v_max=v_max, dt=dt)
            loss = (huber_loss(pred, target, valid, delta=0.7, role_weights=weights)
                    + smoothness_loss(pred, valid, coef=w_smooth)
                    + direction_loss(pred, valid, coef=0.05))

            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
                opt.step()

        rmse = rmse_metric(pred, target, valid)
        total_loss += loss.item()
        total_rmse += rmse
        cnt += 1

    return total_loss / cnt, total_rmse / cnt


# ============================================================
# 🎯 OPTUNA OBJECTIVE
# ============================================================

def objective(trial, df_in, df_out, n_train, n_val, common_keys):
    params = {
        "hid": trial.suggest_categorical("hid", [192, 256, 320]),
        "attn_dim": trial.suggest_categorical("attn_dim", [128, 192, 256]),
        "L": trial.suggest_categorical("L", [9, 11, 13]),
        "k_neighbors": trial.suggest_categorical("k_neighbors", [4, 8, 12, 16]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "w_smooth": trial.suggest_float("w_smooth", 0.03, 0.15, log=True),
        "seed": trial.suggest_int("seed", 0, 250),
    }

    tag = f"L{params['L']}_nei{params['k_neighbors']}_hid{params['hid']}_attn{params['attn_dim']}_lr{params['lr']:.1e}_ws{params['w_smooth']:.3f}"
    print(f"\n🎯 Trial {trial.number}: {tag}")

    try:
        model, best_rmse = train_seq_model(
            df_in, df_out,
            n_train=n_train, n_val=n_val, L=params["L"], k_neighbors=params["k_neighbors"],
            hid=params["hid"], attn_dim=params["attn_dim"],
            lr=params["lr"], w_smooth=params["w_smooth"], seed=params["seed"]
        )

        result = {
            "tag": tag,
            "params": params,
            "val_rmse": float(best_rmse)
        }
        with open(f"models/{tag}.json", "w") as f:
            json.dump(result, f, indent=2)
        torch.save(model.state_dict(), f"models/{tag}.pth")

        del model
        torch.cuda.empty_cache()
        gc.collect()

        return best_rmse

    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return float("inf")
