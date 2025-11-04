import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset


def build_seq_samples_for_play(gid, pid, df_in, df_out, L=11, dt=0.093, k_neighbors=12):
    """
    Build sequence sample for one play: past [N, L, 14], target [N, T, 2].
    Uses absolute coordinates (no centering or flipping).
    """
    try:
        inp = df_in[(df_in.game_id == gid) & (df_in.play_id == pid)].copy()
        out = df_out[(df_out.game_id == gid) & (df_out.play_id == pid)].copy()
        if inp.empty or out.empty:
            return None

        # Basic clipping
        for df in [inp, out]:
            if "y" in df.columns:
                df["y"] = df["y"].clip(0, 53.3)
            if "ball_land_y" in df.columns:
                df["ball_land_y"] = df["ball_land_y"].clip(0, 53.3)

        frames_in = np.sort(inp["frame_id"].unique())
        if len(frames_in) < L:
            return None
        release_f = frames_in.max()
        past_frames = frames_in[-L:]
        num_out = int(inp["num_frames_output"].iloc[0])

        common_ids = np.intersect1d(inp.nfl_id.unique(), out.nfl_id.unique())
        if len(common_ids) == 0:
            return None
        N = len(common_ids)

        # Approximate ball landing position (using passer position)
        passer = inp[(inp.frame_id == release_f) & (inp.player_role == "Passer")]
        if not passer.empty:
            bx, by = passer.iloc[0]["x"], passer.iloc[0]["y"]
        else:
            bx, by = inp[inp.frame_id == release_f][["x", "y"]].mean()

        chunks = []
        for f in past_frames:
            sl = inp[(inp.frame_id == f) & (inp.nfl_id.isin(common_ids))].set_index("nfl_id").reindex(common_ids)
            sl = sl.ffill().bfill()

            x, y = sl["x"].to_numpy(float), sl["y"].to_numpy(float)
            s = sl["s"].fillna(0).to_numpy(float)
            dgr = sl["dir"].fillna(0).to_numpy(float)
            vx, vy = s * np.cos(np.deg2rad(dgr)), s * np.sin(np.deg2rad(dgr))

            role_idx = sl["player_role"].fillna("Other").map(
                {"Targeted Receiver": 0, "Passer": 1, "Defensive Coverage": 2, "Other": 3}
            ).fillna(3).astype(int)
            role_oh = np.eye(4, dtype=float)[role_idx]

            dx_ball = bx - x
            dy_ball = by - y
            dist_ball = np.sqrt(dx_ball**2 + dy_ball**2)
            ang_ball = np.arctan2(dy_ball, dx_ball)
            sin_ball, cos_ball = np.sin(ang_ball), np.cos(ang_ball)

            cx, cy = np.nanmean(x), np.nanmean(y)
            dx_c, dy_c = x - cx, y - cy
            dist_c = np.sqrt(dx_c**2 + dy_c**2)
            ang_c = np.arctan2(dy_c, dx_c)
            sin_c, cos_c = np.sin(ang_c), np.cos(ang_c)

            frame_feat = np.column_stack([
                x, y, vx, vy, role_oh,
                dist_ball, sin_ball, cos_ball,
                dist_c, sin_c, cos_c
            ])
            chunks.append(frame_feat)

        past = np.stack(chunks, axis=1).astype(np.float32)

        Xgt = out.pivot(index="nfl_id", columns="frame_id", values="x").reindex(common_ids)
        Ygt = out.pivot(index="nfl_id", columns="frame_id", values="y").reindex(common_ids)
        x_gt = torch.tensor(Xgt.to_numpy(np.float32))
        y_gt = torch.tensor(Ygt.to_numpy(np.float32))
        valid = (~torch.isnan(x_gt)) & (~torch.isnan(y_gt))

        ball_ctx = torch.tensor([bx, by, dt, 0.0, 0.0], dtype=torch.float32)

        sl_rel = inp[inp.frame_id == release_f].set_index("nfl_id").reindex(common_ids)
        XY = sl_rel[["x", "y"]].to_numpy(float)
        n_nb = min(k_neighbors + 1, max(1, len(common_ids)))
        nbrs = NearestNeighbors(n_neighbors=n_nb, algorithm="kd_tree").fit(XY)
        idxs = nbrs.kneighbors(XY, return_distance=False)
        idxs = idxs[:, 1:] if idxs.shape[1] > 1 else np.zeros((N, 0), dtype=int)

        return {
            "nfl_ids": common_ids,
            "past": torch.tensor(past, dtype=torch.float32),
            "nbr_idx": torch.tensor(idxs, dtype=torch.long),
            "ball_land": ball_ctx,
            "target": torch.stack([x_gt, y_gt], dim=-1),
            "valid": valid,
            "release_frame": int(release_f),
            "T": int(num_out),
            "dt_eff": float(dt),
        }

    except Exception as e:
        print(f"⚠️ Error in play {gid}-{pid}: {e}")
        return None


class PlaysSeqDataset(Dataset):
    """Dataset wrapper for sequential player trajectories."""
    def __init__(self, keys, df_in, df_out, L=11, k_neighbors=12, dt=0.093):
        self.samples = []
        for gid, pid in keys:
            s = build_seq_samples_for_play(gid, pid, df_in, df_out, L=L, k_neighbors=k_neighbors, dt=dt)
            if s is not None:
                self.samples.append(s)
        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found for dataset.")

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, i): 
        return self.samples[i]
