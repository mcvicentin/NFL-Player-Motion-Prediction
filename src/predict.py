import torch
import pandas as pd
from .model import SeqInterModel
from .dataset import build_seq_samples_for_play


def predict_play(model_path, df_in, df_out, gid, pid, device="cuda"):
    """
    Generate trajectory predictions for a single play given a trained model.
    """
    sample = build_seq_samples_for_play(gid, pid, df_in, df_out)
    if sample is None:
        print(f"⚠️ Play {gid}-{pid} not found or invalid.")
        return None

    model = SeqInterModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    past = sample["past"].to(device)
    nbr_idx = sample["nbr_idx"].to(device)
    ball_ctx = sample["ball_land"].to(device)
    x0, y0 = past[:, -1, 0], past[:, -1, 1]
    T = sample["T"]
    dt = sample["dt_eff"]

    with torch.no_grad():
        pred = model(past, nbr_idx, ball_ctx, T, x0, y0, v_max=10.14, dt=dt)

    preds = pred.cpu().numpy()
    nfl_ids = sample["nfl_ids"]
    frames = range(T)

    df_pred = pd.DataFrame(
        [(gid, pid, int(nfl_ids[i]), int(f), preds[i, f, 0], preds[i, f, 1])
         for i in range(len(nfl_ids)) for f in frames],
        columns=["game_id", "play_id", "nfl_id", "frame_id", "x_pred", "y_pred"]
    )

    print(f"✅ Predictions generated for play {gid}-{pid}")
    return df_pred


if __name__ == "__main__":
    # Example usage (replace with your paths)
    df_in = pd.read_csv("data/train/week01_input.csv")
    df_out = pd.read_csv("data/train/week01_output.csv")

    model_path = "models/best_model.pth"
    gid, pid = 2021090900, 75  # example play IDs

    preds = predict_play(model_path, df_in, df_out, gid, pid)
    preds.to_csv("predictions_play.csv", index=False)
    print("💾 Saved predictions to predictions_play.csv")
