# NFL-Player-Motion-Prediction
Deep learning model to predict NFL player trajectories using sequence + graph attention (Kaggle Big Data Bowl 2026).

Predict future **player trajectories** from tracking data using a **sequence encoder + graph attention + autoregressive decoder**.  
This repository contains clean, modular code for data loading, model training, hyperparameter search (Optuna), and evaluation.

---

## Highlights

- **Temporal encoder:** 2-layer GRU with dropout  
- **Spatial reasoning:** Graph Attention over nearest neighbors with distance & angle edge features  
- **Autoregressive decoder:** GRU with scheduled sampling and speed clipping  
- **Losses:** Huber + Smoothness (acceleration penalty) + Directional consistency  
- **Reproducibility:** Fixed seeds, Optuna study DB, saved configs & artifacts

---

## Repository Structure
```
nfl-player-motion-prediction/
│
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ .gitignore
│
├─ data/
│ └─ train/
│ ├─ week01_input.csv
│ ├─ week01_output.csv
│ └─ ...
│
├─ src/
│ ├─ init.py
│ ├─ dataset.py # build_seq_samples_for_play, PlaysSeqDataset
│ ├─ model.py # PastEncoder, GraphAttention, Decoder, SeqInterModel
│ ├─ losses.py # huber_loss, smoothness_loss, direction_loss, rmse_metric
│ ├─ utils.py # dir_to_v, make_dataloaders, helpers
│ ├─ train.py # training loop, Optuna objective, CLI
│ └─ predict.py # load model + generate predictions (optional)
│
├─ notebooks/
│ ├─ 01_exploratory_analysis.ipynb
│ ├─ 02_model_training.ipynb
│ └─ 03_results_visualization.ipynb
│
└─ models/
├─ best_model.pth
├─ optuna_results.csv
└─ optuna_study.db
```


> **Note:** Large files under `data/` and `models/` should not be committed. Use `.gitignore`, DVC, or Git LFS as needed.

---


## Setup

```bash
git clone https://github.com/<your-user>/nfl-player-motion-prediction.git
cd nfl-player-motion-prediction

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

## Requirements
```
python>=3.10
torch>=2.0.0
numpy
pandas
scikit-learn
matplotlib
tqdm
optuna
```
```
pip install -r requirements.txt
```


## Data
Download the competition data from Kaggle:

👉 [NFL Big Data Bowl 2026 – Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)

Then place the CSV files under:
```
data/train/
week01_input.csv
week01_output.csv
```

## Quick start (single train)

```
python src/train.py \
  --epochs 100 \
  --n_train 700 \
  --n_val 150 \
  --L 11 \
  --k_neighbors 12 \
  --hid 256 \
  --attn_dim 192 \
  --lr 5.9e-4 \
  --w_smooth 0.09
```

The script will:

- build datasets from ```data/train/``'
- train the SeqInterModel (encoder → graph attention → decoder)
- print training/validation losses & RMSE
- save the best weights and a JSON log in ```models/```


## Hyperparameter Search (Optuna)

Run Optuna to explore a search space of architectural and training parameters:

- ```models/optuna_results.csv``` — all trials summary
- ```models/*.pth & *.json``` — best checkpoints and metadata
- ```models/optuna_study.db``` — resumable Optuna study database


## Model Overview

- **PastEncoder (GRU)**: encodes a window **L** of per-player features

  - Input features per timestep (14):
      ```x, y, vx, vy, role_onehot(4), dist_to_ball, sinθ_ball, cosθ_ball, dist_to_center, sinθ_c, cosθ_c```

- **GraphAttention**: attends to k nearest neighbors using keys/values + an edge MLP over (distance, sinθ, cosθ)

Decoder (GRU): autoregressively predicts (x, y) for T future frames, with:

scheduled sampling (teacher forcing probability ramp-up)

speed clipping using v_max * dt to ensure physical plausibility

Losses

Huber on positions (role-weighted; higher for Passer/Targeted Receiver)

Smoothness on acceleration (reduces jitter)

Direction loss for angular coherence between consecutive velocities

Metric

RMSE over (x, y) on valid targets
