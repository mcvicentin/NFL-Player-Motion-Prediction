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

