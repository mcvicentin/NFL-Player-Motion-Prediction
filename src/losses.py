import torch
import torch.nn.functional as F


def huber_loss(pred, target, valid, delta=1.0, role_weights=None):
    """Weighted Huber loss for (x, y) coordinates."""
    diff = pred - target
    absdiff = torch.sqrt((diff**2).sum(-1) + 1e-9)
    quad = 0.5 * (absdiff**2)
    lin = delta * (absdiff - 0.5 * delta)
    hub = torch.where(absdiff <= delta, quad, lin)
    mask = valid.float()

    if role_weights is not None:
        role_w = role_weights.unsqueeze(1).expand_as(mask)
        mask = mask * role_w

    return (hub * mask).sum() / (mask.sum() + 1e-9)


def smoothness_loss(pred, valid, dt=0.093, coef=0.1):
    """Penalize acceleration (smooth trajectories)."""
    vel = pred[:, 1:, :] - pred[:, :-1, :]
    vmask = valid[:, 1:] & valid[:, :-1]
    acc = vel[:, 1:, :] - vel[:, :-1, :]
    amask = vmask[:, 1:] & vmask[:, :-1]
    acc_norm = acc.norm(dim=-1)
    smooth = (acc_norm * amask.float()).sum() / (amask.float().sum() + 1e-9)
    return coef * smooth


def direction_loss(pred, valid, coef=0.05):
    """Penalize sharp directional changes."""
    vel = pred[:, 1:, :] - pred[:, :-1, :]
    vmask = valid[:, 1:] & valid[:, :-1]
    vel_norm = F.normalize(vel, dim=-1, eps=1e-6)
    cos_sim = (vel_norm[:, 1:, :] * vel_norm[:, :-1, :]).sum(-1)
    mask = vmask[:, 1:] & vmask[:, :-1]
    loss = (1.0 - cos_sim[mask]).mean() if mask.any() else torch.tensor(0.0, device=pred.device)
    return coef * loss


@torch.no_grad()
def rmse_metric(pred, target, valid):
    """Compute RMSE across valid coordinates."""
    diff = (pred - target)
    se = (diff[..., 0]**2 + diff[..., 1]**2) * valid.float()
    mse = se.sum() / (valid.float().sum() + 1e-9)
    return torch.sqrt(mse).item()
