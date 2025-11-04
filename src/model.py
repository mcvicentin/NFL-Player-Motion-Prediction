import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 🧱 MODEL COMPONENTS
# ============================================================

class PastEncoder(nn.Module):
    """Encodes past L-frame sequences for each player using GRU."""
    def __init__(self, in_dim=14, hid=256, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hid,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, past):
        out, h = self.gru(past)
        emb = self.dropout(h[-1])  # last GRU layer hidden state
        return emb


class GraphAttention(nn.Module):
    """
    Spatial attention among players with edge features (distance, sinθ, cosθ).
    """
    def __init__(self, hid=256, attn_dim=192, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(hid, attn_dim, bias=False)
        self.k = nn.Linear(hid, attn_dim, bias=False)
        self.v = nn.Linear(hid, hid, bias=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(3, attn_dim),
            nn.ReLU(),
            nn.Linear(attn_dim, attn_dim)
        )
        self.scale = 1.0 / math.sqrt(attn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb, nbr_idx, xy=None):
        """
        emb: [N, H]
        nbr_idx: [N, k]
        xy: [N, 2] (optional, player coordinates)
        """
        N, H = emb.size()
        Q = self.q(emb)
        K = self.k(emb)
        V = self.v(emb)

        K_nbr = K[nbr_idx]  # [N, k, D]
        V_nbr = V[nbr_idx]  # [N, k, H]

        # edge features (distance + sinθ + cosθ)
        edge_feat = torch.zeros(N, nbr_idx.size(1), 3, device=emb.device)
        if xy is not None:
            nbr_xy = xy[nbr_idx] - xy[:, None, :]
            dist = nbr_xy.norm(dim=-1, keepdim=True)
            ang = torch.atan2(nbr_xy[..., 1], nbr_xy[..., 0])
            sin_a, cos_a = torch.sin(ang).unsqueeze(-1), torch.cos(ang).unsqueeze(-1)
            edge_feat = torch.cat([dist, sin_a, cos_a], dim=-1)

        edge_emb = self.edge_mlp(edge_feat)
        scores = (Q.unsqueeze(1) * (K_nbr + edge_emb)).sum(-1) * self.scale
        w = F.softmax(scores, dim=1)
        w = self.dropout(w)

        agg = (w.unsqueeze(-1) * V_nbr).sum(1)
        return emb + agg


class Decoder(nn.Module):
    """Autoregressive GRU decoder for trajectory prediction."""
    def __init__(self, hid=256, ctx_dim=5):
        super().__init__()
        self.input_dim = 2 + ctx_dim
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=hid, batch_first=True)
        self.out = nn.Linear(hid, 2)

    def forward(self, init_h, T, ctx, scheduled_sampling_p=0.0, v_max=10.1415,
                dt=0.093, teacher=None, x0=None, y0=None):
        N, H = init_h.size()
        h = init_h.unsqueeze(0)
        pos = torch.stack([x0, y0], dim=-1)

        prev_delta = torch.zeros(N, 2, device=init_h.device)
        outs = []

        for t in range(T):
            dec_in = torch.cat([prev_delta, ctx], dim=1).unsqueeze(1)
            dec_out, h = self.gru(dec_in, h)
            delta = self.out(dec_out.squeeze(1))

            # limit max step by v_max * dt
            max_step = v_max * dt
            delta_norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-6)
            delta = delta * torch.clamp(max_step / delta_norm, max=1.0)

            pos = pos + delta
            outs.append(pos.unsqueeze(1))

            # scheduled sampling
            if (teacher is not None) and (scheduled_sampling_p > 0):
                use_teacher = (torch.rand(N, device=pos.device) < scheduled_sampling_p).float().unsqueeze(1)
                teacher_pos = teacher[:, t, :]
                teacher_delta = teacher_pos - (pos - delta)
                prev_delta = use_teacher * teacher_delta + (1 - use_teacher) * delta
            else:
                prev_delta = delta

        return torch.cat(outs, dim=1)  # [N, T, 2]


class SeqInterModel(nn.Module):
    """Full sequence + interaction model (Encoder → GraphAttention → Decoder)."""
    def __init__(self, in_dim=14, hid=256, attn_dim=192):
        super().__init__()
        self.enc = PastEncoder(in_dim=in_dim, hid=hid)
        self.gatt = GraphAttention(hid=hid, attn_dim=attn_dim)
        self.dec = Decoder(hid=hid, ctx_dim=5)

    def forward(self, past, nbr_idx, ball_ctx, T, x0, y0,
                scheduled_sampling_p=0.0, teacher=None, v_max=10.1415, dt=0.093):
        N = past.size(0)
        emb = self.enc(past)
        xy = past[:, -1, 0:2]
        emb = self.gatt(emb, nbr_idx, xy=xy)
        ctx = ball_ctx.unsqueeze(0).expand(N, -1)
        pred = self.dec(emb, T, ctx,
                        scheduled_sampling_p=scheduled_sampling_p,
                        teacher=teacher, v_max=v_max, dt=dt,
                        x0=x0, y0=y0)
        return pred
