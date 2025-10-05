# multi_listener_full.py
"""
Full refactor: Multi-listener spatio-temporal GAT-based model
- Self-contained Graph Attention implementation (no external GNN deps)
- Speaker & Listener encoders
- Spatio-temporal GNN combining GAT (spatial) + GRU (temporal)
- Listener decoders
- Demo + light training loop on dummy data
"""

import math
import random
import os
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Repro
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# ---------------------------
# Graph Attention Layer (single-head)
# Implementation based on the original GAT (Velickovic et al.)
# ---------------------------
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat  # whether to concat or average heads
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        """
        h: [B, N, in_features]
        adj: [B, N, N]    (soft adjacency weights, could be learned)
        returns: [B, N, out_features]
        """
        B, N, _ = h.shape
        Wh = torch.matmul(h, self.W)  # [B, N, out_features]

        # prepare attention mechanism
        Wh1 = Wh.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, out_features]
        Wh2 = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, out_features]
        cat = torch.cat([Wh1, Wh2], dim=-1)  # [B, N, N, 2*out]
        e = self.leakyrelu(torch.matmul(cat, self.a).squeeze(-1))  # [B, N, N]

        # incorporate provided adjacency as bias (if adj is not None)
        if adj is not None:
            # adj likely in [0,1] range — we add log(adj+eps) as bias to allow attention to consider prior
            # but to keep stability, we normalize by softmax later with mask
            # We'll mask absent edges by setting their e to -inf
            # For dense adj where all entries >0, we simply add small bias
            eps = 1e-6
            e = e + torch.log(adj + eps)

        # attention coefficients: softmax over neighbors
        alpha = F.softmax(e, dim=-1)  # [B, N, N]
        alpha = self.dropout(alpha)

        # linear combination of neighbor features
        h_prime = torch.matmul(alpha, Wh)  # [B, N, out_features]
        return h_prime

# Multi-head wrapper
class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, alpha=0.2, concat=True):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_dim, self.head_dim, dropout=dropout, alpha=alpha, concat=concat)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(out_dim, out_dim) if concat else None
        self.final_ln = nn.LayerNorm(out_dim)

    def forward(self, h, adj):
        # h: [B, N, in_dim]
        outs = [head(h, adj) for head in self.heads]  # list of [B,N,head_dim]
        h_out = torch.cat(outs, dim=-1)  # [B, N, num_heads * head_dim = out_dim]
        if self.out_proj is not None:
            h_out = self.out_proj(h_out)
        h_out = self.final_ln(h_out)
        return h_out  # [B, N, out_dim]


# ---------------------------
# Encoders & Decoders
# ---------------------------
class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim=181, d_model=256, nhead=8, depth=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.agg = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, 181]
        h = self.proj(x)  # [B,T,d]
        h = self.transformer(h)  # [B,T,d]
        attn = F.softmax(self.agg(h).squeeze(-1), dim=1)  # [B,T]
        ctx = torch.sum(h * attn.unsqueeze(-1), dim=1)  # [B,d]
        return self.ln(ctx)


class ListenerEncoder(nn.Module):
    def __init__(self, input_dim=181, d_model=256, num_layers=1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, 181]
        h = self.proj(x)
        out, hn = self.gru(h)
        emb = hn[-1]  # [B,d]
        return self.ln(emb)


class ListenerDecoder(nn.Module):
    def __init__(self, d_model=256, out_dim=181, pred_len=1):
        super().__init__()
        self.pred_len = pred_len
        if pred_len == 1:
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, out_dim))
        else:
            self.rnn = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True)
            self.proj = nn.Linear(d_model, out_dim)

    def forward(self, node_feat):
        # node_feat: [B, d]
        B, d = node_feat.shape
        if self.pred_len == 1:
            out = self.head(node_feat).view(B, 1, -1)  # [B,1,out_dim]
            return out
        else:
            # autoregressive decode
            inp = node_feat.unsqueeze(1)  # [B,1,d]
            hidden = (node_feat.unsqueeze(0).repeat(2,1,1), torch.zeros(2,B,d,device=node_feat.device))
            outs = []
            for _ in range(self.pred_len):
                o, hidden = self.rnn(inp, hidden)
                p = self.proj(o)
                outs.append(p)
                inp = o
            return torch.cat(outs, dim=1)  # [B,pred_len,out_dim]


# ---------------------------
# Spatio-Temporal GNN (uses GAT spatially + GRU temporally)
# Full refactor: clear interface, uses GAT wrapper above
# ---------------------------
class SpatioTemporalGNN(nn.Module):
    def __init__(self, d_model=256, num_nodes=4, gat_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        # GAT for spatial message passing
        self.gat = GAT(in_dim=d_model, out_dim=d_model, num_heads=gat_heads, dropout=dropout)
        # a small MLP to produce adjacency bias from speaker context (B,d_model)->(B,N,N)
        self.adj_from_speaker = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                              nn.Linear(d_model, num_nodes * num_nodes))
        # GRU for temporal update per node (keeps hidden state across timesteps)
        self.temporal_gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, node_feats, speaker_ctx, prev_states=None):
        """
        node_feats: [B, N, d_model]  -- current node features (speaker + listeners)
        speaker_ctx: [B, d_model] -- from speaker encoder
        prev_states: [B, N, d_model] or None -- temporal hidden states
        returns:
            updated_nodes: [B, N, d_model]
            new_states: [B, N, d_model]
        """
        B, N, d = node_feats.shape
        assert N == self.num_nodes and d == self.d_model

        # produce adjacency bias per batch
        adj_bias = self.adj_from_speaker(speaker_ctx).view(B, N, N)  # [B,N,N]
        # convert to softened adjacency weights in (0,1)
        adj = torch.sigmoid(adj_bias)  # learnable soft adjacency

        # spatial message passing via GAT (GAT uses adj as prior bias)
        spatial_out = self.gat(node_feats, adj)  # [B,N,d]

        # combine with residual
        updated = self.ln(node_feats + spatial_out)  # [B,N,d]

        # Temporal GRU: we treat each node as a sequence element with seq_len=1 and carry hidden state
        if prev_states is None:
            prev_states = torch.zeros(B, N, d, device=node_feats.device)
        inp = updated.view(B * N, 1, d)  # [B*N,1,d]
        h0 = prev_states.view(B * N, d).unsqueeze(0).contiguous()  # [1, B*N, d]
        out, h_new = self.temporal_gru(inp, h0)  # out [B*N,1,d]; h_new [1,B*N,d]
        out = out.view(B, N, d)
        new_states = h_new.view(1, B, N, d).squeeze(0).contiguous()  # [B,N,d]

        final = self.ln(out + updated)  # [B,N,d]
        return final, new_states


# ---------------------------
# Full System
# ---------------------------
class MultiListenerSystem(nn.Module):
    def __init__(self, input_dim=181, d_model=256, num_listeners=3, pred_len=1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_listeners = num_listeners
        self.num_nodes = num_listeners + 1
        self.pred_len = pred_len

        self.spk_enc = SpeakerEncoder(input_dim, d_model)
        self.lis_enc = ListenerEncoder(input_dim, d_model)
        self.stgnn = SpatioTemporalGNN(d_model, num_nodes=self.num_nodes, gat_heads=4)
        self.decoders = nn.ModuleList([ListenerDecoder(d_model, input_dim, pred_len) for _ in range(num_listeners)])

        # buffer for temporal states; will be resized per batch
        self.register_buffer('temporal_states', torch.zeros(1, self.num_nodes, d_model))

    def forward(self, speaker_seq, listeners_seq, reset_temporal=False):
        """
        speaker_seq: [B, T, input_dim]
        listeners_seq: [B, T, num_listeners, input_dim]
        returns dict
        """
        B = speaker_seq.shape[0]
        device = speaker_seq.device
        if reset_temporal or self.temporal_states.shape[0] != B:
            self.temporal_states = torch.zeros(B, self.num_nodes, self.d_model, device=device)

        # encode speaker -> context (B,d)
        sp_ctx = self.spk_enc(speaker_seq)

        # encode listeners -> get per-listener embedding (B,d)
        lis_embs = []
        for i in range(self.num_listeners):
            emb = self.lis_enc(listeners_seq[:, :, i, :])  # [B,d]
            lis_embs.append(emb)

        # build nodes [B,N,d] (node 0 = speaker context; nodes 1..N = listeners)
        nodes = torch.stack([sp_ctx] + lis_embs, dim=1)  # [B,N,d]

        # pass through SpatioTemporalGNN
        updated_nodes, new_states = self.stgnn(nodes, sp_ctx, self.temporal_states)
        # detach states to avoid backprop through batches
        self.temporal_states = new_states.detach()

        # decode per listener (skip node 0 which is speaker)
        preds = []
        for i in range(self.num_listeners):
            feat = updated_nodes[:, 1 + i, :]  # [B,d]
            p = self.decoders[i](feat)  # [B, pred_len, input_dim]
            preds.append(p)
        # stack to [B, pred_len, num_listeners, input_dim]
        preds = torch.stack(preds, dim=2)

        return {
            'speaker_context': sp_ctx,
            'listener_embeddings': torch.stack(lis_embs, dim=1),
            'node_features': updated_nodes,
            'listener_predictions': preds,
            'temporal_states': self.temporal_states
        }


# ---------------------------
# Dummy dataset utilities (replace with a real Dataset)
# ---------------------------
# class DummyDataset(torch.utils.data.Dataset):
#     def __init__(self, n_samples=200, seq_len=12, num_listeners=3, input_dim=181):
#         self.n = n_samples
#         self.seq_len = seq_len
#         self.num_listeners = num_listeners
#         self.input_dim = input_dim

#     def __len__(self): return self.n

#     def __getitem__(self, idx):
#         sp = torch.randn(self.seq_len, self.input_dim)
#         lis = torch.randn(self.seq_len, self.num_listeners, self.input_dim)
#         # ground truth: next-frame for each listener (pred_len=1)
#         gt = [torch.randn(1, self.input_dim) for _ in range(self.num_listeners)]
#         return sp, lis, gt


# # collate fn
# def collate_batch(batch):
#     sps, liss, gts = zip(*batch)
#     sps = torch.stack(sps, dim=0)  # [B,T,input_dim]
#     liss = torch.stack(liss, dim=0)  # [B,T,num_listeners,input_dim]
#     # gts: list of lists; restructure to list per listener
#     # Here we keep gts as list-of-lists
#     return sps, liss, gts
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=200, seq_len=12, num_listeners=3, input_dim=181, pred_len=1):
        self.n = n_samples
        self.seq_len = seq_len
        self.num_listeners = num_listeners
        self.input_dim = input_dim
        self.pred_len = pred_len

    def __len__(self): return self.n

    def __getitem__(self, idx):
        sp = torch.randn(self.seq_len, self.input_dim)
        lis = torch.randn(self.seq_len, self.num_listeners, self.input_dim)
        # ground truth: next pred_len frames for each listener => shape [pred_len, num_listeners, input_dim]
        gt = torch.randn(self.pred_len, self.num_listeners, self.input_dim)
        return sp, lis, gt

def collate_batch(batch):
    sps, liss, gts = zip(*batch)
    sps = torch.stack(sps, dim=0)              # [B,T,input_dim]
    liss = torch.stack(liss, dim=0)            # [B,T,num_listeners,input_dim]
    gts = torch.stack(gts, dim=0)              # [B,pred_len,num_listeners,input_dim]
    return sps, liss, gts


# ---------------------------
# Demo & training
# ---------------------------
def demo(device='cpu'):
    print("Demo starting on", device)
    B, T = 4, 12
    N_list = 3
    model = MultiListenerSystem(input_dim=181, d_model=256, num_listeners=N_list, pred_len=1).to(device)
    model.eval()

    # dummy inputs
    sp = torch.randn(B, T, 181, device=device)
    lis = torch.randn(B, T, N_list, 181, device=device)
    with torch.no_grad():
        out = model(sp, lis, reset_temporal=True)
    print("Shapes:")
    print(" speaker_context:", out['speaker_context'].shape)       # [B,d]
    print(" listener_embeddings:", out['listener_embeddings'].shape)  # [B,N_list,d]
    print(" node_features:", out['node_features'].shape)         # [B,N,d]
    print(" listener_predictions:", out['listener_predictions'].shape)  # [B, pred_len, N_list, 181] 

    # toy loss
    crit = nn.MSELoss()
    total_loss = 0.0
    for i in range(N_list):
        pred = out['listener_predictions'][:, :, i, :]  # [B,1,181]
        gt = torch.randn(B, 1, 181, device=device)
        l = crit(pred, gt)
        total_loss += l
        print(f" listener {i} loss: {l.item():.6f}")
    print(" total loss:", total_loss.item())


def train_short(device='cuda'):
    device = device if torch.cuda.is_available() else 'cpu'
    print("Training short demo on", device)
    ds = DummyDataset(n_samples=200, seq_len=12, num_listeners=3, input_dim=181)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_batch)
    model = MultiListenerSystem(input_dim=181, d_model=256, num_listeners=3, pred_len=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    model.train()
    for epoch in range(2):
        epoch_loss = 0.0
        for sp, lis, gt in dl:
            sp = sp.to(device); lis = lis.to(device); gt = gt.to(device)  # gt: [B,pred_len,num_listeners,input_dim]
            out = model(sp, lis, reset_temporal=True)
            loss = 0.0
            for i in range(3):
                pred = out['listener_predictions'][:, :, i, :]  # [B,pred_len,input_dim]
                gt_i = gt[:, :, i, :]                           # [B,pred_len,input_dim]
                loss = loss + crit(pred, gt_i)

            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} avg loss: {epoch_loss/len(dl):.6f}")

# ---------------------------
# Run script
# ---------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Torch device:", device, "CUDA available:", torch.cuda.is_available())
    demo(device=device)
    # run short training if GPU available
    if device == 'cuda':
        train_short(device=device)
