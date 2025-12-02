# src/heads/primitive_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimitiveHead(nn.Module):
    """
    Autoregressive primitive sequence predictor.
    - Takes fused features from the VLA
    - Generates a sequence of discrete primitive IDs
    - Teacher forcing during training
    """

    def __init__(self, num_primitives=16, hidden_dim=256, max_len=10):
        super().__init__()

        self.num_primitives = num_primitives
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Embedding for primitive IDs
        self.embedding = nn.Embedding(num_primitives, hidden_dim)

        # GRU decoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Output classifier
        self.out = nn.Linear(hidden_dim, num_primitives)

        # special ids
        self.START = 1
        self.STOP = 2

        # create h0 from fused features
        self.context_pool = nn.Linear(hidden_dim, hidden_dim)

    # ------------------------------------------------------------
    # TRAINING MODE
    # fused: (B,T,D)
    # target_seq: (B,L)
    # ------------------------------------------------------------
    def forward(self, fused, target_seq=None):

        # Initial state from fused features
        context = fused.mean(dim=1)                  # (B, D)
        h0 = self.context_pool(context).unsqueeze(0) # (1, B, D)

        # Teacher forcing for training
        if target_seq is not None:
            emb = self.embedding(target_seq)         # (B, L, D)
            out, _ = self.gru(emb, h0)
            logits = self.out(out)                   # (B, L, num_primitives)
            return logits

        # --------------------------------------------------------
        # INFERENCE MODE — autoregressive primitive decoding
        # --------------------------------------------------------
        B = fused.size(0)
        device = fused.device

        prev = torch.full((B, 1), self.START, device=device, dtype=torch.long)
        prev_emb = self.embedding(prev)

        outputs = []
        h = h0

        for _ in range(self.max_len):
            out, h = self.gru(prev_emb, h)
            logits = self.out(out)
            token = torch.argmax(logits, dim=-1)     # (B,1)

            outputs.append(token)
            prev_emb = self.embedding(token)

            if (token == self.STOP).all():
                break

        return torch.cat(outputs, dim=1)  # (B,L)
