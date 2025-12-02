# src/heads/cot_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoTHead(nn.Module):
    """
    Chain-of-Thought Generator Head (GRU-based)
    - Teacher forcing during training
    - Autoregressive sampling during inference
    """

    def __init__(self, vocab_size=1000, hidden_dim=256, max_len=64):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Embedding for CoT tokens
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # GRU decoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Output projection to vocab
        self.out = nn.Linear(hidden_dim, vocab_size)

        # Special tokens
        self.BOS = 1
        self.EOS = 2

        # Small pooling to convert fused tokens → decoder initial state
        self.context_pool = nn.Linear(hidden_dim, hidden_dim)

    # ------------------------------------------------------------
    # TRAINING: teacher forcing
    # fused: (B, T, D)
    # target_tokens: (B, L)
    # ------------------------------------------------------------
    def forward(self, fused, target_tokens=None):

        # initial GRU state from pooled fused
        context = fused.mean(dim=1)                # (B, D)
        h0 = self.context_pool(context).unsqueeze(0)  # (1, B, D)

        # TRAINING MODE
        if target_tokens is not None:
            emb = self.embedding(target_tokens)       # (B, L, D)
            out, _ = self.gru(emb, h0)                # (B, L, D)
            logits = self.out(out)                    # (B, L, vocab)
            return logits

        # --------------------------------------------------------
        # INFERENCE MODE — autoregressive CoT generation
        # --------------------------------------------------------
        B = fused.size(0)
        device = fused.device

        # start with <BOS>
        prev = torch.full((B, 1), self.BOS, device=device, dtype=torch.long)
        prev_emb = self.embedding(prev)

        outputs = []
        h = h0

        for _ in range(self.max_len):
            out, h = self.gru(prev_emb, h)
            logits = self.out(out)               # (B,1,vocab)
            token = torch.argmax(logits, dim=-1) # (B,1)

            outputs.append(token)
            prev_emb = self.embedding(token)

            if (token == self.EOS).all():
                break

        return torch.cat(outputs, dim=1)  # (B, L)
