# src/vla/vla_model.py

import torch
import torch.nn as nn

from src.vision.patch_embed import PatchEmbed
from src.text.tokenizer_wrapper import TextTokenizer
from src.CoT.cot_head import CoTHead
from src.CoT.primitive_head import PrimitiveHead


# ---------------------------------------------------------
# 1) Vision Encoder ( PatchEmbed + Transformer )
# ---------------------------------------------------------
class VisionEncoder(nn.Module):
    def __init__(self, dim=256, img_size=(224,224), patch_size=16):
        super().__init__()

        # PatchEmbed 
        self.patch = PatchEmbed(
            in_ch=3,
            patch_size=patch_size,
            d_model=dim,
            img_size=img_size
        )

        #  transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, img):
        # img: B, 3, H, W
        tokens = self.patch(img)  # (B, N, dim)
        tokens = self.transformer(tokens)
        return tokens


# ---------------------------------------------------------
# 2) Text Encoder (tokenizer_wrapper.py)
# ---------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(self, d_model=256, max_length=32):
        super().__init__()
        self.tokenizer = TextTokenizer(
            model_name_or_path="facebook/opt-125m",
            d_model=d_model,
            use_embedding_layer=True
        )
        self.max_length = max_length

    def forward(self, text):
        toks = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True
        )

        ids = toks["input_ids"]       # (B, T)
        mask = toks["attention_mask"] # (B, T)

        emb = self.tokenizer.embed(ids)  # (B, T, d_model)
        return emb, ids, mask


# ---------------------------------------------------------
# 3) CROSS ATTENTION 
# ---------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=256, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, text_tokens, vision_tokens):
        attended, attn_weights = self.attn(
            query=text_tokens,
            key=vision_tokens,
            value=vision_tokens
        )
        return self.norm(attended + text_tokens), attn_weights

# ---------------------------------------------------------
# VLA MODEL COMBINED
# ---------------------------------------------------------
class VLA(nn.Module):
    def __init__(self, d_model=256, vocab_size=1000):
        super().__init__()
        self.vision = VisionEncoder(dim=d_model)
        self.text = TextEncoder(d_model=d_model)
        self.cross = CrossAttentionBlock(dim=d_model)
        self.cot = CoTHead(vocab_size=vocab_size,hidden_dim=d_model)
        self.primitive = PrimitiveHead(num_primitives=16, hidden_dim=d_model)

    def forward(self, image, text, cot_target=None):
        vision_tokens = self.vision(image)          # (B, Nv, D)
        text_emb, ids, mask = self.text(text)       # (B, Nt, D)
        text_emb = text_emb.to(vision_tokens.device) # for same device
        fused, attn = self.cross(text_emb, vision_tokens)
    
        # training mode
        if cot_target is not None:
            cot_logits = self.cot(fused, target_tokens=cot_target)
            return {
                "fused": fused,
                "attn": attn,
                "text_ids": ids,
                "vision_tokens": vision_tokens,
                "cot_logits": cot_logits
            }

        # inference mode
        cot_tokens = self.cot(fused)
        return {
            "fused": fused,
            "attn": attn,
            "text_ids": ids,
            "vision_tokens": vision_tokens,
            "cot_tokens": cot_tokens
        }
# ---------------------------------------------------------