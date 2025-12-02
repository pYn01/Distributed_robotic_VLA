# src/vision/patch_embed.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """
    Convert an image to patch tokens (ViT-style) without using an explicit Conv2d layer.
    - img: (B, C, H, W)
    - patch_size: int (e.g., 16)
    Output:
    - patches: (B, N, d_model)  where N = (H/patch)*(W/patch)
    - pos_emb: (1, N, d_model) added internally (learnable)
    """
    def __init__(self, in_ch=3, patch_size=16, d_model=768, img_size=(224,224)):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.in_ch = in_ch

        # projection: linear on flattened patch vector
        patch_dim = in_ch * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, d_model)

        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch_size"
        self.grid_size = (H // patch_size, W // patch_size)
        n_patches = self.grid_size[0] * self.grid_size[1]
        # learnable pos embedding
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

    def forward(self, x):
        # x: B, C, H, W
        B, C, H, W = x.shape
        p = self.patch_size
        # unfold -> B, C * p * p, N
        patches = x.unfold(2, p, p).unfold(3, p, p)  # B, C, H/p, W/p, p, p
        patches = patches.contiguous().view(B, C, -1, p * p)  # B, C, N, p*p
        # move channels inside flattened vector
        patches = patches.permute(0, 2, 1, 3).reshape(B, -1, C * p * p)  # B, N, patch_dim
        # linear projection
        tokens = self.proj(patches)  # B, N, d_model
        # add pos emb (broadcast)
        if tokens.shape[1] != self.pos_emb.shape[1]:
            # if image size differs, recompute pos_emb on the fly (simple resize via interpolation)
            # safe fallback: interpolate
            pos = F.interpolate(self.pos_emb.permute(0,2,1), size=tokens.shape[1], mode='linear', align_corners=False)
            pos = pos.permute(0,2,1)
        else:
            pos = self.pos_emb
        return tokens + pos

if __name__ == "__main__":
    # quick unit test
    x = torch.randn(2, 3, 224, 224)
    pe = PatchEmbed(in_ch=3, patch_size=16, d_model=256, img_size=(224,224))
    out = pe(x)
    print("patch tokens shape:", out.shape)  # expected (2, 196, 256)
