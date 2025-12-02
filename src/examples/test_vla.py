#!/usr/bin/env python3
# src/examples/test_vla.py
"""
Quick integration test for VLA + PrimitiveHead.

- If the VLA instance doesn't already have a `primitive` attribute,
  this script creates one (num_primitives=16) and attaches it.
- Runs a forward pass to obtain fused features, then tests:
  * primitive head training call (teacher forcing stub with random targets)
  * primitive head inference (autoregressive)
"""

import os
import sys
import torch

# add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.vla.vla_model import VLA
from src.CoT.primitive_head import PrimitiveHead

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- instantiate model ----
    d_model = 256
    num_primitives = 16
    model = VLA(d_model=d_model, vocab_size=1000).to(device)
    model.eval()

    # attach primitive head if not present
    if not hasattr(model, "primitive"):
        print("Attaching PrimitiveHead(num_primitives=%d) to model..." % num_primitives)
        model.primitive = PrimitiveHead(num_primitives=num_primitives, hidden_dim=d_model).to(device)
    else:
        print("Model already has attribute 'primitive' - using existing one.")
        # ensure it's on the right device
        model.primitive.to(device)

    # ---- prepare dummy input ----
    B = 1
    img_tensor = torch.randn(B, 3, 224, 224, device=device)  # random image
    # Use a short text instruction list (the TextEncoder inside VLA may contact HF tokenizer)
    text = ["move the leaf to the left"]

    # ---- forward pass through VLA to obtain fused features ----
    with torch.no_grad():
        out = model(img_tensor, text)   # returns dict with "fused" or "cot_tokens" depending on implementation

    if "fused" not in out:
        raise RuntimeError("vla_model.forward did not return 'fused' in the output dict. Check vla_model.py")

    fused = out["fused"]   # (B, Nt, D)
    print("Fused tokens shape:", fused.shape)
    if "attn" in out:
        print("Attention shape:", out["attn"].shape)

    # ---- Test primitive head in training mode (teacher forcing) ----
    # Create random target primitive sequence (B, L) with values in [0, num_primitives-1]
    L = 6
    target_seq = torch.randint(low=0, high=num_primitives, size=(B, L), device=device, dtype=torch.long)
    print("Target primitive seq (random):", target_seq)

    try:
        prim_logits = model.primitive(fused, target_seq=target_seq)  # (B, L, num_primitives)
        print("Primitive logits shape (training):", prim_logits.shape)
    except Exception as e:
        print("Error running primitive head in training mode:", e)
        raise

    # ---- Test primitive head in inference mode (autoregressive) ----
    try:
        prim_seq = model.primitive(fused)  # (B, L_out) variable length per implementation
        print("Predicted primitive sequence shape (inference):", prim_seq.shape)
        print("Predicted primitive sequence (ids):", prim_seq)
    except Exception as e:
        print("Error running primitive head in inference mode:", e)
        raise

    # ---- If VLA has CoT tokens, print basic info ----
    if "cot_tokens" in out:
        print("CoT tokens present. Shape:", out["cot_tokens"].shape)
    elif "cot_logits" in out:
        print("CoT logits present. Shape:", out["cot_logits"].shape)
    else:
        print("No CoT output found in VLA output (ok if your model uses a different head).")

    print("\nAll tests completed successfully.")

if __name__ == "__main__":
    main()
