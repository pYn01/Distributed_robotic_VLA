import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# ----------------------------------------------------------
# Aggiunge il path principale del progetto
# ----------------------------------------------------------
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from src.vla.vla_model import VLA  # modello completo


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():

    # ----------------------------------------------------------
    # 1) Carica immagine
    # ----------------------------------------------------------
    img_dir = "/workspace/Dataset/micro/test/20"
    img_file = os.path.join(img_dir, os.listdir(img_dir)[0])

    print(f"Using image: {img_file}")

    img = Image.open(img_file).convert("RGB")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    img_tensor = transform(img).unsqueeze(0)  # (1,3,224,224)

    # ----------------------------------------------------------
    # 2) Carica VLA completo (vision + text + cross-attn)
    # ----------------------------------------------------------
    model = VLA(d_model=256)

    # ----------------------------------------------------------
    # 3) Testo
    # ----------------------------------------------------------
    text = "look at the worm"

    # ----------------------------------------------------------
    # 4) Forward
    # ----------------------------------------------------------
    with torch.no_grad():
        fused, attn, token_ids, mask, vision_tokens = model(img_tensor, text)

    # ----------------------------------------------------------
    # 5) Output info
    # ----------------------------------------------------------
    print("\n---- OUTPUT SHAPES ----")
    print("Vision tokens:", vision_tokens.shape)   # (1, 196, 256)
    print("Token IDs:", token_ids.shape)           # (1, T)
    print("Fused features:", fused.shape)          # (1, T, 256)
    print("Attention map:", attn.shape)            # (1, heads, T, N)


# ----------------------------------------------------------
# Run
# ----------------------------------------------------------
if __name__ == "__main__":
    main()
