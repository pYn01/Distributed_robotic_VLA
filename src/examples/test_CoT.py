import torch
import torchvision.transforms as T
from PIL import Image
import os
import sys

# aggiungi path src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.vla.vla_model import VLA
from src.text.tokenizer_wrapper import TextTokenizer

# -----------------------------
# Config e paths
# -----------------------------
IMG_PATH = "/workspace/Dataset_images/micro/test/Tomato___healthy/fc849a80-15fd-486d-bedc-0239e1278660___RS_HL 0494.JPG"  # scegli la tua immagine
TEXT = ["move the leaf to the left"]                     # lista di stringhe
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Carica e trasforma immagine
# -----------------------------
img = Image.open(IMG_PATH).convert("RGB")
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])
img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # (1,3,224,224)

# -----------------------------
# Inizializza modello
# -----------------------------
vocab_size = 1000  # default per OPT tokenizer
model = VLA(d_model=256, vocab_size=vocab_size).to(DEVICE)
model.eval()

# -----------------------------
# Forward pass
# -----------------------------
with torch.no_grad():
    output = model(img_tensor, TEXT)

# -----------------------------
# Stampa risultati
# -----------------------------
print("---- OUTPUT SHAPES ----")
print("Fused tokens:", output["fused"].shape)
print("Vision tokens:", output["vision_tokens"].shape)
print("Text IDs:", output["text_ids"].shape)
print("CoT tokens:", output["cot_tokens"].shape if "cot_tokens" in output else output["cot_logits"].shape)

# Mostriamo i primi token CoT generati
print("\nFirst few CoT tokens:")
print(output["cot_tokens"][0, :10] if "cot_tokens" in output else output["cot_logits"][0, :10])

# 1. Create text tokenizer
text_tok = TextTokenizer(
    model_name_or_path="facebook/opt-125m",
    d_model=256
)
cot_tokens = output["cot_tokens"]
# 2. Decode CoT tokens to text
cot_decoded = text_tok.tokenizer.batch_decode(
    cot_tokens,
    skip_special_tokens=True
)
print("\nDecoded CoT:")
print(cot_decoded[0])
