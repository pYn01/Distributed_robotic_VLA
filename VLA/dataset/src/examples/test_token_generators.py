# src/examples/test_token_generators.py
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from src.vision.patch_embed import PatchEmbed
from src.text.tokenizer_wrapper import TextTokenizer

def load_dummy_image(size=(224,224)):
    arr = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    return img

def preprocess_image(img, img_size=(224,224)):
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),  # -> [0,1], shape C,H,W
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return tf(img).unsqueeze(0)  # 1,C,H,W

def main():
    img = load_dummy_image((224,224))
    x = preprocess_image(img)
    # Vision
    patch = PatchEmbed(in_ch=3, patch_size=16, d_model=256, img_size=(224,224))
    v_tokens = patch(x)  # 1, N, d
    print("Vision tokens:", v_tokens.shape)

    # Text
    txt = ["lift the leaf and inspect the underside"]
    tt = TextTokenizer(model_name_or_path="facebook/opt-125m", d_model=256)
    toks = tt.encode(txt, max_length=32)
    print("Token ids:", toks['input_ids'].shape, "mask:", toks['attention_mask'].shape)
    t_emb = tt.embed(toks['input_ids'])
    print("Text embeddings:", t_emb.shape)

    # simple shapes check
    assert v_tokens.shape[-1] == t_emb.shape[-1], "d_model mismatch between vision and text embeddings"

if __name__ == "__main__":
    main()
