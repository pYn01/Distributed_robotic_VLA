#!/usr/bin/env python3
# external_models/train_primitive_head.py
"""
Train only the PrimitiveHead using fused features extracted from
Embodied-CoT model (ecot-openvla-7b-bridge).

Usage example:
python3 external_models/train_primitive_head.py \
  --model_id Embodied-CoT/ecot-openvla-7b-bridge \
  --data_root /../dataset \
  --train_json annotations/train.jsonl \
  --batch_size 2 --epochs 5 --lr 1e-4 --save_dir external_models/checkpoints
"""

import argparse
import os
import json
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from transformers import AutoModelForVision2Seq, AutoProcessor
from src.CoT.primitive_head import PrimitiveHead

# --------------------------
# Dataset wrapper
# --------------------------
class VLADataset(Dataset):
    def __init__(self, jsonl_path, images_root="", processor=None):
        self.rows = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.rows.append(json.loads(line.strip()))
        self.images_root = images_root
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img_path = r["image"]
        if self.images_root and not os.path.isabs(img_path):
            img_path = os.path.join(self.images_root, img_path)
        img = Image.open(img_path).convert("RGB")
        instruction = r.get("instruction", "")
        primitive_ids = r.get("primitive_ids", [])
        proc_inputs = self.processor(images=img, text=instruction, return_tensors="pt", padding="max_length", truncation=True)
        # squeeze batch dim
        proc_inputs = {k: v.squeeze(0) for k, v in proc_inputs.items()}
        return proc_inputs, torch.tensor(primitive_ids, dtype=torch.long)

def collate_fn(batch):
    keys = batch[0][0].keys()
    batch_proc = {}
    for k in keys:
        batch_proc[k] = torch.stack([b[0][k] for b in batch])
    prims = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True, padding_value=15)  # STOP id = 15
    return batch_proc, prims

# --------------------------
# Fused extractor (robust)
# --------------------------
def extract_fused_from_model(model, batch_proc, device):
    inputs = {k: v.to(device) for k, v in batch_proc.items()}
    model.eval()
    with torch.no_grad():
        # If model provides helper
        if hasattr(model, "get_image_features") and "pixel_values" in inputs:
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            # If shape (B, D) expand to (B, 1, D)
            if feats.dim() == 2:
                feats = feats.unsqueeze(1)
            return feats

        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        # Try common fields
        if hasattr(outputs, "encoder_last_hidden_state"):
            return outputs.encoder_last_hidden_state
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            return outputs.hidden_states[-1]

    raise RuntimeError("Can't extract fused features — inspect model outputs.")

# --------------------------
# Training loop
# --------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading processor and model:", args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    # load model: use quantization flags if needed
    load_kwargs = {"trust_remote_code": True}
    if args.load_in_4bit:
        load_kwargs.update({"load_in_4bit": True, "device_map": "auto"})
    else:
        load_kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})

    model = AutoModelForVision2Seq.from_pretrained(args.model_id, **load_kwargs)
    model.to(device)
    model.eval()
    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    # Create dataset
    train_json = os.path.join(args.data_root, args.train_json)
    ds = VLADataset(train_json, images_root=args.data_root, processor=processor)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4, pin_memory=True)

    # Primitive head
    prim_head = PrimitiveHead(num_primitives=args.num_primitives, hidden_dim=args.hidden_dim, max_len=args.max_len).to(device)
    optimizer = optim.AdamW(prim_head.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=15)  # ignore pad/STOP

    os.makedirs(args.save_dir, exist_ok=True)

    prim_head.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (batch_proc, prim_targets) in enumerate(dl):
            fused = extract_fused_from_model(model, batch_proc, device)  # (B, seq_len, D) on device
            prim_targets = prim_targets.to(device)
            logits = prim_head(fused, target_seq=prim_targets)  # (B, L, C)
            B, L, C = logits.shape
            loss = criterion(logits.view(B*L, C), prim_targets.view(B*L))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % args.log_every == 0:
                avg = running_loss / args.log_every
                print(f"Epoch {epoch+1} step {i+1}/{len(dl)} — avg loss {avg:.4f}")
                running_loss = 0.0
        # save checkpoint per epoch
        ckpt_path = os.path.join(args.save_dir, f"primitive_head_epoch{epoch+1}.pt")
        torch.save(prim_head.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # final save
    final_path = os.path.join(args.save_dir, "primitive_head_final.pt")
    torch.save(prim_head.state_dict(), final_path)
    print("Training finished. Saved:", final_path)

# --------------------------
# Argparse
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="Embodied-CoT/ecot-openvla-7b-bridge")
    p.add_argument("--data_root", type=str, required=True, help="root folder containing images and annotations")
    p.add_argument("--train_json", type=str, default="annotations/train.jsonl")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_dir", type=str, default="external_models/checkpoints")
    p.add_argument("--num_primitives", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--max_len", type=int, default=10)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--load_in_4bit", action="store_true", help="set if you want 4-bit loading via bitsandbytes")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
