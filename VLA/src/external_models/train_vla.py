#!/usr/bin/env python3
"""
Train script for VLA model (CoT + Primitive heads).

- Uses the JSONL dataset at dataset/annotation/train.jsonl (see workspace).
- By default freezes the VLA backbone and trains only the CoT + Primitive heads.
- To fine-tune entire VLA, pass --unfreeze_vla.
"""
import os
import json
import argparse
from typing import List

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# import workspace model & helpers
from src.vla.vla_model import VLA  # [VLA](VLA/src/vla/vla_model.py)
from src.text.tokenizer_wrapper import TextTokenizer  # [TextTokenizer](VLA/src/text/tokenizer_wrapper.py)

# -----------------------
# Dataset
# -----------------------
class VLATrainDataset(Dataset):
    def __init__(self, jsonl_path: str, images_root: str = "", cot_max_len: int = 64, img_size=(224,224), tokenizer=None):
        self.rows = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        self.images_root = images_root
        self.cot_max_len = cot_max_len
        self.tokenizer = tokenizer  # instance of TextTokenizer (workspace)
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img_path = r["image"]
        if self.images_root and not os.path.isabs(img_path):
            img_path = os.path.join(self.images_root, img_path)
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)

        # instruction used as text input to VLA (list[str] expected by VLA.text)
        instruction = r.get("instruction", "")

        # CoT: join array of sentences into one target string
        cot_list: List[str] = r.get("cot", [])
        cot_text = " ".join(cot_list) if isinstance(cot_list, list) else str(cot_list)

        # tokenize CoT target using provided tokenizer (returns tensors)
        toks = self.tokenizer.encode(
            [cot_text],
            max_length=self.cot_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        cot_ids = toks["input_ids"].squeeze(0)  # (L,)

        # optional primitive targets
        prim_ids = r.get("primitive_ids", [])
        prim_tensor = torch.tensor(prim_ids, dtype=torch.long) if prim_ids is not None else torch.tensor([], dtype=torch.long)

        return img_t, instruction, cot_ids, prim_tensor

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)             # (B,3,H,W)
    texts = [b[1] for b in batch]                                # list[str]
    cot_targets = torch.stack([b[2] for b in batch], dim=0)      # (B, L)
    prim_targets = [b[3] for b in batch]                         # list of variable length tensors
    # pad primitive targets to same length (if present)
    if any([p.numel() > 0 for p in prim_targets]):
        prims_padded = nn.utils.rnn.pad_sequence(prim_targets, batch_first=True, padding_value=15)
    else:
        prims_padded = None
    return imgs, texts, cot_targets, prims_padded

# -----------------------
# Training
# -----------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print("Device:", device)

    # Instantiate an external tokenizer to get vocab_size that matches model's CoT head
    external_tok = TextTokenizer(model_name_or_path=args.tokenizer_name, d_model=args.d_model, use_embedding_layer=True)
    vocab_size = external_tok.vocab_size

    # Instantiate model with vocab_size matching tokenizer
    model = VLA(d_model=args.d_model, vocab_size=vocab_size).to(device)  # [VLA](VLA/src/vla/vla_model.py)
    print("Model instantiated.")

    # Optionally freeze VLA backbone (vision/text/cross). Train only heads by default.
    if args.freeze_vla:
        print("Freezing vision/text/cross modules. Training CoT + Primitive heads only.")
        for name, p in model.named_parameters():
            if name.startswith("cot") or name.startswith("primitive"):
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        print("Fine-tuning entire VLA model.")
        for p in model.parameters():
            p.requires_grad = True

    # prepare dataset & dataloader
    jsonl_path = args.train_json if os.path.isabs(args.train_json) else os.path.join(args.data_root, args.train_json)
    ds = VLATrainDataset(jsonl_path=jsonl_path, images_root=args.data_root, cot_max_len=args.cot_max_len, img_size=(args.img_size,args.img_size), tokenizer=external_tok)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # losses and optimizer
    pad_id = external_tok.tokenizer.pad_token_id if hasattr(external_tok, "tokenizer") else 0
    cot_criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    prim_pad_id = 15  # dataset uses 15 as padding/STOP id in examples
    prim_criterion = nn.CrossEntropyLoss(ignore_index=prim_pad_id)

    # collect parameters to optimize
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, (imgs, texts, cot_targets, prim_targets) in enumerate(dl):
            imgs = imgs.to(device)
            # cot_targets: (B, L)
            cot_targets = cot_targets.to(device)

            # Forward pass through VLA to get fused features.
            # If backbone frozen, run vision+text without grads for speed.
            if args.freeze_vla:
                with torch.no_grad():
                    out = model(imgs, texts)
                    fused = out["fused"]
            else:
                out = model(imgs, texts)
                fused = out["fused"]

            # CoT head training (teacher forcing)
            cot_logits = model.cot(fused, target_tokens=cot_targets)  # (B, L, V)
            B, L, V = cot_logits.shape
            loss_cot = cot_criterion(cot_logits.view(B*L, V), cot_targets.view(B*L))

            # Primitive head training if targets present
            loss_prim = torch.tensor(0.0, device=device)
            if prim_targets is not None:
                prim_targets = prim_targets.to(device)
                prim_logits = model.primitive(fused, target_seq=prim_targets)  # (B, Lp, C)
                b2, lp, c = prim_logits.shape
                loss_prim = prim_criterion(prim_logits.view(b2*lp, c), prim_targets.view(b2*lp))

            loss = loss_cot + args.prim_loss_weight * loss_prim

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % args.log_every == 0:
                avg = running_loss / args.log_every
                print(f"Epoch {epoch+1}/{args.epochs} step {step+1}/{len(dl)} — avg loss {avg:.4f}")
                running_loss = 0.0

        # save checkpoint per epoch
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt = os.path.join(args.save_dir, f"vla_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args)
        }, ckpt)
        print("Saved checkpoint:", ckpt)

    print("Training finished.")

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="../dataset", help="root for images if paths in jsonl are relative")
    p.add_argument("--train_json", type=str, default="dataset/annotation/train.jsonl", help="train jsonl path (relative to data_root if not absolute)")
    p.add_argument("--tokenizer_name", type=str, default="facebook/opt-125m")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--cot_max_len", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_dir", type=str, default="VLA/external_models/checkpoints")
    p.add_argument("--freeze_vla", action="store_true", default=True, help="freeze VLA backbone and train only heads")
    p.add_argument("--unfreeze_vla", action="store_true", help="alias to not freeze backbone")
    p.add_argument("--use_cuda", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--prim_loss_weight", type=float, default=1.0)
    args = p.parse_args()
    # handle mutually exclusive flag
    if args.unfreeze_vla:
        args.freeze_vla = True
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)