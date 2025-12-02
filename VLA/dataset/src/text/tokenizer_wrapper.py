# src/text/tokenizer_wrapper.py
from transformers import AutoTokenizer
import torch
import torch.nn as nn

class TextTokenizer:
    """
    Wrapper around a HuggingFace tokenizer that returns token ids, attention_mask and
    a learnable token embedding layer (optional).
    """
    def __init__(self, model_name_or_path="facebook/opt-125m", d_model=768, use_embedding_layer=True):
        # tokenizer (fast)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        # if tokenizer has no pad token, set one
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.vocab_size = len(self.tokenizer)
        self.d_model = d_model
        self.use_embedding_layer = use_embedding_layer
        if use_embedding_layer:
            self.embedding = nn.Embedding(self.vocab_size, d_model)
        else:
            self.embedding = None

    def encode(self, texts, max_length=64, padding=True, truncation=True, return_tensors="pt"):
        # returns token_ids (B, T), attention_mask (B, T)
        toks = self.tokenizer(texts, padding='max_length' if padding else False,
                              truncation=truncation, max_length=max_length, return_tensors=return_tensors)
        return toks

    def embed(self, input_ids):
        # input_ids: (B, T)
        if not self.use_embedding_layer:
            raise RuntimeError("embedding layer disabled")
        return self.embedding(input_ids)  # (B, T, d_model)

if __name__ == "__main__":
    # quick test
    wrapper = TextTokenizer(model_name_or_path="facebook/opt-125m", d_model=256)
    texts = ["lift the leaf and check underneath", "inspect for pests"]
    toks = wrapper.encode(texts, max_length=16)
    print("input_ids shape:", toks['input_ids'].shape, "attention_mask shape:", toks['attention_mask'].shape)
    emb = wrapper.embed(toks['input_ids'])
    print("emb shape:", emb.shape)
