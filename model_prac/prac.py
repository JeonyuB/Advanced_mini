#!/usr/bin/env python3
"""
Tiny Transformer LLM for CSV QA dataset (PyTorch)
- Character-level tokenizer on CSV 'question' + 'answer'
- GPT-style causal self-attention
- Train and save .pth checkpoint
"""
import argparse
import os
import math
from dataclasses import dataclass
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- Tokenizer ---------------------
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s: str):
        return torch.tensor([self.stoi[c] for c in s if c in self.stoi], dtype=torch.long)

    def decode(self, idx: torch.Tensor):
        return ''.join([self.itos[int(i)] for i in idx])

# --------------------- Model ---------------------
@dataclass
class GPTConfig:
    vocab_size: int
    n_embd: int = 300 # 각 토큰을 표현하는 벡터 크기. 차원이 클수록 더 풍부하게 의미 담음.
    n_head: int = 4
    n_layer: int = 7 #모델 안에 쌓이는 Transformer 블록의 개수
    block_size: int = 128
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.drop(y)
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config.n_embd, config.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        tok = self.token_emb(idx)
        pos = self.pos_emb(pos)
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# --------------------- CSV Dataset ---------------------
class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, block_size=128, text_col='question', answer_col='answer'):
        df = pd.read_csv(csv_path)
        texts = [str(q) + " " + str(a) for q, a in zip(df[text_col], df[answer_col])]
        text = "\n".join(texts)
        self.tokenizer = CharTokenizer(text)
        self.data_ids = self.tokenizer.encode(text)
        self.block_size = block_size

    def __len__(self):
        return len(self.data_ids) - self.block_size

    def __getitem__(self, idx):
        x = self.data_ids[idx: idx + self.block_size]
        y = self.data_ids[idx + 1: idx + 1 + self.block_size]
        return x, y

# --------------------- Training ---------------------
def train(model, optimizer, train_loader, device, steps, log_interval=100, grad_clip=1.0):
    model.train()
    step = 0
    ema_loss = None
    start = time.time()
    while step < steps:
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            step += 1
            ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()
            if step % log_interval == 0:
                elapsed = time.time() - start
                print(f"step {step:6d} | loss {ema_loss:.4f} | {elapsed:.1f}s")
                start = time.time()
            if step >= steps:
                break

# --------------------- Main ---------------------
def main(csv_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=csv_path)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--block', type=int, default=128)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--checkpoint', type=str, default='tinyllm_qa.pth')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    dataset = CSVDataset(args.data, block_size=args.block)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    config = GPTConfig(
        vocab_size=dataset.tokenizer.vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block,
        dropout=args.dropout,
    )

    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Training…")
    train(model, optimizer, loader, device, steps=args.steps)

    ckpt = {
        'config': {
            'vocab_size': config.vocab_size,
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_layer': config.n_layer,
            'block_size': config.block_size,
            'dropout': config.dropout,
        },
        'model': model.state_dict(),
        'tokenizer': {
            'stoi': dataset.tokenizer.stoi,
            'itos': dataset.tokenizer.itos,
        }
    }
    torch.save(ckpt, args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")

if __name__ == '__main__':
    main("data/qa_dataset.csv")
