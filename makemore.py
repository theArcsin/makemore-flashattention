"""
you give this script some words (one per line) and it will generate more things like it.
uses super state of the art Transformer AI tech
this code is intended to be super hackable. tune it to your needs.

Changes from minGPT:
- I removed the from_pretrained function where we init with GPT2 weights
- I removed dropout layers because the models we train here are small,
  it's not necessary to understand at this stage and at this scale.
- I removed weight decay and all of the complexity around what parameters are
  and are not weight decayed. I don't believe this should make a massive
  difference at the scale that we operate on here.
"""

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    block_size: int = None  # length of the input sequences of integers
    vocab_size: int = None  # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4
    attn_impl: str = "sdpa"  # NEW: 'vanilla' or 'sdpa'

# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# -----------------------------------------------------------------------------
# Attention implementations

class CausalSelfAttentionVanilla(nn.Module):
    """
    Original manual causal attention (masked softmax).
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class CausalSelfAttentionSDPA(nn.Module):
    """
    Modified to use scaled_dot_product_attention (FlashAttention when available).
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        bias = False
        is_causal = True
        dropout = 0.0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=bias)

        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

        self.num_heads = config.n_head
        self.embed_dimension = config.n_embd
        self.is_causal = is_causal

    def forward(self, x):
        qkv = self.c_attn(x)
        batch_size = qkv.size(0)
        embed_dim = qkv.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = qkv.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        dropout = self.dropout if self.training else 0.0
        is_causal = self.is_causal  # keep causal even in eval

        y = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=None,
            dropout_p=dropout,
            is_causal=is_causal
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * head_dim)
        y = self.resid_dropout(self.c_proj(y))
        return y

# -----------------------------------------------------------------------------

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # NEW: selectable attention implementation
        if config.attn_impl == "vanilla":
            self.attn = CausalSelfAttentionVanilla(config)
        elif config.attn_impl == "sdpa":
            self.attn = CausalSelfAttentionSDPA(config)
        else:
            raise ValueError(f"Unknown attention impl: {config.attn_impl}")

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss

# -----------------------------------------------------------------------------
# Bag of Words (BoW) language model

class CausalBoW(nn.Module):
    """
    Causal bag of words. Averages the preceding elements.
    """
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ x
        return y

class BoWBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cbow = CausalBoW(config)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, config.n_embd2),
            c_proj  = nn.Linear(config.n_embd2, config.n_embd),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(torch.tanh(m.c_fc(x)))

    def forward(self, x):
        x = x + self.cbow(x)
        x = x + self.mlpf(x)
        return x

class BoW(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.context_block = BoWBlock(config)
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        x = self.context_block(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

# -----------------------------------------------------------------------------
# Recurrent Neural Net language model

class RNNCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        ht = torch.tanh(self.xh_to_h(xh))
        return ht

class GRUCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        r = torch.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = torch.tanh(self.xh_to_hbar(xhr))
        z = torch.sigmoid(self.xh_to_z(xh))
        ht = (1 - z) * hprev + z * hbar
        return ht

class RNN(nn.Module):
    def __init__(self, config, cell_type):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2))
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        if cell_type == "rnn":
            self.cell = RNNCell(config)
        elif cell_type == "gru":
            self.cell = GRUCell(config)
        else:
            raise ValueError(cell_type)
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        emb = self.wte(idx)

        hprev = self.start.expand((b, -1))
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :]
            ht = self.cell(xt, hprev)
            hprev = ht
            hiddens.append(ht)

        hidden = torch.stack(hiddens, 1)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

# -----------------------------------------------------------------------------
# MLP language model

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        embs = []
        for _ in range(self.block_size):
            tok_emb = self.wte(idx)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size
            embs.append(tok_emb)

        x = torch.cat(embs, -1)
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

# -----------------------------------------------------------------------------
# Bigram language model

class Bigram(nn.Module):
    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        return 1

    def forward(self, idx, targets=None):
        logits = self.logits[idx]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def print_samples(num=10):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to("cpu")
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist()
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print("-" * 80)
    for lst, desc in [(train_samples, "in train"), (test_samples, "in test"), (new_samples, "new")]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print("-" * 80)

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
            logits, loss = model(X, Y)

        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train()
    return mean_loss

# -----------------------------------------------------------------------------
# helper functions for creating the training and test Datasets that emit words

class CharDataset(Dataset):
    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1

    def get_output_length(self):
        return self.max_word_length + 1

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        return "".join(self.itos[i] for i in ix)

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1 + len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix) + 1:] = -1
        return x, y

def create_datasets(input_file):
    with open(input_file, "r") as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words]
    words = [w for w in words if w]
    chars = sorted(list(set("".join(words))))
    max_word_length = max(len(w) for w in words)
    print(f"number of examples in the dataset: {len(words)}")
    print(f"max word length: {max_word_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print("vocabulary:")
    print("".join(chars))

    test_set_size = min(1000, int(len(words) * 0.1))
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)
    return train_dataset, test_dataset

class InfiniteDataLoader:
    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make More")
    parser.add_argument("--input-file", "-i", type=str, default="names.txt", help="input file with things one per line")
    parser.add_argument("--work-dir", "-o", type=str, default="out", help="output working directory")
    parser.add_argument("--resume", action="store_true", help="resume optimization from existing model in the workdir")
    parser.add_argument("--sample-only", action="store_true", help="just sample from the model and quit, don't train")
    parser.add_argument("--num-workers", "-n", type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument("--max-steps", type=int, default=-1, help="max number of optimization steps, or -1 for infinite")
    parser.add_argument("--device", type=str, default="cuda", help="device to use: cpu|cuda|cuda:2|mps")
    parser.add_argument("--seed", type=int, default=3407, help="seed")

    # sampling
    parser.add_argument("--top-k", type=int, default=-1, help="top-k for sampling, -1 means no top-k")

    # model
    parser.add_argument("--type", type=str, default="transformer", help="model class: bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument("--n-layer", type=int, default=4, help="number of layers")
    parser.add_argument("--n-head", type=int, default=4, help="number of heads")
    parser.add_argument("--n-embd", type=int, default=64, help="embedding dim")
    parser.add_argument("--n-embd2", type=int, default=64, help="secondary dim")

    # NEW: attention switch + timing controls
    parser.add_argument("--attn", type=str, default="sdpa", choices=["vanilla", "sdpa"],
                        help="attention impl: vanilla or sdpa")
    parser.add_argument("--stop-step", type=int, default=-1,
                        help="stop training after this step (for timing). -1 = no early stop")
    parser.add_argument("--warmup-steps", type=int, default=50,
                        help="ignore first N steps in timing stats")

    # optimization
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="batch size")
    parser.add_argument("--learning-rate", "-l", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight-decay", "-w", type=float, default=0.01, help="weight decay")
    args = parser.parse_args()
    print(vars(args))

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")

    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_embd2=args.n_embd2,
        attn_impl=args.attn,  # NEW
    )

    if args.type == "transformer":
        model = Transformer(config)
    elif args.type == "bigram":
        model = Bigram(config)
    elif args.type == "mlp":
        model = MLP(config)
    elif args.type == "rnn":
        model = RNN(config, cell_type="rnn")
    elif args.type == "gru":
        model = RNN(config, cell_type="gru")
    elif args.type == "bow":
        model = BoW(config)
    else:
        raise ValueError(f"model type {args.type} is not recognized")

    model.to(args.device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")

    if args.resume or args.sample_only:
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, "model.pt"), map_location=args.device))
    if args.sample_only:
        print_samples(num=50)
        sys.exit()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8
    )

    batch_loader = InfiniteDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers
    )

    best_loss = None
    step = 0

    timed_steps = 0
    timed_ms_sum = 0.0

    while True:
        t0 = time.time()

        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
            logits, loss = model(X, Y)

        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

        t1 = time.time()
        step_ms = (t1 - t0) * 1000.0

        # timing stats (ignore warmup)
        if step >= args.warmup_steps:
            timed_steps += 1
            timed_ms_sum += step_ms
            if timed_steps % 50 == 0:
                print(f"[timing] attn={args.attn} avg_step_ms={timed_ms_sum / timed_steps:.2f} over {timed_steps} steps")

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {step_ms:.2f}ms")

        # evaluate
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss = evaluate(model, test_dataset, batch_size=100, max_batches=10)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # sample
        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        # early stop for timing
        if args.stop_step >= 0 and step >= args.stop_step:
            avg = timed_ms_sum / max(1, timed_steps)
            print(f"[done] attn={args.attn} avg_step_ms={avg:.2f} (warmup={args.warmup_steps}, timed_steps={timed_steps})")
            break

        step += 1
        if args.max_steps >= 0 and step >= args.max_steps:
            break
