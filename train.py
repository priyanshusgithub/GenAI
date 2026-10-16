"""Comparable word-level next-token experiments on Alice in Wonderland.

Run: python train.py --epochs 5
The corpus is downloaded on first run. No target words enter a model's context.
"""
import argparse
import copy
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
SOURCE = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?;:]", re.I)
SEEDS = ["alice was beginning to", "the rabbit said to alice"]
TEMPERATURES = (0.5, 1.0, 1.5)


def corpus():
    path = ROOT / "data" / "pg11.txt"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with urlopen(SOURCE, timeout=40) as response:
            path.write_bytes(response.read())
    raw = path.read_text(encoding="utf-8-sig")
    start = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", raw, re.I)
    end = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", raw, re.I)
    if not start or not end or end.start() <= start.end():
        raise ValueError("Could not locate Gutenberg book boundaries; inspect source format")
    return TOKEN_RE.findall(raw[start.end():end.start()].lower())


class Windows(Dataset):
    def __init__(self, ids, context):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.context = context

    def __len__(self):
        return len(self.ids) - self.context

    def __getitem__(self, i):
        return self.ids[i:i + self.context], self.ids[i + self.context]


class LanguageModel(nn.Module):
    def __init__(self, vocab, kind, emb=48, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb)
        recurrent = {"rnn": nn.RNN, "birnn": nn.RNN,
                     "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.recurrent = recurrent(emb, hidden, batch_first=True,
                                   bidirectional=(kind == "birnn"))
        self.output = nn.Linear(hidden * (2 if kind == "birnn" else 1), vocab)
        self.kind = kind

    def forward(self, x):
        states, _ = self.recurrent(self.embedding(x))
        # Both directions only process words already in the input window.
        if self.kind == "birnn":
            summary = torch.cat((states[:, -1, :64], states[:, 0, 64:]), dim=1)
        else:
            summary = states[:, -1, :]
        return self.output(summary)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = correct = count = 0
    for x, y in loader:
        logits = model(x.to(device))
        loss_sum += nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item()
        correct += (logits.argmax(1) == y.to(device)).sum().item()
        count += len(y)
    mean = loss_sum / count
    return {"loss": mean, "perplexity": math.exp(mean), "accuracy": correct / count}


def train(model, train_loader, val_loader, epochs, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total = count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
        val = evaluate(model, val_loader, device)
        record = {"epoch": epoch + 1, "train_loss": total / count,
                  "validation_loss": val["loss"], "validation_accuracy": val["accuracy"]}
        history.append(record)
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        print(model.kind, record, flush=True)
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    return history, elapsed, best_epoch


@torch.no_grad()
def generate(model, seed, stoi, itos, context, temperature, count, rng, device):
    model.eval()
    words = TOKEN_RE.findall(seed.lower())
    for _ in range(count):
        ids = [stoi.get(w, stoi["<unk>"]) for w in words[-context:]]
        if len(ids) < context:
            ids = [stoi["<pad>"]] * (context - len(ids)) + ids
        logits = model(torch.tensor([ids], device=device))[0].cpu()
        logits[stoi["<pad>"]] = -float("inf")
        logits[stoi["<unk>"]] = -float("inf")
        probabilities = torch.softmax(logits / temperature, dim=0)
        next_id = torch.multinomial(probabilities, 1, generator=rng).item()
        words.append(itos[next_id])
    return words


def diversity(words, generated_only):
    words = words[-generated_only:]
    bigrams = list(zip(words, words[1:]))
    return {"distinct_2": len(set(bigrams)) / len(bigrams),
            "adjacent_repetition_rate": sum(a == b for a, b in bigrams) / len(bigrams)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--context", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-train-windows", type=int, default=0,
                   help="Optional training cap for a quick trial; 0 uses the full training partition")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(2)
    tokens = corpus()
    n = len(tokens)
    # Chronological, disjoint token partitions; windows cannot cross boundaries.
    train_words, val_words, test_words = tokens[:int(.8*n)], tokens[int(.8*n):int(.9*n)], tokens[int(.9*n):]
    counts = Counter(train_words)
    itos = ["<pad>", "<unk>"] + sorted(w for w, count in counts.items() if count >= 2)
    stoi = {word: i for i, word in enumerate(itos)}
    to_ids = lambda seq: [stoi.get(word, stoi["<unk>"]) for word in seq]
    train_ids = to_ids(train_words)
    if args.max_train_windows and len(train_ids) - args.context > args.max_train_windows:
        train_ids = train_ids[:args.max_train_windows + args.context]
    datasets = [Windows(ids, args.context) for ids in [train_ids, to_ids(val_words), to_ids(test_words)]]
    if min(len(d) for d in datasets) <= 0:
        raise ValueError("Corpus shorter than the context window")
    train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(datasets[1], batch_size=256)
    test_loader = DataLoader(datasets[2], batch_size=256)
    results = {"source": SOURCE, "dataset": {"tokens": n, "train_partition_tokens": len(train_words),
               "train_tokens_used": len(train_ids), "validation_tokens": len(val_words),
               "test_tokens": len(test_words), "vocabulary": len(itos),
               "unknown_validation_rate": val_words and sum(w not in stoi for w in val_words)/len(val_words),
               "unknown_test_rate": test_words and sum(w not in stoi for w in test_words)/len(test_words)},
               "configuration": vars(args), "models": {}}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    device = torch.device(args.device)
    for kind in ("rnn", "birnn", "lstm", "gru"):
        torch.manual_seed(42)
        model = LanguageModel(len(itos), kind)
        history, elapsed, best_epoch = train(model, train_loader, val_loader, args.epochs, device)
        test = evaluate(model, test_loader, device)
        samples = []
        for seed in SEEDS:
            for temp in TEMPERATURES:
                rng = torch.Generator().manual_seed(2026)  # same sampling RNG for each setting
                words = generate(model, seed, stoi, itos, args.context, temp, 50, rng, device)
                samples.append({"seed": seed, "temperature": temp, "text": " ".join(words),
                                **diversity(words, 50)})
        results["models"][kind] = {"parameters": sum(p.numel() for p in model.parameters()),
            "training_seconds": elapsed, "selected_epoch": best_epoch,
            "history": history, "test": test, "samples": samples}
        (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for kind, value in results["models"].items():
        plt.plot([v["epoch"] for v in value["history"]],
                 [v["validation_loss"] for v in value["history"]], marker="o", label=kind.upper())
    plt.xlabel("Epoch"); plt.ylabel("Validation cross-entropy")
    plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(out / "validation_loss.png", dpi=160)
    plt.close()
    with (out / "generated_samples.md").open("w", encoding="utf-8") as f:
        f.write("# Generated text samples\n\nEach continuation contains 50 generated tokens.\n\n")
        for kind, value in results["models"].items():
            f.write(f"## {kind.upper()}\n\n")
            for s in value["samples"]:
                f.write(f"**Seed:** {s['seed']} | **Temperature:** {s['temperature']}\n\n{s['text']}\n\n")
    print("Saved results to", out)


if __name__ == "__main__":
    main()
