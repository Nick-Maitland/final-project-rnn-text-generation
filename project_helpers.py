from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

TORCH_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "4"))
torch.set_num_threads(TORCH_THREADS)

DEFAULT_SEED = 42
WORD_RE = re.compile(r"[@#]?\w+(?:'\w+)?|[.,!?;:]")


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_basic_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.lower()
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_shakespeare_text(raw_path: str | Path) -> str:
    text = Path(raw_path).read_text(encoding="utf-8")
    start_marker = "ACT I"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK HAMLET ***"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start == -1:
        start = 0
    if end == -1:
        end = len(text)
    text = text[start:end]
    text = normalize_basic_text(text)
    return text


def clean_social_media_text(csv_path: str | Path, num_rows: int = 3000) -> str:
    df = pd.read_csv(csv_path, encoding="latin1")
    texts = df["Text"].astype(str).tolist()[:num_rows]
    text = " ".join(texts)
    text = normalize_basic_text(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text)


def count_word_tokens(text: str) -> int:
    return len(tokenize_words(text))


def detokenize_word_tokens(tokens: Sequence[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)
    return text.strip()


class NextTokenWordDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 15, stride: int = 4):
        tokens = tokenize_words(text)
        vocab = ["<pad>", "<unk>"] + sorted(set(tokens))
        self.tokens = tokens
        self.vocab = vocab
        self.stoi = {token: idx for idx, token in enumerate(vocab)}
        self.itos = {idx: token for token, idx in self.stoi.items()}
        token_ids = [self.stoi[token] for token in tokens]
        xs, ys = [], []
        for i in range(0, len(token_ids) - seq_len, stride):
            xs.append(token_ids[i : i + seq_len])
            ys.append(token_ids[i + seq_len])
        self.x = torch.tensor(np.array(xs), dtype=torch.long)
        self.y = torch.tensor(np.array(ys), dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class NextCharDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 60, stride: int = 20):
        vocab = sorted(set(text))
        self.text = text
        self.vocab = vocab
        self.stoi = {char: idx for idx, char in enumerate(vocab)}
        self.itos = {idx: char for char, idx in self.stoi.items()}
        char_ids = np.array([self.stoi[char] for char in text], dtype=np.int64)
        xs, ys = [], []
        for i in range(0, len(char_ids) - seq_len, stride):
            xs.append(char_ids[i : i + seq_len])
            ys.append(char_ids[i + seq_len])
        self.x = torch.tensor(np.stack(xs), dtype=torch.long)
        self.y = torch.tensor(np.array(ys), dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class WordLSTMModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out[:, -1, :])
        logits = self.output(out)
        return logits, hidden


class CharGRUModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.gru = nn.GRU(vocab_size, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float()
        out, hidden = self.gru(x, hidden)
        out = self.dropout(out[:, -1, :])
        logits = self.output(out)
        return logits, hidden


def build_model_from_config(config: Dict) -> nn.Module:
    model_type = config["model_type"]
    if model_type == "word_lstm":
        return WordLSTMModel(
            vocab_size=config["vocab_size"],
            emb_dim=config.get("emb_dim", 64),
            hidden_dim=config.get("hidden_dim", 128),
            dropout=config.get("dropout", 0.2),
        )
    if model_type == "char_gru":
        return CharGRUModel(
            vocab_size=config["vocab_size"],
            hidden_dim=config.get("hidden_dim", 64),
            dropout=config.get("dropout", 0.1),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def make_dataloaders(dataset: Dataset, batch_size: int = 64, val_split: float = 0.1, seed: int = DEFAULT_SEED):
    total = len(dataset)
    val_size = max(1, int(total * val_split))
    train_size = total - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str = "cpu") -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            total_examples += xb.size(0)
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
    avg_loss = total_loss / max(total_examples, 1)
    acc = total_correct / max(total_examples, 1)
    perplexity = float(math.exp(min(avg_loss, 20)))
    return {"loss": avg_loss, "accuracy": acc, "perplexity": perplexity}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 0.003,
    patience: int = 2,
    device: str = "cpu",
    verbose: bool = True,
    run_name: str = "model",
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val_loss = float("inf")
    wait = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        total_correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_examples += xb.size(0)
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()

        train_loss = total_loss / max(total_examples, 1)
        train_acc = total_correct / max(total_examples, 1)
        train_perplexity = float(math.exp(min(train_loss, 20)))
        val_metrics = evaluate_model(model, val_loader, criterion, device=device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_perplexity": train_perplexity,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_perplexity": val_metrics["perplexity"],
        }
        history.append(row)

        if verbose:
            print(
                f"  {run_name} epoch {epoch}: train_loss={train_loss:.4f} val_loss={row['val_loss']:.4f} "
                f"train_acc={train_acc:.3f} val_acc={row['val_accuracy']:.3f}",
                flush=True,
            )

        if row["val_loss"] + 1e-6 < best_val_loss:
            best_val_loss = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def save_model_bundle(model: nn.Module, metadata: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_metadata = json.dumps(metadata, ensure_ascii=False)
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("metadata_json", data=np.bytes_(serializable_metadata))
        state_group = h5f.create_group("state_dict")
        for name, tensor in model.state_dict().items():
            state_group.create_dataset(name, data=tensor.detach().cpu().numpy())


def load_model_bundle(path: str | Path, device: str = "cpu") -> Dict:
    path = Path(path)
    with h5py.File(path, "r") as h5f:
        metadata = json.loads(h5f["metadata_json"][()].decode("utf-8"))
        state_dict = {name: torch.tensor(h5f["state_dict"][name][()]) for name in h5f["state_dict"].keys()}
    model = build_model_from_config(metadata["model_config"])
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    vocab = metadata["vocab"]
    stoi = {token: idx for idx, token in enumerate(vocab)}
    itos = {idx: token for token, idx in stoi.items()}
    bundle = {
        "path": str(path),
        "metadata": metadata,
        "model": model,
        "vocab": vocab,
        "stoi": stoi,
        "itos": itos,
    }
    return bundle


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> int:
    logits = logits / max(temperature, 1e-5)
    if top_k is not None and top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.numel()))
        probs = torch.softmax(values, dim=-1)
        sampled_index = torch.multinomial(probs, 1).item()
        return indices[sampled_index].item()
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate_word_text(
    bundle: Dict,
    prompt: str,
    length: int = 40,
    temperature: float = 0.9,
    top_k: int = 10,
    device: str = "cpu",
) -> str:
    model = bundle["model"].to(device)
    metadata = bundle["metadata"]
    stoi = bundle["stoi"]
    itos = bundle["itos"]
    seq_len = metadata["model_config"]["seq_len"]
    tokens = tokenize_words(prompt.lower())
    if not tokens:
        tokens = [metadata.get("default_word_prompt", "the")]
    generated_tokens = list(tokens)
    current_ids = [stoi.get(token, stoi.get("<unk>", 1)) for token in tokens]
    pad_id = stoi.get(metadata.get("default_word_prompt", "the"), 0)
    if len(current_ids) < seq_len:
        current_ids = [pad_id] * (seq_len - len(current_ids)) + current_ids
    else:
        current_ids = current_ids[-seq_len:]

    for _ in range(length):
        xb = torch.tensor([current_ids[-seq_len:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(xb)
        next_id = sample_from_logits(logits[0].cpu(), temperature=temperature, top_k=top_k)
        current_ids.append(next_id)
        generated_tokens.append(itos[next_id])
    return detokenize_word_tokens(generated_tokens)


def generate_char_text(
    bundle: Dict,
    prompt: str,
    length: int = 160,
    temperature: float = 0.8,
    top_k: int = 8,
    device: str = "cpu",
) -> str:
    model = bundle["model"].to(device)
    metadata = bundle["metadata"]
    stoi = bundle["stoi"]
    itos = bundle["itos"]
    seq_len = metadata["model_config"]["seq_len"]
    prompt = prompt.lower()
    if not prompt:
        prompt = metadata.get("default_char_prompt", "love ")
    fallback_char = " " if " " in stoi else metadata["vocab"][0]
    current_ids = [stoi.get(ch, stoi[fallback_char]) for ch in prompt]
    if len(current_ids) < seq_len:
        current_ids = [stoi[fallback_char]] * (seq_len - len(current_ids)) + current_ids
    else:
        current_ids = current_ids[-seq_len:]
    generated_chars = list(prompt)

    for _ in range(length):
        xb = torch.tensor([current_ids[-seq_len:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(xb)
        next_id = sample_from_logits(logits[0].cpu(), temperature=temperature, top_k=top_k)
        current_ids.append(next_id)
        generated_chars.append(itos[next_id])
    return "".join(generated_chars)


def generate_text(
    bundle: Dict,
    prompt: str,
    length: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str = "cpu",
) -> str:
    granularity = bundle["metadata"]["granularity"]
    if granularity == "word":
        if top_k is None:
            top_k = 10
        return generate_word_text(bundle, prompt=prompt, length=length, temperature=temperature, top_k=top_k, device=device)
    if top_k is None:
        top_k = 8
    return generate_char_text(bundle, prompt=prompt, length=length, temperature=temperature, top_k=top_k, device=device)


def summarize_corpus(text: str, granularity: str) -> Dict[str, int]:
    if granularity == "word":
        tokens = tokenize_words(text)
        return {
            "units": len(tokens),
            "vocab_size": len(set(tokens)),
            "characters": len(text),
        }
    return {
        "units": len(text),
        "vocab_size": len(set(text)),
        "characters": len(text),
    }
