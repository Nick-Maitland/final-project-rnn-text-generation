from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import re
import sys
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split

TORCH_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "4"))
torch.set_num_threads(TORCH_THREADS)

DEFAULT_SEED = 42
WORD_RE = re.compile(r"[@#]?\w+(?:'\w+)?|[.,!?;:]")
ROMAN_NUMERALS = {
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
}


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_basic_text(text: str, lower: bool = False) -> str:
    """Light normalization that keeps punctuation and optionally case."""
    text = text.replace("\ufeff", "")
    if lower:
        text = text.lower()
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_shakespeare_text(raw_path: str | Path, lower: bool = False) -> str:
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
    text = normalize_basic_text(text, lower=lower)
    return text


def clean_social_media_text(csv_path: str | Path, num_rows: int | None = None, lower: bool = False) -> str:
    """Read tweets, strip links, keep mentions/hashtags, optional row cap."""
    texts: List[str] = []
    with open(csv_path, newline="", encoding="latin1") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if num_rows is not None and i >= num_rows:
                break
            txt = str(row.get("Text", ""))
            texts.append(txt)
    text = " ".join(texts)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = normalize_basic_text(text, lower=lower)
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


def normalize_word_token(token: str, profile: str = "none") -> str:
    if profile == "none" or token in {".", ",", "!", "?", ";", ":"}:
        return token

    if profile == "social_strict":
        if token.startswith("@") and len(token) > 1:
            return "@USER"
        if token.startswith("#"):
            tag_body = token[1:]
            return f"#{tag_body.lower()}" if tag_body else "#"
        if any(ch.isdigit() for ch in token):
            return "<NUM>"
        if token.isalpha() and token.isupper() and len(token) > 3:
            return token.lower()
        return token

    if profile == "shakespeare_strict":
        if token.isalpha() and token.isupper() and len(token) > 3 and token not in ROMAN_NUMERALS:
            return token.lower()
        return token

    raise ValueError(f"Unsupported normalization profile: {profile}")


def normalize_word_tokens(tokens: Sequence[str], profile: str = "none") -> List[str]:
    return [normalize_word_token(token, profile=profile) for token in tokens]


def sentencepiece_install_message() -> str:
    return "sentencepiece is required for subword models. Install it with `pip install sentencepiece`."


def load_sentencepiece_module():
    try:
        import sentencepiece as spm  # type: ignore
    except Exception as exc:
        raise RuntimeError(sentencepiece_install_message()) from exc
    return spm


def load_sentencepiece_processor(model_path: str | Path):
    spm = load_sentencepiece_module()
    return spm.SentencePieceProcessor(model_file=str(model_path))


class NextTokenWordDataset(Dataset):
    def __init__(
        self,
        text: str,
        seq_len: int = 30,
        stride: int = 5,
        min_freq: int = 3,
        tokenizer_type: str = "word",
        normalization_profile: str = "none",
        sentencepiece_model: str | Path | None = None,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.tokenizer_type = tokenizer_type
        self.normalization_profile = normalization_profile
        self.sp = None
        self.token_count = 0
        self.coverage_stats = {
            "tokenizer_type": tokenizer_type,
            "normalization_profile": normalization_profile,
            "token_count": 0,
            "known_token_rate": 1.0,
            "unk_token_rate": 0.0,
        }

        if tokenizer_type == "sentencepiece":
            if sentencepiece_model is None:
                raise ValueError("sentencepiece_model is required when tokenizer_type='sentencepiece'")
            self.sp = load_sentencepiece_processor(sentencepiece_model)
            self.vocab = [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]
            self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
            self.itos = {i: tok for tok, i in self.stoi.items()}
            token_ids = self.sp.encode(text, out_type=int)
            self.token_count = len(token_ids)
            unk_piece = self.sp.id_to_piece(self.sp.unk_id()) if self.sp.unk_id() >= 0 else "<unk>"
            unk_count = sum(1 for token_id in token_ids if self.sp.id_to_piece(token_id) == unk_piece)
            known_rate = 1.0 - (unk_count / max(self.token_count, 1))
            self.coverage_stats = {
                "tokenizer_type": tokenizer_type,
                "normalization_profile": normalization_profile,
                "token_count": self.token_count,
                "known_token_rate": known_rate,
                "unk_token_rate": 1.0 - known_rate,
            }
        else:
            tokens = normalize_word_tokens(tokenize_words(text), profile=normalization_profile)
            freqs = Counter(tokens)
            vocab_tokens = [tok for tok, c in freqs.items() if c >= min_freq]
            vocab_tokens.sort()
            vocab = ["<pad>", "<unk>"] + vocab_tokens
            stoi = {token: idx for idx, token in enumerate(vocab)}
            unk_idx = stoi["<unk>"]
            token_ids = [stoi.get(token, unk_idx) for token in tokens]
            unk_count = sum(1 for token_id in token_ids if token_id == unk_idx)
            self.token_count = len(tokens)
            known_rate = 1.0 - (unk_count / max(self.token_count, 1))
            self.vocab = vocab
            self.stoi = stoi
            self.itos = {idx: token for token, idx in stoi.items()}
            self.coverage_stats = {
                "tokenizer_type": tokenizer_type,
                "normalization_profile": normalization_profile,
                "token_count": self.token_count,
                "known_token_rate": known_rate,
                "unk_token_rate": 1.0 - known_rate,
            }

        xs, ys = [], []
        for i in range(0, len(token_ids) - seq_len, stride):
            xs.append(token_ids[i : i + seq_len])
            ys.append(token_ids[i + seq_len])
        self.x = torch.tensor(np.array(xs), dtype=torch.long)
        self.y = torch.tensor(np.array(ys), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class NextCharDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 120, stride: int = 30):
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
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Identity() if hidden_dim == emb_dim else nn.Linear(hidden_dim, emb_dim)
        self.output = nn.Linear(emb_dim, vocab_size, bias=False)
        # Weight tying
        self.output.weight = self.embedding.weight

    def forward(self, x: torch.Tensor, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out[:, -1, :])
        out = self.proj(out)
        logits = self.output(out)
        return logits, hidden


class CharGRUModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 256, dropout: float = 0.25, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        out = self.dropout(out[:, -1, :])
        logits = self.output(out)
        return logits, hidden


def build_model_from_config(config: Dict) -> nn.Module:
    model_type = config["model_type"]
    if model_type == "word_lstm":
        return WordLSTMModel(
            vocab_size=config["vocab_size"],
            emb_dim=config.get("emb_dim", 128),
            hidden_dim=config.get("hidden_dim", 256),
            dropout=config.get("dropout", 0.3),
            num_layers=config.get("num_layers", 2),
        )
    if model_type == "char_gru":
        return CharGRUModel(
            vocab_size=config["vocab_size"],
            emb_dim=config.get("emb_dim", 64),
            hidden_dim=config.get("hidden_dim", 256),
            dropout=config.get("dropout", 0.25),
            num_layers=config.get("num_layers", 2),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def resolve_num_workers(requested: int) -> int:
    override = os.environ.get("PROJECT_NUM_WORKERS")
    if override is not None:
        try:
            return max(0, int(override))
        except ValueError:
            pass

    if requested <= 0:
        return 0

    # Multiprocess DataLoaders tend to cost more RAM than they save on Apple Silicon laptops.
    if sys.platform == "darwin":
        if torch.backends.mps.is_available():
            return 0
        return min(requested, 1)

    cpu_count = os.cpu_count() or 1
    return min(requested, max(1, cpu_count // 2))


def resolve_pin_memory(requested: bool) -> bool:
    # Pinned host memory is primarily useful for CUDA transfers.
    return bool(requested and torch.cuda.is_available())


def clear_accelerator_cache(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def is_worker_startup_fallback_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return (
        "torch_shm_manager" in message
        or "_share_filename_cpu_" in message
        or "unable to open shared memory object" in message.lower()
    )


def clone_dataloader_single_process(loader: DataLoader) -> DataLoader:
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=isinstance(loader.sampler, RandomSampler),
        num_workers=0,
        pin_memory=False,
        drop_last=loader.drop_last,
        collate_fn=loader.collate_fn,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        generator=loader.generator,
    )


def ensure_dataloader_runtime_compatible(loader: DataLoader, loader_name: str) -> DataLoader:
    if loader.num_workers <= 0:
        return loader

    try:
        iterator = iter(loader)
        del iterator
        return loader
    except RuntimeError as exc:
        if not is_worker_startup_fallback_error(exc):
            raise
        print(
            f"  {loader_name}: DataLoader worker startup failed ({str(exc).splitlines()[0]}). "
            "Retrying with num_workers=0.",
            flush=True,
        )
        return clone_dataloader_single_process(loader)


def make_dataloaders(
    dataset: Dataset,
    batch_size: int = 64,
    splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = DEFAULT_SEED,
    num_workers: int = 2,
    pin_memory: bool = False,
):
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1"
    total = len(dataset)
    val_size = max(1, int(total * splits[1]))
    test_size = max(1, int(total * splits[2]))
    train_size = total - val_size - test_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    effective_num_workers = resolve_num_workers(num_workers)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": effective_num_workers,
        "pin_memory": resolve_pin_memory(pin_memory),
    }
    if effective_num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str = "cpu") -> Dict[str, float]:
    model.eval()
    loader = ensure_dataloader_runtime_compatible(loader, "eval_loader")
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            total_examples += xb.size(0)
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            del logits, loss, xb, yb
    avg_loss = total_loss / max(total_examples, 1)
    acc = total_correct / max(total_examples, 1)
    perplexity = float(math.exp(min(avg_loss, 20)))
    return {"loss": avg_loss, "accuracy": acc, "perplexity": perplexity}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 0.002,
    weight_decay: float = 0.01,
    patience: int = 3,
    min_delta: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
    run_name: str = "model",
    grad_clip: float = 1.0,
    use_amp: bool = False,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    model = model.to(device)
    train_loader = ensure_dataloader_runtime_compatible(train_loader, f"{run_name} train_loader")
    val_loader = ensure_dataloader_runtime_compatible(val_loader, f"{run_name} val_loader")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=2e-4
    )
    best_state = None
    best_val_loss = float("inf")
    wait = 0
    history: List[Dict[str, float]] = []

    autocast_device = None
    if use_amp and device in {"cuda", "mps"}:
        autocast_device = device

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        total_correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, dtype=torch.float16) if autocast_device else nullcontext():
                logits, _ = model(xb)
                loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_examples += xb.size(0)
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            del logits, loss, xb, yb

        train_loss = total_loss / max(total_examples, 1)
        train_acc = total_correct / max(total_examples, 1)
        train_perplexity = float(math.exp(min(train_loss, 20)))
        val_metrics = evaluate_model(model, val_loader, criterion, device=device)
        scheduler.step(val_metrics["loss"])
        if autocast_device:
            gc.collect()
            clear_accelerator_cache(device)

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

        if best_val_loss - row["val_loss"] > min_delta:
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
    metadata_bytes = serializable_metadata.encode("utf-8")
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("metadata_json", data=np.frombuffer(metadata_bytes, dtype=np.uint8))
        state_group = h5f.create_group("state_dict")
        for name, tensor in model.state_dict().items():
            state_group.create_dataset(name, data=tensor.detach().cpu().numpy())


def load_model_bundle(path: str | Path, device: str = "cpu") -> Dict:
    path = Path(path)
    with h5py.File(path, "r") as h5f:
        raw_metadata = h5f["metadata_json"][()]
        if isinstance(raw_metadata, np.ndarray):
            metadata_json = raw_metadata.tobytes().decode("utf-8")
        else:
            metadata_json = raw_metadata.decode("utf-8")
        metadata = json.loads(metadata_json)
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
    tokenizer_type = metadata.get("tokenizer_type", "word")
    if tokenizer_type == "sentencepiece":
        tokenizer_relpath = metadata.get("tokenizer_model_relpath")
        if not tokenizer_relpath:
            raise RuntimeError(f"Subword bundle at {path} is missing tokenizer_model_relpath metadata.")
        tokenizer_path = path.parent / tokenizer_relpath
        bundle["tokenizer"] = load_sentencepiece_processor(tokenizer_path)
        bundle["tokenizer_path"] = str(tokenizer_path)
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
    normalization_profile = metadata.get("normalization_profile", "none")
    tokens = normalize_word_tokens(tokenize_words(prompt), profile=normalization_profile)
    if not tokens:
        default_prompt = metadata.get("default_word_prompt", "the")
        tokens = normalize_word_tokens(tokenize_words(default_prompt), profile=normalization_profile)
        if not tokens:
            tokens = [default_prompt]
    generated_tokens = list(tokens)
    current_ids = [stoi.get(token, stoi.get("<unk>", 1)) for token in tokens]
    default_prompt_tokens = normalize_word_tokens(
        tokenize_words(metadata.get("default_word_prompt", "the")),
        profile=normalization_profile,
    )
    pad_token = default_prompt_tokens[0] if default_prompt_tokens else metadata.get("default_word_prompt", "the")
    pad_id = stoi.get(pad_token, 0)
    if len(current_ids) < seq_len:
        current_ids = [pad_id] * (seq_len - len(current_ids)) + current_ids
    else:
        current_ids = current_ids[-seq_len:]

    for _ in range(length):
        xb = torch.tensor([current_ids[-seq_len:]], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits, _ = model(xb)
        next_id = sample_from_logits(logits[0].cpu(), temperature=temperature, top_k=top_k)
        current_ids.append(next_id)
        generated_tokens.append(itos[next_id])
        del logits, xb
    return detokenize_word_tokens(generated_tokens)


def generate_subword_text(
    bundle: Dict,
    prompt: str,
    length: int = 80,
    temperature: float = 0.9,
    top_k: int = 16,
    device: str = "cpu",
) -> str:
    model = bundle["model"].to(device)
    metadata = bundle["metadata"]
    tokenizer = bundle.get("tokenizer")
    if tokenizer is None:
        raise RuntimeError("Subword bundle is missing a loaded tokenizer.")

    seq_len = metadata["model_config"]["seq_len"]
    if not prompt:
        prompt = metadata.get("default_word_prompt", "love ")
    prompt_ids = tokenizer.encode(prompt, out_type=int)
    if not prompt_ids:
        prompt = metadata.get("default_word_prompt", "love ")
        prompt_ids = tokenizer.encode(prompt, out_type=int)
    if not prompt_ids:
        fallback_id = tokenizer.unk_id() if tokenizer.unk_id() >= 0 else 0
        prompt_ids = [fallback_id]

    generated_ids = list(prompt_ids)
    pad_id = prompt_ids[0]
    current_ids = list(prompt_ids)
    if len(current_ids) < seq_len:
        current_ids = [pad_id] * (seq_len - len(current_ids)) + current_ids
    else:
        current_ids = current_ids[-seq_len:]

    for _ in range(length):
        xb = torch.tensor([current_ids[-seq_len:]], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits, _ = model(xb)
        next_id = sample_from_logits(logits[0].cpu(), temperature=temperature, top_k=top_k)
        current_ids.append(next_id)
        generated_ids.append(next_id)
        del logits, xb
    return tokenizer.decode(generated_ids)


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
        with torch.inference_mode():
            logits, _ = model(xb)
        next_id = sample_from_logits(logits[0].cpu(), temperature=temperature, top_k=top_k)
        current_ids.append(next_id)
        generated_chars.append(itos[next_id])
        del logits, xb
    return "".join(generated_chars)


def generate_text(
    bundle: Dict,
    prompt: str,
    length: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str = "cpu",
) -> str:
    metadata = bundle["metadata"]
    granularity = metadata["granularity"]
    tokenizer_type = metadata.get("tokenizer_type", "word")
    if granularity == "subword" or tokenizer_type == "sentencepiece":
        if top_k is None:
            top_k = 16
        return generate_subword_text(bundle, prompt=prompt, length=length, temperature=temperature, top_k=top_k, device=device)
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
