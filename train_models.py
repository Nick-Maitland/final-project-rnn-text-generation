from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import torch

from project_helpers import (
    DEFAULT_SEED,
    NextCharDataset,
    NextTokenWordDataset,
    WordLSTMModel,
    CharGRUModel,
    clean_shakespeare_text,
    clean_social_media_text,
    count_word_tokens,
    evaluate_model,
    generate_text,
    load_model_bundle,
    load_sentencepiece_module,
    make_dataloaders,
    save_model_bundle,
    set_seed,
    train_model,
)

BASE_DIR = Path('/Users/nicolasmaitland/Library/CloudStorage/OneDrive-DurhamCollege/Agile/Final Project/final_project_submission/technical')
MODELS_DIR = BASE_DIR / 'models'
DATA_RAW_DIR = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
ARTIFACTS_DIR = BASE_DIR / 'artifacts'

RAW_HAMLET = DATA_RAW_DIR / 'hamlet_gutenberg_raw.txt'
RAW_TWITTER = DATA_RAW_DIR / 'twittersubset_public_tweets.csv'
SOCIAL_MEDIA_ROW_LIMIT = 3000
PRIMARY_MODEL_NAMES = ('shakespeare_word', 'shakespeare_char', 'social_word', 'social_char')
EXTENSION_MODEL_NAMES = ('shakespeare_subword', 'social_subword')
MODEL_NAMES = PRIMARY_MODEL_NAMES + EXTENSION_MODEL_NAMES

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be a positive integer')
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be a positive number')
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError('value must be a non-negative number')
    return parsed


def dropout_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError('dropout must be in the range [0, 1)')
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train RNN text-generation models for the final project.')
    parser.add_argument(
        '--models',
        nargs='+',
        choices=MODEL_NAMES,
        help='Optional subset of model names to train. Defaults to all six models.',
    )
    parser.add_argument(
        '--social-row-limit',
        type=positive_int,
        default=SOCIAL_MEDIA_ROW_LIMIT,
        help=f'Number of social-media rows to load. Default: {SOCIAL_MEDIA_ROW_LIMIT}.',
    )
    parser.add_argument(
        '--epochs-override',
        type=positive_int,
        help='Optional epoch override applied to every selected model for pilot tuning runs.',
    )
    parser.add_argument(
        '--batch-size-override',
        type=positive_int,
        help='Optional batch-size override applied to every selected model for pilot tuning runs.',
    )
    parser.add_argument(
        '--seq-len-override',
        type=positive_int,
        help='Optional sequence-length override applied to every selected model for pilot tuning runs.',
    )
    parser.add_argument(
        '--hidden-dim-override',
        type=positive_int,
        help='Optional hidden-dimension override applied to every selected model for pilot tuning runs.',
    )
    parser.add_argument(
        '--emb-dim-override',
        type=positive_int,
        help='Optional embedding-dimension override applied to every selected model for pilot tuning runs.',
    )
    parser.add_argument(
        '--dropout-override',
        type=dropout_float,
        help='Optional dropout override applied to every selected model for pilot tuning runs.',
    )
    parser.add_argument(
        '--num-layers-override',
        type=positive_int,
        help='Optional recurrent-layer override applied to every selected model for pilot tuning runs.',
    )
    parser.add_argument(
        '--stride-override',
        type=positive_int,
        help='Optional dataset stride override applied to every selected model for pilot tuning runs.',
    )
    parser.add_argument(
        '--lr-override',
        type=positive_float,
        help='Optional learning-rate override applied to every selected model for pilot tuning runs.',
    )
    parser.add_argument(
        '--weight-decay-override',
        type=nonnegative_float,
        help='Optional weight-decay override applied to every selected model for pilot tuning runs.',
    )
    parser.add_argument(
        '--patience-override',
        type=positive_int,
        help='Optional early-stopping patience override applied to every selected model for pilot tuning runs.',
    )
    parser.add_argument(
        '--min-delta-override',
        type=nonnegative_float,
        help='Optional early-stopping min-delta override applied to every selected model for pilot tuning runs.',
    )
    parser.add_argument(
        '--word-min-freq-override',
        type=positive_int,
        help='Optional minimum token frequency override applied to selected word models for pilot tuning runs.',
    )
    parser.add_argument(
        '--sp-vocab-size-override',
        type=positive_int,
        help='Optional SentencePiece vocab-size override applied to selected subword models for pilot tuning runs.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the resolved training plan and exit without training or writing artifacts.',
    )
    return parser.parse_args()


def all_models_selected(selected_models: list[str] | None) -> bool:
    if selected_models is None:
        return True
    return len(selected_models) == len(MODEL_NAMES) and set(selected_models) == set(MODEL_NAMES)


def is_submission_run(args: argparse.Namespace) -> bool:
    return (
        all_models_selected(args.models)
        and args.social_row_limit == SOCIAL_MEDIA_ROW_LIMIT
        and args.epochs_override is None
        and args.batch_size_override is None
        and args.seq_len_override is None
        and args.hidden_dim_override is None
        and args.emb_dim_override is None
        and args.dropout_override is None
        and args.num_layers_override is None
        and args.stride_override is None
        and args.lr_override is None
        and args.weight_decay_override is None
        and args.patience_override is None
        and args.min_delta_override is None
        and args.word_min_freq_override is None
        and args.sp_vocab_size_override is None
    )


def format_float_label(value: float) -> str:
    return f"{value:.4g}".replace('.', 'p')


def build_config_label(args: argparse.Namespace) -> str:
    selected_models = args.models or list(MODEL_NAMES)
    parts = ["-".join(selected_models)]
    if args.social_row_limit != SOCIAL_MEDIA_ROW_LIMIT:
        parts.append(f"rows{args.social_row_limit}")
    if args.epochs_override is not None:
        parts.append(f"ep{args.epochs_override}")
    if args.batch_size_override is not None:
        parts.append(f"bs{args.batch_size_override}")
    if args.seq_len_override is not None:
        parts.append(f"seq{args.seq_len_override}")
    if args.hidden_dim_override is not None:
        parts.append(f"hd{args.hidden_dim_override}")
    if args.emb_dim_override is not None:
        parts.append(f"emb{args.emb_dim_override}")
    if args.dropout_override is not None:
        parts.append(f"dr{format_float_label(args.dropout_override)}")
    if args.num_layers_override is not None:
        parts.append(f"layers{args.num_layers_override}")
    if args.stride_override is not None:
        parts.append(f"stride{args.stride_override}")
    if args.lr_override is not None:
        parts.append(f"lr{format_float_label(args.lr_override)}")
    if args.weight_decay_override is not None:
        parts.append(f"wd{format_float_label(args.weight_decay_override)}")
    if args.patience_override is not None:
        parts.append(f"pat{args.patience_override}")
    if args.min_delta_override is not None:
        parts.append(f"md{format_float_label(args.min_delta_override)}")
    if args.word_min_freq_override is not None:
        parts.append(f"mf{args.word_min_freq_override}")
    if args.sp_vocab_size_override is not None:
        parts.append(f"sp{args.sp_vocab_size_override}")
    return '_'.join(parts)


def build_run_name(args: argparse.Namespace) -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{build_config_label(args)}"


def resolve_run_paths(args: argparse.Namespace, create_dirs: bool) -> dict[str, Path | bool]:
    if is_submission_run(args):
        return {
            'root_dir': BASE_DIR,
            'models_dir': MODELS_DIR,
            'artifacts_dir': ARTIFACTS_DIR,
            'processed_data_dir': DATA_PROCESSED_DIR,
            'dataset_sources_path': BASE_DIR / 'dataset_sources.md',
            'readme_path': BASE_DIR / 'README.md',
            'submission_run': True,
        }

    run_root = ARTIFACTS_DIR / 'runs' / build_run_name(args)
    models_dir = run_root / 'models'
    artifacts_dir = run_root / 'artifacts'
    processed_data_dir = run_root / 'data' / 'processed'
    if create_dirs:
        models_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        processed_data_dir.mkdir(parents=True, exist_ok=True)
    return {
        'root_dir': run_root,
        'models_dir': models_dir,
        'artifacts_dir': artifacts_dir,
        'processed_data_dir': processed_data_dir,
        'dataset_sources_path': run_root / 'dataset_sources.md',
        'readme_path': None,
        'submission_run': False,
    }


def build_corpora(
    social_row_limit: int = SOCIAL_MEDIA_ROW_LIMIT,
    processed_data_dir: Path | None = DATA_PROCESSED_DIR,
):
    shakespeare_text = clean_shakespeare_text(RAW_HAMLET, lower=False)
    # Keep the social corpus within the assignment size target while preserving tweet order.
    social_text = clean_social_media_text(RAW_TWITTER, num_rows=social_row_limit, lower=False)

    if processed_data_dir is not None:
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        (processed_data_dir / 'shakespeare_hamlet_clean.txt').write_text(shakespeare_text, encoding='utf-8')
        (processed_data_dir / 'social_media_tweets_clean.txt').write_text(social_text, encoding='utf-8')
    return shakespeare_text, social_text


def maybe_train_sentencepiece_model(
    text: str,
    model_path: Path,
    vocab_size: int = 2048,
    force_retrain: bool = False,
) -> Path:
    spm = load_sentencepiece_module()
    if model_path.exists() and not force_retrain:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    if force_retrain:
        vocab_path = model_path.with_suffix('.vocab')
        if model_path.exists():
            model_path.unlink()
        if vocab_path.exists():
            vocab_path.unlink()
    temp_input = model_path.parent / f'{model_path.stem}_input.txt'
    chunks = []
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > 3500:
            split_at = sentence.rfind(' ', 0, 3500)
            if split_at <= 0:
                split_at = 3500
            chunks.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()
        if sentence:
            chunks.append(sentence)
    temp_input.write_text("\n".join(chunks), encoding='utf-8')
    try:
        spm.SentencePieceTrainer.train(
            input=str(temp_input),
            model_prefix=str(model_path.with_suffix('')),
            vocab_size=vocab_size,
            model_type='unigram',
            character_coverage=1.0,
            input_sentence_size=200000,
            shuffle_input_sentence=True,
            max_sentence_length=4096,
        )
    finally:
        if temp_input.exists():
            temp_input.unlink()
    return model_path


def existing_sentencepiece_vocab_size(model_path: Path) -> int | None:
    if not model_path.exists():
        return None
    spm = load_sentencepiece_module()
    processor = spm.SentencePieceProcessor()
    processor.load(str(model_path))
    return int(processor.get_piece_size())


def model_specs():
    return {
        'shakespeare_word': {
            'corpus_label': 'Shakespeare (Hamlet)',
            'granularity': 'word',
            'corpus_key': 'shakespeare',
            'tokenizer_type': 'word',
            'normalization_profile': 'shakespeare_strict',
            'comparison_role': 'primary',
            'dataset_config': {
                'dataset_type': 'word',
                'seq_len': 40,
                'stride': 5,
                'min_freq': 2,
            },
            'epochs': 8,
            'lr': 0.0008,
            'weight_decay': 0.002,
            'patience': 3,
            'min_delta': 5e-4,
            'batch_size': 64,
            'model_config': {
                'model_type': 'word_lstm',
                'seq_len': 40,
                'vocab_size': None,
                'emb_dim': 192,
                'hidden_dim': 224,
                'dropout': 0.25,
                'num_layers': 2,
            },
            'default_prompt': 'love',
            'sample_length': 50,
            'temperatures': [0.75, 0.9, 1.05],
            'top_k': 12,
        },
        'shakespeare_char': {
            'corpus_label': 'Shakespeare (Hamlet)',
            'granularity': 'char',
            'corpus_key': 'shakespeare',
            'tokenizer_type': 'char',
            'normalization_profile': 'none',
            'comparison_role': 'primary',
            'dataset_config': {
                'dataset_type': 'char',
                'seq_len': 120,
                'stride': 30,
            },
            'epochs': 8,
            'lr': 0.0015,
            'weight_decay': 0.005,
            'patience': 3,
            'min_delta': 5e-4,
            'batch_size': 128,
            'model_config': {
                'model_type': 'char_gru',
                'seq_len': 120,
                'vocab_size': None,
                'emb_dim': 64,
                'hidden_dim': 256,
                'dropout': 0.25,
                'num_layers': 2,
            },
            'default_prompt': 'love ',
            'sample_length': 240,
            'temperatures': [0.65, 0.8, 0.95],
            'top_k': 8,
        },
        'social_word': {
            'corpus_label': 'Social media tweets',
            'granularity': 'word',
            'corpus_key': 'social_media',
            'tokenizer_type': 'word',
            'normalization_profile': 'social_strict',
            'comparison_role': 'primary',
            'dataset_config': {
                'dataset_type': 'word',
                'seq_len': 24,
                'stride': 5,
                'min_freq': 2,
            },
            'epochs': 6,
            'lr': 0.001,
            'weight_decay': 0.01,
            'patience': 2,
            'min_delta': 1e-3,
            'batch_size': 64,
            'model_config': {
                'model_type': 'word_lstm',
                'seq_len': 24,
                'vocab_size': None,
                'emb_dim': 128,
                'hidden_dim': 128,
                'dropout': 0.3,
                'num_layers': 2,
            },
            'default_prompt': 'love',
            'sample_length': 50,
            'temperatures': [0.75, 0.9, 1.05],
            'top_k': 12,
        },
        'social_char': {
            'corpus_label': 'Social media tweets',
            'granularity': 'char',
            'corpus_key': 'social_media',
            'tokenizer_type': 'char',
            'normalization_profile': 'none',
            'comparison_role': 'primary',
            'dataset_config': {
                'dataset_type': 'char',
                'seq_len': 128,
                'stride': 32,
            },
            'epochs': 8,
            'lr': 0.0015,
            'weight_decay': 0.005,
            'patience': 3,
            'min_delta': 5e-4,
            'batch_size': 128,
            'model_config': {
                'model_type': 'char_gru',
                'seq_len': 128,
                'vocab_size': None,
                'emb_dim': 96,
                'hidden_dim': 256,
                'dropout': 0.2,
                'num_layers': 2,
            },
            'default_prompt': 'love ',
            'sample_length': 240,
            'temperatures': [0.65, 0.8, 0.95],
            'top_k': 8,
        },
        'shakespeare_subword': {
            'corpus_label': 'Shakespeare (Hamlet)',
            'granularity': 'subword',
            'corpus_key': 'shakespeare',
            'tokenizer_type': 'sentencepiece',
            'normalization_profile': 'sentencepiece_unigram',
            'comparison_role': 'extension',
            'dataset_config': {
                'dataset_type': 'sentencepiece',
                'seq_len': 64,
                'stride': 8,
                'sp_vocab_size': 1024,
            },
            'epochs': 10,
            'lr': 0.0008,
            'weight_decay': 0.002,
            'patience': 3,
            'min_delta': 5e-4,
            'batch_size': 64,
            'model_config': {
                'model_type': 'word_lstm',
                'seq_len': 64,
                'vocab_size': None,
                'emb_dim': 192,
                'hidden_dim': 256,
                'dropout': 0.2,
                'num_layers': 2,
            },
            'default_prompt': 'love ',
            'sample_length': 80,
            'temperatures': [0.7, 0.9, 1.1],
            'top_k': 16,
        },
        'social_subword': {
            'corpus_label': 'Social media tweets',
            'granularity': 'subword',
            'corpus_key': 'social_media',
            'tokenizer_type': 'sentencepiece',
            'normalization_profile': 'sentencepiece_unigram',
            'comparison_role': 'extension',
            'dataset_config': {
                'dataset_type': 'sentencepiece',
                'seq_len': 64,
                'stride': 8,
                'sp_vocab_size': 1024,
            },
            'epochs': 10,
            'lr': 0.0008,
            'weight_decay': 0.002,
            'patience': 3,
            'min_delta': 5e-4,
            'batch_size': 64,
            'model_config': {
                'model_type': 'word_lstm',
                'seq_len': 64,
                'vocab_size': None,
                'emb_dim': 192,
                'hidden_dim': 256,
                'dropout': 0.2,
                'num_layers': 2,
            },
            'default_prompt': 'love ',
            'sample_length': 80,
            'temperatures': [0.7, 0.9, 1.1],
            'top_k': 16,
        },
    }


def resolve_specs(
    selected_models: list[str] | None,
    epochs_override: int | None,
    batch_size_override: int | None,
    seq_len_override: int | None,
    hidden_dim_override: int | None,
    emb_dim_override: int | None,
    dropout_override: float | None,
    num_layers_override: int | None,
    stride_override: int | None,
    lr_override: float | None,
    weight_decay_override: float | None,
    patience_override: int | None,
    min_delta_override: float | None,
    word_min_freq_override: int | None,
    sp_vocab_size_override: int | None,
):
    specs = model_specs()
    model_names = selected_models or list(MODEL_NAMES)
    resolved = {}
    for model_name in model_names:
        spec = copy.deepcopy(specs[model_name])
        if epochs_override is not None:
            spec['epochs'] = epochs_override
        if batch_size_override is not None:
            spec['batch_size'] = batch_size_override
        if seq_len_override is not None:
            spec['dataset_config']['seq_len'] = seq_len_override
            spec['model_config']['seq_len'] = seq_len_override
        if hidden_dim_override is not None:
            spec['model_config']['hidden_dim'] = hidden_dim_override
        if emb_dim_override is not None:
            spec['model_config']['emb_dim'] = emb_dim_override
        if dropout_override is not None:
            spec['model_config']['dropout'] = dropout_override
        if num_layers_override is not None:
            spec['model_config']['num_layers'] = num_layers_override
        if stride_override is not None:
            spec['dataset_config']['stride'] = stride_override
        if lr_override is not None:
            spec['lr'] = lr_override
        if weight_decay_override is not None:
            spec['weight_decay'] = weight_decay_override
        if patience_override is not None:
            spec['patience'] = patience_override
        if min_delta_override is not None:
            spec['min_delta'] = min_delta_override
        if word_min_freq_override is not None and spec['granularity'] == 'word':
            spec['dataset_config']['min_freq'] = word_min_freq_override
        if sp_vocab_size_override is not None and spec['tokenizer_type'] == 'sentencepiece':
            spec['dataset_config']['sp_vocab_size'] = sp_vocab_size_override
        resolved[model_name] = spec
    return resolved


def tokenizer_model_path(model_name: str, models_dir: Path) -> Path:
    return models_dir / f'{model_name}_tokenizer.model'


def tokenizer_vocab_path(model_name: str, models_dir: Path) -> Path:
    return models_dir / f'{model_name}_tokenizer.vocab'


def build_tokenizer_assets(
    specs: dict[str, dict],
    shakespeare_text: str,
    social_text: str,
    models_dir: Path,
    dry_run: bool,
) -> dict[str, dict]:
    corpus_text = {
        'shakespeare': shakespeare_text,
        'social_media': social_text,
    }
    assets: dict[str, dict] = {}
    for model_name, spec in specs.items():
        if spec['tokenizer_type'] != 'sentencepiece':
            continue
        model_path = tokenizer_model_path(model_name, models_dir)
        vocab_path = tokenizer_vocab_path(model_name, models_dir)
        asset = {
            'model_path': model_path,
            'vocab_path': vocab_path,
            'tokenizer_type': spec['tokenizer_type'],
            'corpus_key': spec['corpus_key'],
            'normalization_profile': spec['normalization_profile'],
            'vocab_size': spec['dataset_config']['sp_vocab_size'],
        }
        existing_vocab_size = existing_sentencepiece_vocab_size(model_path)
        asset['ready'] = existing_vocab_size == spec['dataset_config']['sp_vocab_size']
        if not dry_run and not asset['ready']:
            maybe_train_sentencepiece_model(
                corpus_text[spec['corpus_key']],
                model_path,
                vocab_size=spec['dataset_config']['sp_vocab_size'],
                force_retrain=True,
            )
            asset['ready'] = True
        assets[model_name] = asset
    return assets


def build_dataset_for_spec(
    model_name: str,
    spec: dict,
    shakespeare_text: str,
    social_text: str,
    tokenizer_assets: dict[str, dict],
):
    source_text = shakespeare_text if spec['corpus_key'] == 'shakespeare' else social_text
    dataset_config = spec['dataset_config']
    if dataset_config['dataset_type'] == 'word':
        return NextTokenWordDataset(
            source_text,
            seq_len=dataset_config['seq_len'],
            stride=dataset_config['stride'],
            min_freq=dataset_config['min_freq'],
            tokenizer_type='word',
            normalization_profile=spec['normalization_profile'],
        )
    if dataset_config['dataset_type'] == 'sentencepiece':
        asset = tokenizer_assets.get(model_name)
        if asset is None:
            raise RuntimeError(f'Missing tokenizer asset for {model_name}.')
        return NextTokenWordDataset(
            source_text,
            seq_len=dataset_config['seq_len'],
            stride=dataset_config['stride'],
            tokenizer_type='sentencepiece',
            normalization_profile=spec['normalization_profile'],
            sentencepiece_model=asset['model_path'],
        )
    return NextCharDataset(
        source_text,
        seq_len=dataset_config['seq_len'],
        stride=dataset_config['stride'],
    )


def print_run_plan(
    specs: dict[str, dict],
    social_row_limit: int,
    shakespeare_text: str,
    social_text: str,
    tokenizer_assets: dict[str, dict],
) -> None:
    print('Dry run: resolved training plan')
    print(f'  social_row_limit={social_row_limit}')
    print(f'  shakespeare_words={count_word_tokens(shakespeare_text)}')
    print(f'  social_words={count_word_tokens(social_text)}')
    for model_name, spec in specs.items():
        dataset = None
        coverage = None
        if spec['tokenizer_type'] != 'sentencepiece' or tokenizer_assets.get(model_name, {}).get('ready', False):
            dataset = build_dataset_for_spec(model_name, spec, shakespeare_text, social_text, tokenizer_assets)
            coverage = dataset.coverage_stats if hasattr(dataset, 'coverage_stats') else None
        print(
            f"  {model_name}: epochs={spec['epochs']} batch_size={spec['batch_size']} "
            f"seq_len={spec['model_config']['seq_len']} stride={spec['dataset_config'].get('stride', '-')} "
            f"emb_dim={spec['model_config'].get('emb_dim', '-')} "
            f"hidden_dim={spec['model_config']['hidden_dim']} layers={spec['model_config'].get('num_layers', '-')} "
            f"dropout={spec['model_config'].get('dropout', '-')} lr={spec['lr']} "
            f"weight_decay={spec.get('weight_decay', '-')} patience={spec.get('patience', '-')} "
            f"min_delta={spec.get('min_delta', '-')} tokenizer={spec['tokenizer_type']} "
            f"norm={spec['normalization_profile']} min_freq={spec['dataset_config'].get('min_freq', '-')}"
            f" sp_vocab={spec['dataset_config'].get('sp_vocab_size', '-')}"
            f" sequences={len(dataset) if dataset is not None else 'pending-tokenizer'} "
            f"vocab={len(dataset.vocab) if dataset is not None else 'pending-tokenizer'}"
        )
        if coverage is not None:
            print(
                f"    known_token_rate={coverage['known_token_rate']:.4f} "
                f"unk_token_rate={coverage['unk_token_rate']:.4f}"
            )
        if dataset is None and spec['tokenizer_type'] == 'sentencepiece':
            print(f"    tokenizer_asset={tokenizer_assets[model_name]['model_path']}")


def count_sample_unk_tokens(samples: list[dict[str, str | float]]) -> int:
    return sum(str(sample.get('text', '')).count('<unk>') for sample in samples)


def write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_tuning_summary(
    run_rows: list[dict],
    run_artifacts_dir: Path,
    global_artifacts_dir: Path,
) -> None:
    if not run_rows:
        return

    fieldnames = list(run_rows[0].keys())
    run_summary_path = run_artifacts_dir / 'tuning_summary.csv'
    write_csv_rows(run_summary_path, run_rows, fieldnames)

    global_summary_path = global_artifacts_dir / 'tuning_summary.csv'
    if global_summary_path == run_summary_path:
        return

    merged: dict[tuple[str, str], dict] = {}
    if global_summary_path.exists():
        with open(global_summary_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                merged[(row['model_name'], row['config_label'])] = row
    for row in run_rows:
        merged[(row['model_name'], row['config_label'])] = row

    merged_rows = [merged[key] for key in sorted(merged.keys())]
    write_csv_rows(global_summary_path, merged_rows, fieldnames)


def main() -> None:
    args = parse_args()
    run_paths = resolve_run_paths(args, create_dirs=not args.dry_run)
    config_label = 'canonical_defaults' if is_submission_run(args) else build_config_label(args)
    set_seed(DEFAULT_SEED)
    shakespeare_text, social_text = build_corpora(
        social_row_limit=args.social_row_limit,
        processed_data_dir=None if args.dry_run else run_paths['processed_data_dir'],
    )

    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    torch.set_num_threads(4)

    shakespeare_source_path = (
        Path(run_paths['processed_data_dir']) / 'shakespeare_hamlet_clean.txt'
        if not args.dry_run
        else DATA_PROCESSED_DIR / 'shakespeare_hamlet_clean.txt'
    )
    social_source_path = (
        Path(run_paths['processed_data_dir']) / 'social_media_tweets_clean.txt'
        if not args.dry_run
        else DATA_PROCESSED_DIR / 'social_media_tweets_clean.txt'
    )

    specs = resolve_specs(
        selected_models=args.models,
        epochs_override=args.epochs_override,
        batch_size_override=args.batch_size_override,
        seq_len_override=args.seq_len_override,
        hidden_dim_override=args.hidden_dim_override,
        emb_dim_override=args.emb_dim_override,
        dropout_override=args.dropout_override,
        num_layers_override=args.num_layers_override,
        stride_override=args.stride_override,
        lr_override=args.lr_override,
        weight_decay_override=args.weight_decay_override,
        patience_override=args.patience_override,
        min_delta_override=args.min_delta_override,
        word_min_freq_override=args.word_min_freq_override,
        sp_vocab_size_override=args.sp_vocab_size_override,
    )
    tokenizer_assets = build_tokenizer_assets(
        specs,
        shakespeare_text,
        social_text,
        Path(run_paths['models_dir']),
        dry_run=args.dry_run,
    )
    corpus_stats = {
        'shakespeare': {
            'source_file': str(shakespeare_source_path),
            'word_count': count_word_tokens(shakespeare_text),
            'char_count': len(shakespeare_text),
        },
        'social_media': {
            'source_file': str(social_source_path),
            'word_count': count_word_tokens(social_text),
            'char_count': len(social_text),
        },
        'tokenizer_assets': {
            model_name: {
                'tokenizer_type': asset['tokenizer_type'],
                'corpus_key': asset['corpus_key'],
                'model_file': str(asset['model_path']),
                'vocab_file': str(asset['vocab_path']),
                'vocab_size': asset['vocab_size'],
                'normalization_profile': asset['normalization_profile'],
            }
            for model_name, asset in tokenizer_assets.items()
        },
    }
    if args.dry_run:
        print_run_plan(specs, args.social_row_limit, shakespeare_text, social_text, tokenizer_assets)
        return

    histories = {}
    metrics_rows = []
    sample_outputs = {}
    model_inventory_rows = []
    tuning_rows = []

    for model_name, spec in specs.items():
        dataset = build_dataset_for_spec(model_name, spec, shakespeare_text, social_text, tokenizer_assets)
        batch_size = spec.get('batch_size', 64)
        pin_memory = device != 'cpu'
        train_loader, val_loader, test_loader = make_dataloaders(
            dataset,
            batch_size=batch_size,
            splits=(0.8, 0.1, 0.1),
            seed=DEFAULT_SEED,
            num_workers=2,
            pin_memory=pin_memory,
        )

        spec['model_config']['vocab_size'] = len(dataset.vocab)
        coverage_stats = getattr(dataset, 'coverage_stats', {'known_token_rate': None, 'unk_token_rate': None})
        if spec['granularity'] in {'word', 'subword'}:
            model = WordLSTMModel(
                vocab_size=len(dataset.vocab),
                emb_dim=spec['model_config']['emb_dim'],
                hidden_dim=spec['model_config']['hidden_dim'],
                dropout=spec['model_config']['dropout'],
                num_layers=spec['model_config']['num_layers'],
            )
        else:
            model = CharGRUModel(
                vocab_size=len(dataset.vocab),
                emb_dim=spec['model_config']['emb_dim'],
                hidden_dim=spec['model_config']['hidden_dim'],
                dropout=spec['model_config']['dropout'],
                num_layers=spec['model_config']['num_layers'],
            )

        print(f"Training {model_name} | sequences={len(dataset)} | vocab={len(dataset.vocab)} | device={device}")
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=spec['epochs'],
            lr=spec['lr'],
            weight_decay=spec.get('weight_decay', 0.01),
            patience=spec['patience'],
            min_delta=spec.get('min_delta', 1e-3),
            device=device,
            verbose=True,
            run_name=model_name,
            grad_clip=1.0,
            use_amp=(device in {'cuda', 'mps'}),
        )
        histories[model_name] = history

        metadata = {
            'model_name': model_name,
            'corpus_label': spec['corpus_label'],
            'granularity': spec['granularity'],
            'tokenizer_type': spec['tokenizer_type'],
            'normalization_profile': spec['normalization_profile'],
            'comparison_role': spec['comparison_role'],
            'seed': DEFAULT_SEED,
            'vocab': dataset.vocab,
            'model_config': spec['model_config'],
            'default_word_prompt': spec['default_prompt'] if spec['granularity'] in {'word', 'subword'} else None,
            'default_char_prompt': spec['default_prompt'] if spec['granularity'] == 'char' else None,
            'tokenizer_model_relpath': (
                tokenizer_assets[model_name]['model_path'].relative_to(Path(run_paths['models_dir'])).as_posix()
                if model_name in tokenizer_assets
                else None
            ),
            'dataset_summary': {
                'num_sequences': len(dataset),
                'sequence_length': spec['model_config']['seq_len'],
                'vocab_size': len(dataset.vocab),
                'known_token_rate': coverage_stats.get('known_token_rate'),
                'unk_token_rate': coverage_stats.get('unk_token_rate'),
            },
        }
        model_path = Path(run_paths['models_dir']) / f'{model_name}.h5'
        save_model_bundle(model, metadata, model_path)
        bundle = load_model_bundle(model_path)

        best_epoch = min(history, key=lambda row: row['val_loss'])
        test_metrics = evaluate_model(model, test_loader, torch.nn.CrossEntropyLoss(), device=device)
        metrics_row = {
            'model_name': model_name,
            'corpus_label': spec['corpus_label'],
            'granularity': spec['granularity'],
            'tokenizer_type': spec['tokenizer_type'],
            'normalization_profile': spec['normalization_profile'],
            'epochs_ran': len(history),
            'train_loss': best_epoch['train_loss'],
            'val_loss': best_epoch['val_loss'],
            'test_loss': test_metrics['loss'],
            'train_accuracy': best_epoch['train_accuracy'],
            'val_accuracy': best_epoch['val_accuracy'],
            'test_accuracy': test_metrics['accuracy'],
            'train_perplexity': best_epoch['train_perplexity'],
            'val_perplexity': best_epoch['val_perplexity'],
            'test_perplexity': test_metrics['perplexity'],
            'num_sequences': len(dataset),
            'vocab_size': len(dataset.vocab),
            'known_token_rate': coverage_stats.get('known_token_rate'),
            'unk_token_rate': coverage_stats.get('unk_token_rate'),
            'model_file': str(model_path),
            'device': device,
        }
        metrics_rows.append(metrics_row)
        model_inventory_rows.append({
            'model_name': model_name,
            'model_file': str(model_path),
            'corpus_label': spec['corpus_label'],
            'granularity': spec['granularity'],
            'tokenizer_type': spec['tokenizer_type'],
            'sequence_length': spec['model_config']['seq_len'],
            'vocab_size': len(dataset.vocab),
        })

        samples = []
        for idx, temp in enumerate(spec['temperatures'], start=1):
            set_seed(DEFAULT_SEED + idx)
            sample_text = generate_text(
                bundle,
                prompt=spec['default_prompt'],
                length=spec['sample_length'],
                temperature=temp,
                top_k=spec['top_k'],
            )
            samples.append({'temperature': temp, 'text': sample_text})
        sample_outputs[model_name] = samples
        tuning_rows.append({
            'model_name': model_name,
            'config_label': config_label,
            'best_epoch': best_epoch['epoch'],
            'val_loss': best_epoch['val_loss'],
            'val_accuracy': best_epoch['val_accuracy'],
            'known_token_rate': coverage_stats.get('known_token_rate'),
            'unk_token_rate': coverage_stats.get('unk_token_rate'),
            'sample_unk_count': count_sample_unk_tokens(samples),
        })

    metrics_rows = sorted(metrics_rows, key=lambda r: MODEL_NAMES.index(r['model_name']))
    model_inventory_rows = sorted(model_inventory_rows, key=lambda r: MODEL_NAMES.index(r['model_name']))

    if metrics_rows:
        with open(Path(run_paths['artifacts_dir']) / 'metrics_summary.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_rows[0].keys())
            writer.writeheader()
            writer.writerows(metrics_rows)
    if model_inventory_rows:
        with open(Path(run_paths['artifacts_dir']) / 'model_inventory.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=model_inventory_rows[0].keys())
            writer.writeheader()
            writer.writerows(model_inventory_rows)

    artifacts_dir = Path(run_paths['artifacts_dir'])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / 'histories.json').write_text(json.dumps(histories, indent=2), encoding='utf-8')
    (artifacts_dir / 'sample_outputs.json').write_text(json.dumps(sample_outputs, indent=2), encoding='utf-8')
    (artifacts_dir / 'corpus_stats.json').write_text(json.dumps(corpus_stats, indent=2), encoding='utf-8')
    update_tuning_summary(tuning_rows, artifacts_dir, ARTIFACTS_DIR)

    subword_note_lines = [
        (
            f"  - `{model_name}` uses a per-corpus SentencePiece unigram tokenizer "
            f"with `vocab_size={spec['dataset_config']['sp_vocab_size']}`."
        )
        for model_name, spec in specs.items()
        if spec['tokenizer_type'] == 'sentencepiece'
    ]
    subword_note_block = "\n".join(subword_note_lines) if subword_note_lines else "  - No subword models were trained in this run."

    dataset_sources_md = f"""# Dataset sources

## Shakespeare corpus
- **Corpus used:** *Hamlet* by William Shakespeare (Project Gutenberg plain-text edition)
- **Downloaded raw source:** https://www.gutenberg.org/ebooks/1524.txt.utf-8
- **Project page:** https://www.gutenberg.org/ebooks/1524
- **Prepared training file in this package:** `{shakespeare_source_path}`

## Social media corpus
- **Corpus used:** first {args.social_row_limit:,} rows from a public tweet subset CSV
- **Raw source file:** https://raw.githubusercontent.com/HarshalSanap/Twitter-Text-Mining/master/twittersubset.csv
- **Repository page:** https://github.com/HarshalSanap/Twitter-Text-Mining
- **Prepared training file in this package:** `{social_source_path}`

## Tokenization / modeling notes
- Primary comparison models: `shakespeare_word`, `shakespeare_char`, `social_word`, `social_char`
- Extension models: `shakespeare_subword`, `social_subword`
- Strict word models use selective normalization at tokenization time with `min_freq=2`:
  - Shakespeare word model lowercases all-caps alphabetic tokens longer than 3 characters, except Roman numerals.
  - Social word model normalizes `@handle` to `@USER`, digit-bearing tokens to `<NUM>`, lowercases hashtag bodies, and lowercases shouty all-caps tokens longer than 3 characters.
- Subword extension models use per-corpus SentencePiece unigram tokenizers:
{subword_note_block}
"""
    Path(run_paths['dataset_sources_path']).write_text(dataset_sources_md, encoding='utf-8')

    if run_paths['submission_run']:
        readme_md = """# Final Project – Text Generation Using RNNs

This folder contains the technical deliverables for the Agile AI final project.

## Contents
- `final_project_rnn_text_generation.ipynb` – executed notebook for loading models and generating text
- `project_helpers.py` – reusable helper module used by the notebook and training script
- `models/` – six trained model bundles in `.h5` format, plus SentencePiece tokenizer assets for the two subword extension models
- `data/raw/` – raw downloaded source files
- `data/processed/` – cleaned corpora actually used for training
- `artifacts/metrics_summary.csv` – final metrics summary
- `artifacts/histories.json` – per-epoch training history
- `artifacts/sample_outputs.json` – generated comparison samples
- `dataset_sources.md` – dataset links and source notes

## Model files
The `.h5` files are HDF5-packaged PyTorch model bundles. Use the helper function `load_model_bundle()` from `project_helpers.py`, or the convenience functions already included in the notebook, to load any model and generate new text.

Primary comparison models:
- `shakespeare_word`
- `shakespeare_char`
- `social_word`
- `social_char`

Extension models:
- `shakespeare_subword`
- `social_subword`

## SentencePiece dependency
The two subword extension models require `sentencepiece` for training and inference:

```bash
pip install sentencepiece
```

## Quick start
1. Open `final_project_rnn_text_generation.ipynb`.
2. Run the notebook cells from top to bottom if needed.
3. Use `quick_generate(model_name, prompt, length)` to test any of the six models.
"""
        Path(run_paths['readme_path']).write_text(readme_md, encoding='utf-8')

    print('\nTraining complete. Saved artifacts to:')
    print(run_paths['root_dir'])


if __name__ == '__main__':
    main()
