from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from project_helpers import (
    DEFAULT_SEED,
    NextCharDataset,
    NextTokenWordDataset,
    WordLSTMModel,
    CharGRUModel,
    clean_shakespeare_text,
    clean_social_media_text,
    count_word_tokens,
    generate_text,
    load_model_bundle,
    make_dataloaders,
    save_model_bundle,
    set_seed,
    summarize_corpus,
    train_model,
)

BASE_DIR = Path('/Users/nicolasmaitland/Library/CloudStorage/OneDrive-DurhamCollege/Agile/Final Project/final_project_submission/technical')
MODELS_DIR = BASE_DIR / 'models'
DATA_RAW_DIR = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
ARTIFACTS_DIR = BASE_DIR / 'artifacts'

DOWNLOADS_DIR = Path('/mnt/data/downloads')
HAMLET_SOURCE = DOWNLOADS_DIR / 'hamlet_raw.txt'
TWITTER_SOURCE = DOWNLOADS_DIR / 'twittersubset.csv'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def stage_raw_files() -> None:
    shutil.copy2(HAMLET_SOURCE, DATA_RAW_DIR / 'hamlet_gutenberg_raw.txt')
    shutil.copy2(TWITTER_SOURCE, DATA_RAW_DIR / 'twittersubset_public_tweets.csv')


def build_corpora():
    shakespeare_text = clean_shakespeare_text(HAMLET_SOURCE)
    social_text = clean_social_media_text(TWITTER_SOURCE, num_rows=3000)
    (DATA_PROCESSED_DIR / 'shakespeare_hamlet_clean.txt').write_text(shakespeare_text, encoding='utf-8')
    (DATA_PROCESSED_DIR / 'social_media_tweets_clean.txt').write_text(social_text, encoding='utf-8')
    return shakespeare_text, social_text


def model_specs(shakespeare_text: str, social_text: str):
    return {
        'shakespeare_word': {
            'corpus_label': 'Shakespeare (Hamlet)',
            'granularity': 'word',
            'dataset': NextTokenWordDataset(shakespeare_text, seq_len=15, stride=4),
            'epochs': 5,
            'lr': 0.003,
            'patience': 2,
            'model_config': {
                'model_type': 'word_lstm',
                'seq_len': 15,
                'vocab_size': None,
                'emb_dim': 64,
                'hidden_dim': 128,
                'dropout': 0.2,
            },
            'default_prompt': 'love',
            'sample_length': 35,
            'temperatures': [0.8, 1.0, 1.2],
            'top_k': 10,
        },
        'shakespeare_char': {
            'corpus_label': 'Shakespeare (Hamlet)',
            'granularity': 'char',
            'dataset': NextCharDataset(shakespeare_text, seq_len=60, stride=20),
            'epochs': 4,
            'lr': 0.003,
            'patience': 1,
            'model_config': {
                'model_type': 'char_gru',
                'seq_len': 60,
                'vocab_size': None,
                'hidden_dim': 64,
                'dropout': 0.1,
            },
            'default_prompt': 'love ',
            'sample_length': 160,
            'temperatures': [0.7, 0.9, 1.1],
            'top_k': 8,
        },
        'social_word': {
            'corpus_label': 'Social media tweets',
            'granularity': 'word',
            'dataset': NextTokenWordDataset(social_text, seq_len=15, stride=4),
            'epochs': 5,
            'lr': 0.003,
            'patience': 2,
            'model_config': {
                'model_type': 'word_lstm',
                'seq_len': 15,
                'vocab_size': None,
                'emb_dim': 64,
                'hidden_dim': 128,
                'dropout': 0.2,
            },
            'default_prompt': 'love',
            'sample_length': 35,
            'temperatures': [0.8, 1.0, 1.2],
            'top_k': 10,
        },
        'social_char': {
            'corpus_label': 'Social media tweets',
            'granularity': 'char',
            'dataset': NextCharDataset(social_text, seq_len=60, stride=20),
            'epochs': 4,
            'lr': 0.003,
            'patience': 1,
            'model_config': {
                'model_type': 'char_gru',
                'seq_len': 60,
                'vocab_size': None,
                'hidden_dim': 64,
                'dropout': 0.1,
            },
            'default_prompt': 'love ',
            'sample_length': 160,
            'temperatures': [0.7, 0.9, 1.1],
            'top_k': 8,
        },
    }


def main() -> None:
    set_seed(DEFAULT_SEED)
    stage_raw_files()
    shakespeare_text, social_text = build_corpora()

    corpus_stats = {
        'shakespeare': {
            'source_file': str(DATA_PROCESSED_DIR / 'shakespeare_hamlet_clean.txt'),
            'word_count': count_word_tokens(shakespeare_text),
            'char_count': len(shakespeare_text),
        },
        'social_media': {
            'source_file': str(DATA_PROCESSED_DIR / 'social_media_tweets_clean.txt'),
            'word_count': count_word_tokens(social_text),
            'char_count': len(social_text),
        },
    }

    specs = model_specs(shakespeare_text, social_text)
    histories = {}
    metrics_rows = []
    sample_outputs = {}
    model_inventory_rows = []

    for model_name, spec in specs.items():
        dataset = spec['dataset']
        train_loader, val_loader = make_dataloaders(dataset, batch_size=64, val_split=0.1, seed=DEFAULT_SEED)

        spec['model_config']['vocab_size'] = len(dataset.vocab)
        if spec['granularity'] == 'word':
            model = WordLSTMModel(
                vocab_size=len(dataset.vocab),
                emb_dim=spec['model_config']['emb_dim'],
                hidden_dim=spec['model_config']['hidden_dim'],
                dropout=spec['model_config']['dropout'],
            )
        else:
            model = CharGRUModel(
                vocab_size=len(dataset.vocab),
                hidden_dim=spec['model_config']['hidden_dim'],
                dropout=spec['model_config']['dropout'],
            )

        print(f"Training {model_name} | sequences={len(dataset)} | vocab={len(dataset.vocab)}")
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=spec['epochs'],
            lr=spec['lr'],
            patience=spec['patience'],
            device='cpu',
            verbose=True,
            run_name=model_name,
        )
        histories[model_name] = history

        metadata = {
            'model_name': model_name,
            'corpus_label': spec['corpus_label'],
            'granularity': spec['granularity'],
            'seed': DEFAULT_SEED,
            'vocab': dataset.vocab,
            'model_config': spec['model_config'],
            'default_word_prompt': 'love' if spec['granularity'] == 'word' else None,
            'default_char_prompt': 'love ' if spec['granularity'] == 'char' else None,
            'dataset_summary': {
                'num_sequences': len(dataset),
                'sequence_length': spec['model_config']['seq_len'],
                'vocab_size': len(dataset.vocab),
            },
        }
        model_path = MODELS_DIR / f'{model_name}.h5'
        save_model_bundle(model, metadata, model_path)
        bundle = load_model_bundle(model_path)

        final_epoch = history[-1]
        metrics_row = {
            'model_name': model_name,
            'corpus_label': spec['corpus_label'],
            'granularity': spec['granularity'],
            'epochs_ran': len(history),
            'train_loss': final_epoch['train_loss'],
            'val_loss': final_epoch['val_loss'],
            'train_accuracy': final_epoch['train_accuracy'],
            'val_accuracy': final_epoch['val_accuracy'],
            'train_perplexity': final_epoch['train_perplexity'],
            'val_perplexity': final_epoch['val_perplexity'],
            'num_sequences': len(dataset),
            'vocab_size': len(dataset.vocab),
            'model_file': str(model_path),
        }
        metrics_rows.append(metrics_row)
        model_inventory_rows.append({
            'model_name': model_name,
            'model_file': str(model_path),
            'corpus_label': spec['corpus_label'],
            'granularity': spec['granularity'],
            'sequence_length': spec['model_config']['seq_len'],
            'vocab_size': len(dataset.vocab),
        })

        samples = []
        for temp in spec['temperatures']:
            sample_text = generate_text(
                bundle,
                prompt=spec['default_prompt'],
                length=spec['sample_length'],
                temperature=temp,
                top_k=spec['top_k'],
            )
            samples.append({'temperature': temp, 'text': sample_text})
        sample_outputs[model_name] = samples

    metrics_df = pd.DataFrame(metrics_rows).sort_values(['corpus_label', 'granularity'])
    inventory_df = pd.DataFrame(model_inventory_rows).sort_values(['corpus_label', 'granularity'])

    metrics_df.to_csv(ARTIFACTS_DIR / 'metrics_summary.csv', index=False)
    inventory_df.to_csv(ARTIFACTS_DIR / 'model_inventory.csv', index=False)
    (ARTIFACTS_DIR / 'histories.json').write_text(json.dumps(histories, indent=2), encoding='utf-8')
    (ARTIFACTS_DIR / 'sample_outputs.json').write_text(json.dumps(sample_outputs, indent=2), encoding='utf-8')
    (ARTIFACTS_DIR / 'corpus_stats.json').write_text(json.dumps(corpus_stats, indent=2), encoding='utf-8')

    dataset_sources_md = """# Dataset sources

## Shakespeare corpus
- **Corpus used:** *Hamlet* by William Shakespeare (Project Gutenberg plain-text edition)
- **Downloaded raw source:** https://www.gutenberg.org/ebooks/1524.txt.utf-8
- **Project page:** https://www.gutenberg.org/ebooks/1524
- **Prepared training file in this package:** `technical/data/processed/shakespeare_hamlet_clean.txt`

## Social media corpus
- **Corpus used:** first 3,000 rows from a public tweet subset CSV
- **Raw source file:** https://raw.githubusercontent.com/HarshalSanap/Twitter-Text-Mining/master/twittersubset.csv
- **Repository page:** https://github.com/HarshalSanap/Twitter-Text-Mining
- **Prepared training file in this package:** `technical/data/processed/social_media_tweets_clean.txt`

## Notes
- Both corpora were normalized to lowercase and cleaned for CPU-friendly training.
- The social-media corpus retains punctuation, mentions, and short informal expressions to preserve stylistic signals.
"""
    (BASE_DIR / 'dataset_sources.md').write_text(dataset_sources_md, encoding='utf-8')

    readme_md = """# Final Project – Text Generation Using RNNs

This folder contains the technical deliverables for the Agile AI final project.

## Contents
- `final_project_rnn_text_generation.ipynb` – executed notebook for loading models and generating text
- `project_helpers.py` – reusable helper module used by the notebook and training script
- `models/` – four trained model bundles in `.h5` format
- `data/raw/` – raw downloaded source files
- `data/processed/` – cleaned corpora actually used for training
- `artifacts/metrics_summary.csv` – final metrics summary
- `artifacts/histories.json` – per-epoch training history
- `artifacts/sample_outputs.json` – generated comparison samples
- `dataset_sources.md` – dataset links and source notes

## Model files
The `.h5` files are HDF5-packaged PyTorch model bundles. Use the helper function `load_model_bundle()` from `project_helpers.py`, or the convenience functions already included in the notebook, to load any model and generate new text.

## Quick start
1. Open `final_project_rnn_text_generation.ipynb`.
2. Run the notebook cells from top to bottom if needed.
3. Use `quick_generate(model_name, prompt, length)` to test any of the four models.
"""
    (BASE_DIR / 'README.md').write_text(readme_md, encoding='utf-8')

    print('\nTraining complete. Saved artifacts to:')
    print(BASE_DIR)


if __name__ == '__main__':
    main()
