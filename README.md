# Final Project – Text Generation Using RNNs

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
