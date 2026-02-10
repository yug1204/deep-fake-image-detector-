<!-- Copilot / AI agent instructions for the Deepfake demo repo -->
# Copilot Instructions — Deepfake Detector (DFD-FORENSICS)

This repository is a small research/demo project combining a Streamlit UI with a compact custom CNN for deepfake detection. Use these notes to be productive immediately.

- **High-level architecture**: The UI (`app.py`) is the primary entry point (Streamlit). It loads a model produced by `model.py` (function `build_dcnn_model()`) and optionally weights saved by `train.py`. Sample generators live in `gen_samples.py` and `gen_more_samples.py` which write to `samples/`.

- **Key files**:
  - `app.py`: Streamlit app and demo logic. Detects TensorFlow availability and falls back to a deterministic demo engine when TF is missing. Important constants: `IMG_SIZE = 160`, `WEIGHTS_FILE = 'dummy_weights.h5'`. Demo override logic uses filename tokens and `demo_mode_force` in the sidebar.
  - `model.py`: `build_dcnn_model(input_shape=(160,160,3))` — the D-CNN architecture (Conv blocks, BatchNorm, pooling, dropout, final sigmoid). Optimizer: Adam lr=0.01. Model compiled with `binary_crossentropy`.
  - `train.py`: `train_model(data_dir, weights_output_path)` — expects dataset layout `data_dir/train/REAL`, `data_dir/train/DEEPFAKE`, and similar `val/` folders. Uses Keras ImageDataGenerator, checkpoint saves weights via `ModelCheckpoint(save_weights_only=True)`.
  - `gen_more_samples.py` & `gen_samples.py`: generate example images in `samples/` with naming conventions like `real_*` and `deepfake_*` used by the demo.
  - `requirements.txt`: primary dependencies: `tensorflow`, `streamlit`, `numpy`, `pillow`.

- **Common workflows / commands**:
  - Run the demo UI (local):

    streamlit run app.py

  - Train a model (example):

    python train.py --data_dir path/to/dataset --output dummy_weights.h5

  - Generate demo samples:

    python gen_more_samples.py

  - Quick test of model build (no training): run `python model.py` — it prints `model.summary()`.

- **Data & naming conventions**:
  - Training dataset: folder structure required by `flow_from_directory`: `train/REAL`, `train/DEEPFAKE`, `val/REAL`, `val/DEEPFAKE`.
  - Sample filenames are used by `app.py` demo heuristics: files containing substrings like `fake`, `deep`, `gen`, `ai` are treated as deepfake examples; `real`, `auth`, `person` are treated as real. Keep these tokens if you add demo images.
  - Default image sizes: the model expects 160x160 inputs; sample generators save 160×160 images (see `gen_more_samples.py`).

- **Project-specific patterns / gotchas**:
  - `app.py` supports a NO-TF path: if TensorFlow isn't installed, the app runs in UI-only demo mode using deterministic heuristics. Agents should handle both modes when editing or testing UI behavior.
  - The app caches the model via `@st.cache_resource` and loads weights with `model.load_weights(WEIGHTS_FILE)` — the code expects weight files (saved with `save_weights_only=True` in training).
  - High default learning rate (0.01) and heavy dropout layers are intentional for demonstration; changes to training hyperparameters should be validated on small runs.
  - `train.py` prints helpful errors if expected dataset folders are missing — use those messages to triage dataset issues.

- **Testing & debugging tips**:
  - To reproduce UI behavior without TensorFlow, temporarily uninstall TF or set TF import to fail to exercise the demo deterministic branch.
  - To reproduce inference deterministically, use the sample files in `samples/` (names carry demo tokens) or seed pseudo-random generation using the image-hash logic in `app.py`.
  - When changing model shapes, update `IMG_SIZE` in `app.py` and `train.py` consistently.

- **Integration points / external dependencies**:
  - TensorFlow (GPU/CPU) — optional for the UI but required for real training and inference. Streamlit is required to run the UI.
  - The app expects a weights file (path in `WEIGHTS_FILE`) if you want model-based inference; `train.py` will produce such files when run with `--output`.

- **When editing code** — concrete examples to follow:
  - Preserve the filename token behavior in `app.py` if you want the demo to keep showing “correct” results for example images.
  - If changing the model, update the optimizer or compile settings inside `model.py` and mirror any new input shape in `app.py`/`train.py`.
  - Use `samples/` for small functional tests rather than requiring a full dataset during development.

If anything here is unclear or you want more detail on a specific part (data preprocessing, a training run example, or how to adapt the UI), tell me which section to expand and I will iterate.  
