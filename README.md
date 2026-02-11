# Deep Fake Image Detector

This repository contains a Deep Convolutional Neural Network (D-CNN) implementation for detecting deepfake images. It includes scripts for training the model and generating sample data.

## Project Structure

- `model.py`: Defines the D-CNN model architecture.
- `train.py`: Script to train the model on a dataset.
- `app.py`: Streamlit application for demonstrating the detector (if applicable).
- `gen_samples.py`, `gen_more_samples.py`: Helper scripts to generate sample data or visualizations.
- `requirements.txt`: List of Python dependencies.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yug1204/deep-fake-image-detector-.git
    cd deep-fake-image-detector-
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the model, you need a dataset organized into `train` and `val` directories, each containing `REAL` and `DEEPFAKE` subdirectories.

```bash
python train.py --data_dir /path/to/dataset --output my_model_weights.h5
```

### Running the App

(If `app.py` is a Streamlit app)

```bash
streamlit run app.py
```

## Model Architecture

The model is a Deep CNN designed to classify images as either "Real" or "Fake". See `model.py` for details.
