# Step Counter: Deep Learning for Accelerometer-Based Step Counting

Predict step counts from hip-worn accelerometer data using 1D Convolutional Neural Networks.

**Key Features:**
- 🏃 Regression-based step counting (not just detection)
- 🧠 Two CNN architectures: ShallowCNN and DeepCNN
- 📊 Trained on 39 participants with ground-truth annotations
- 🚀 **Fully runnable** via command-line scripts (not just notebooks!)

## Quick Start

### 1. Install Dependencies

```bash
cd Step_counter_project

# Create conda environment
conda create -n step-counter python=3.10
conda activate step-counter

# Install PyTorch (check pytorch.org for your system)
conda install pytorch torchvision torchaudio -c pytorch

# Install other packages
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Matplotlib, Seaborn, scikit-learn

### 2. Download Dataset

Download the [OxWalk Dataset](https://doi.org/10.5287/bodleian:KJe7VgMNV) (hip-worn accelerometer data from 39 healthy adults with video-annotated step counts).

Extract to `data/OxWalk_Dec2022/`:
```
Step_counter_project/
└── data/
    └── OxWalk_Dec2022/
        ├── Hip_100Hz/
        └── metadata.csv
```

### 3. Prepare Data

Process raw accelerometer data into windowed format:

```bash
python src/prepare_data.py
```

This generates processed data in `data/processed/`:
- `cnn_train_data.npz` - Training data (81K windows)
- `cnn_val_data.npz` - Validation data (29K windows)
- `cnn_test_data.npz` - Test data (13K windows)

### 4. Train Model

Train a CNN model for step count regression:

```bash
# Train ShallowCNN (baseline)
python src/train.py --model shallow_cnn --epochs 50 --lr 0.001

# Train DeepCNN (best performance)
python src/train.py --model deep_cnn --batch_size 32 --n_filters 128 --dropout 0.3 --notes "optimized hyperparams"
```

**Training options:**
- `--model`: Architecture (`shallow_cnn` or `deep_cnn`)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 32)
- `--n_filters`: Number of filters (default: 64)
- `--dropout`: Dropout rate (default: 0.5)
- `--patience`: Early stopping patience (default: 10)

Models are saved to `models/saved/<experiment_name>/`

### 5. Evaluate Model

Evaluate on the held-out test set:

```bash
python src/evaluate.py --model_path models/saved/deep_cnn_20240315_143022/final_model.pth
```

**Example output:**
```
TEST SET EVALUATION
============================================================
Metrics:
  Loss (MSE): 0.3274
  MAE: 0.3246
  RMSE: 0.5721
  R² Score: 0.9021

Step Count Accuracy:
  Exact Match: 68.5%
  Total True Steps: 7532
  Total Predicted Steps: 7489
  Total Error: -43 steps
```

## Project Structure

```
Step_counter_project/
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── src/
│   ├── prepare_data.py       # Data preparation script
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Model evaluation script
│   ├── utils.py              # Helper functions
│   └── models/
│       ├── shallow_cnn.py    # Shallow CNN (1 conv layer)
│       └── deep_cnn.py       # Deep CNN (5 conv layers)
├── notebooks/                 # Exploratory notebooks (optional)
│   ├── eda.ipynb             # Data exploration
│   └── hyperparameter_search_colab.ipynb  # Colab training
├── data/                      # Data directory (not in repo)
└── models/saved/              # Trained models (not in repo)
```

## Available Models

### ShallowCNN
- Single convolutional layer baseline
- ~410K parameters
- Fast training (~2 min on CPU)
- Good for quick experiments

### DeepCNN
- 5 convolutional layers with batch normalization
- ~1.5M parameters
- Better performance (~0.32 MAE)
- Recommended for production

Both models perform **regression** to predict step counts (0-7 steps per 2-second window).

## Alternative: Using Notebooks

If you prefer Jupyter notebooks:

1. **Prepare data**: Run `notebooks/data_preparation_cnn.ipynb`
2. **Train model**: Run `notebooks/model_development.ipynb`
3. **Evaluate**: Run `notebooks/evaluation.ipynb`
4. **Explore data**: Run `notebooks/eda.ipynb`

## Google Colab Training (GPU)

For faster training with GPU:

1. Upload processed data to Google Drive
2. Upload `notebooks/hyperparameter_search_colab.ipynb` to Colab
3. Set runtime to GPU (T4)
4. Update paths in notebook to point to your Drive folder

See notebook for detailed instructions.

## Results

| Model | MAE | RMSE | R² | Exact Match |
|-------|-----|------|----|----|
| ShallowCNN | 0.34 | 0.58 | 0.89 | 66% |
| DeepCNN | 0.32 | 0.57 | 0.90 | 69% |

*Metrics on held-out test set (8 participants)*

## Citation

Dataset:
```
OxWalk: Hip Accelerometer Step Count Dataset with Manually Annotated Steps
DOI: 10.5287/bodleian:KJe7VgMNV
```

