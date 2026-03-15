# Step Counter - Source Code

This directory contains the core implementation of the Step Counter project.

## Structure

```
src/
├── models/              # CNN model architectures
│   ├── shallow_cnn.py   # Shallow CNN with 1 conv layer
│   └── deep_cnn.py      # Deep CNN with 5 conv layers
├── prepare_data.py      # Data preparation script
├── train.py             # Model training script
├── evaluate.py          # Model evaluation script
└── utils.py             # Shared utility functions
```

## Quick Start

### 1. Prepare Data

Process raw accelerometer data into windowed format for CNN training:

```bash
cd Step_counter_project
python src/prepare_data.py
```

This creates processed data in `data/processed/`:
- `cnn_train_data.npz` - Training data
- `cnn_val_data.npz` - Validation data
- `cnn_test_data.npz` - Test data

### 2. Train Model

Train a CNN model for step count regression:

```bash
# Train ShallowCNN
python src/train.py --model shallow_cnn --epochs 50 --lr 0.001

# Train DeepCNN with custom hyperparameters
python src/train.py --model deep_cnn --batch_size 32 --n_filters 128 --dropout 0.3
```

**Training options:**
- `--model`: Model architecture (`shallow_cnn` or `deep_cnn`)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 32)
- `--n_filters`: Number of convolutional filters (default: 64)
- `--dropout`: Dropout rate (default: 0.5)
- `--patience`: Early stopping patience (default: 10)
- `--notes`: Experiment notes

Models are saved to `models/saved/<experiment_name>/`

### 3. Evaluate Model

Evaluate a trained model on the test set:

```bash
python src/evaluate.py --model_path models/saved/deep_cnn_20240315_143022/final_model.pth
```

Test results are saved to the model directory as `test_results.json`

## Example Workflow

```bash
# 1. Prepare data (run once)
python src/prepare_data.py

# 2. Train multiple models
python src/train.py --model shallow_cnn --notes "baseline"
python src/train.py --model deep_cnn --n_filters 128 --notes "increased filters"

# 3. Evaluate best model
python src/evaluate.py --model_path models/saved/deep_cnn_20240315_143022/final_model.pth
```

## Using in Python Scripts

You can also import and use these modules in your own scripts:

```python
from src.models.deep_cnn import DeepCNN
from src.utils import calculate_regression_metrics, plot_predictions

# Load model
model = DeepCNN(input_channels=4, sequence_length=200)

# Calculate metrics
metrics = calculate_regression_metrics(y_true, y_pred)

# Visualize predictions
plot_predictions(y_true, y_pred, save_path='predictions.png')
```

## Dependencies

See `requirements.txt` in the project root:

```bash
pip install -r requirements.txt
```
