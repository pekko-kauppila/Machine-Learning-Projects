"""
Evaluation script for trained Step Counter CNN models.

This script handles:
- Loading a trained model
- Evaluating on the test set
- Calculating metrics (MAE, RMSE, R²)
- Saving test results

Usage:
    python src/evaluate.py --model_path models/saved/deep_cnn_20240315_143022/final_model.pth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models.shallow_cnn import ShallowCNN
from models.deep_cnn import DeepCNN


def evaluate_model(model_path, data_dir='data/processed', batch_size=64):
    """
    Evaluate a trained model on the test set.

    Args:
        model_path: Path to saved model checkpoint
        data_dir: Directory containing processed data
        batch_size: Batch size for evaluation

    Returns:
        test_results: Dictionary of test metrics
    """

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model checkpoint
    print(f'\nLoading model from {model_path}...')
    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint['config']
    input_shape = checkpoint['input_shape']
    task_type = checkpoint.get('task_type', 'regression')

    print(f'Model Configuration:')
    print(f'  Task: {task_type}')
    print(f'  Input shape: {input_shape}')
    print(f'  Model type: {config.get("model_type", "unknown")}')

    # Determine model architecture
    model_type = str(config.get('experiment_name', '')).split('_')[0]

    if model_type == 'shallow' or 'shallow' in str(config.get('model_type', '')):
        model = ShallowCNN(
            input_channels=input_shape[1],
            sequence_length=input_shape[0],
            num_filters=config.get('n_filters', 64),
            dropout_rate=config.get('dropout_rate', 0.5)
        )
    elif model_type == 'deep' or 'deep' in str(config.get('model_type', '')):
        model = DeepCNN(
            input_channels=input_shape[1],
            sequence_length=input_shape[0],
            num_filters=config.get('n_filters', 64),
            dropout_rate=config.get('dropout_rate', 0.5)
        )
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel loaded: {model_type.upper()}')
    print(f'Parameters: {total_params:,}')

    # Load test data
    data_dir = Path(data_dir)
    print(f'\nLoading test data from {data_dir}...')

    test_data = np.load(data_dir / 'cnn_test_data.npz')
    X_test = test_data['X']
    y_test = test_data['y_count'].astype(np.float32)

    # Convert to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 2, 1)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create DataLoader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Test data loaded:')
    print(f'  X_test: {X_test.shape} -> Tensor: {X_test_tensor.shape}')
    print(f'  y_test: {y_test.shape}')
    print(f'  Batches: {len(test_loader)}')

    # Make predictions
    print(f'\nEvaluating on test set...')
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).squeeze()

            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(batch_y.numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)

    # Calculate metrics
    print(f'\n{"="*60}')
    print('TEST SET EVALUATION')
    print(f'{"="*60}')

    test_mae = np.abs(test_preds - test_targets).mean()
    test_rmse = np.sqrt(((test_preds - test_targets) ** 2).mean())
    test_r2 = 1 - (np.sum((test_targets - test_preds) ** 2) / np.sum((test_targets - test_targets.mean()) ** 2))

    criterion = nn.MSELoss()
    test_loss = criterion(torch.FloatTensor(test_preds), torch.FloatTensor(test_targets)).item()

    print(f'\nMetrics:')
    print(f'  Loss (MSE): {test_loss:.4f}')
    print(f'  MAE: {test_mae:.4f}')
    print(f'  RMSE: {test_rmse:.4f}')
    print(f'  R² Score: {test_r2:.4f}')

    # Additional info
    test_pred_rounded = np.maximum(np.round(test_preds).astype(int), 0)
    exact_match = (test_targets == test_pred_rounded).mean()
    total_error = test_pred_rounded.sum() - test_targets.sum()

    print(f'\nStep Count Accuracy:')
    print(f'  Exact Match: {exact_match:.4f} ({exact_match*100:.1f}%)')
    print(f'  Total True Steps: {int(test_targets.sum())}')
    print(f'  Total Predicted Steps: {int(test_pred_rounded.sum())}')
    print(f'  Total Error: {int(total_error)} steps')

    print(f'\n{"="*60}')

    # Prepare results
    test_results = {
        'task_type': 'regression',
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2),
        'exact_match': float(exact_match),
        'total_true_steps': int(test_targets.sum()),
        'total_pred_steps': int(test_pred_rounded.sum()),
        'total_error': int(total_error)
    }

    # Save test results
    model_dir = Path(model_path).parent
    test_results_path = model_dir / 'test_results.json'

    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f'\nTest results saved to: {test_results_path}')

    return test_results


def main(args):
    """Main evaluation pipeline."""

    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')

    print(f'Evaluating model: {model_path.parent.name}')

    test_results = evaluate_model(
        model_path=model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    print('\nEvaluation complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained Step Counter CNN model')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model checkpoint (.pth file)')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed test data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')

    args = parser.parse_args()
    main(args)
