"""
Evaluation script for GroupKFold cross-validation results.

This script:
- Loads models trained on all folds
- Evaluates each fold on its respective test set
- Computes aggregate metrics (mean ± std) across folds
- Saves comprehensive cross-validation results

Usage:
    python src/evaluate_cv.py --model shallow_cnn
    python src/evaluate_cv.py --model deep_cnn --n_folds 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models.shallow_cnn import ShallowCNN
from models.deep_cnn import DeepCNN


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mae = np.abs(y_true - y_pred).mean()
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Exact match percentage
    exact_match = (y_true == y_pred).mean()

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'exact_match': float(exact_match)
    }


def evaluate_fold(model_path, fold_idx, data_dir, batch_size=64):
    """
    Evaluate a model trained on a specific fold.

    Args:
        model_path: Path to trained model
        fold_idx: Fold number
        data_dir: Data directory
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with test metrics and predictions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    # Recreate model
    if 'shallow' in str(model_path).lower():
        model = ShallowCNN(
            in_channels=checkpoint['input_shape'][0],
            out_features=1,
            n_filters=config.get('n_filters', 64),
            dropout_rate=config.get('dropout_rate', 0.5)
        )
    else:
        model = DeepCNN(
            in_channels=checkpoint['input_shape'][0],
            out_features=1,
            n_filters=config.get('n_filters', 64),
            dropout_rate=config.get('dropout_rate', 0.5)
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test data for this fold
    test_data = np.load(Path(data_dir) / f'cnn_fold{fold_idx}_test.npz')
    X_test = test_data['X']
    y_test = test_data['y_count'].astype(np.float32)
    test_participants = test_data['participants']

    # Convert to tensors
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 2, 1)
    y_test_tensor = torch.FloatTensor(y_test)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get predictions
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # Round predictions to nearest integer for step counting
    y_pred_rounded = np.round(y_pred).astype(int)
    y_pred_rounded = np.maximum(y_pred_rounded, 0)  # No negative steps

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred_rounded)

    return {
        'metrics': metrics,
        'y_true': y_true,
        'y_pred': y_pred_rounded,
        'participants': test_participants,
        'n_test_samples': len(y_true)
    }


def main(args):
    """Main cross-validation evaluation pipeline."""

    print('=' * 70)
    print('GROUPKFOLD CROSS-VALIDATION EVALUATION')
    print('=' * 70)
    print(f'\nModel: {args.model}')
    print(f'Number of folds: {args.n_folds}')
    print(f'Models directory: {args.models_dir}\n')

    models_dir = Path(args.models_dir)
    data_dir = Path(args.data_dir)

    # Find model paths for each fold
    fold_results = []

    for fold_idx in range(args.n_folds):
        print(f'\n--- Evaluating Fold {fold_idx} ---')

        # Find most recent model for this fold
        pattern = f'{args.model}_fold{fold_idx}_*'
        fold_dirs = sorted(models_dir.glob(pattern))

        if not fold_dirs:
            print(f'  WARNING: No model found for fold {fold_idx}')
            continue

        # Use most recent
        model_dir = fold_dirs[-1]
        model_path = model_dir / 'final_model.pth'

        if not model_path.exists():
            print(f'  WARNING: Model not found at {model_path}')
            continue

        print(f'  Model: {model_dir.name}')

        # Evaluate
        result = evaluate_fold(model_path, fold_idx, data_dir, args.batch_size)
        result['fold'] = fold_idx
        result['model_path'] = str(model_path)

        fold_results.append(result)

        # Print fold metrics
        metrics = result['metrics']
        print(f'  Test samples: {result["n_test_samples"]}')
        print(f'  MAE: {metrics["mae"]:.4f}')
        print(f'  RMSE: {metrics["rmse"]:.4f}')
        print(f'  R²: {metrics["r2"]:.4f}')
        print(f'  Exact Match: {metrics["exact_match"]:.4f} ({metrics["exact_match"]*100:.1f}%)')

    if not fold_results:
        print('\nERROR: No models found for evaluation!')
        return

    # Aggregate metrics across folds
    print('\n' + '=' * 70)
    print('AGGREGATED CROSS-VALIDATION RESULTS')
    print('=' * 70)

    metrics_df = pd.DataFrame([r['metrics'] for r in fold_results])

    print(f'\nResults across {len(fold_results)} folds:\n')
    for metric in ['mae', 'rmse', 'r2', 'exact_match']:
        mean = metrics_df[metric].mean()
        std = metrics_df[metric].std()
        print(f'{metric.upper():12s}: {mean:.4f} ± {std:.4f}')

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregated metrics
    summary = {
        'model': args.model,
        'n_folds': len(fold_results),
        'metrics_mean': {k: float(v) for k, v in metrics_df.mean().to_dict().items()},
        'metrics_std': {k: float(v) for k, v in metrics_df.std().to_dict().items()},
        'fold_results': [
            {
                'fold': r['fold'],
                'metrics': r['metrics'],
                'n_samples': r['n_test_samples'],
                'model_path': r['model_path']
            }
            for r in fold_results
        ]
    }

    summary_path = output_dir / f'{args.model}_cv_results.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nResults saved to: {summary_path}')

    # Save detailed per-fold metrics
    metrics_csv_path = output_dir / f'{args.model}_fold_metrics.csv'
    metrics_df_full = pd.DataFrame([
        {
            'fold': r['fold'],
            **r['metrics'],
            'n_samples': r['n_test_samples']
        }
        for r in fold_results
    ])
    metrics_df_full.to_csv(metrics_csv_path, index=False)
    print(f'Per-fold metrics saved to: {metrics_csv_path}')

    print('\n' + '=' * 70)
    print('EVALUATION COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GroupKFold cross-validation results')

    parser.add_argument('--model', type=str, required=True,
                        choices=['shallow_cnn', 'deep_cnn'],
                        help='Model architecture to evaluate')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds used in cross-validation')
    parser.add_argument('--models_dir', type=str, default='models/saved',
                        help='Directory containing trained models')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing fold data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')

    args = parser.parse_args()
    main(args)
