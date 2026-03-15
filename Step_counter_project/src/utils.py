"""
Utility functions for Step Counter project.

This module contains shared helper functions for data processing,
visualization, and model utilities.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics.

    Args:
        history: Training history dictionary with keys:
                 'loss', 'val_loss', 'mae', 'val_mae', 'rmse', 'val_rmse'
        save_path: Optional path to save the plot
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # RMSE
    axes[2].plot(history['rmse'], label='Train RMSE', linewidth=2)
    axes[2].plot(history['val_rmse'], label='Val RMSE', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('RMSE')
    axes[2].set_title('Root Mean Squared Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Plot saved to: {save_path}')

    plt.show()


def plot_predictions(y_true, y_pred, save_path=None):
    """
    Visualize regression predictions vs actual values.

    Args:
        y_true: True step counts
        y_pred: Predicted step counts
        save_path: Optional path to save the plot
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    y_pred_rounded = np.maximum(np.round(y_pred).astype(int), 0)

    # Predicted vs Actual
    axes[0].scatter(y_true, y_pred_rounded, alpha=0.3, s=10)
    max_val = max(y_true.max(), y_pred_rounded.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')
    axes[0].set_xlabel('True Step Count')
    axes[0].set_ylabel('Predicted Step Count')
    axes[0].set_title('Predicted vs Actual')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error Distribution
    errors = y_pred_rounded - y_true
    axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].set_xlabel('Prediction Error (Predicted - True)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Error Distribution (MAE={np.abs(errors).mean():.2f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Error by True Count
    unique_counts = np.unique(y_true)
    mae_by_count = [np.abs(errors[y_true == c]).mean() for c in unique_counts]

    axes[2].bar(unique_counts, mae_by_count, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('True Step Count')
    axes[2].set_ylabel('Mean Absolute Error')
    axes[2].set_title('Error by Step Count')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Plot saved to: {save_path}')

    plt.show()


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """

    mae = np.abs(y_pred - y_true).mean()
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))

    # Step count specific metrics
    y_pred_rounded = np.maximum(np.round(y_pred).astype(int), 0)
    exact_match = (y_true == y_pred_rounded).mean()
    total_error = y_pred_rounded.sum() - y_true.sum()

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'exact_match': float(exact_match),
        'total_true': int(y_true.sum()),
        'total_pred': int(y_pred_rounded.sum()),
        'total_error': int(total_error)
    }


def print_metrics(metrics, title='Metrics'):
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics display
    """

    print(f'\n{"="*60}')
    print(f'{title}')
    print(f'{"="*60}')

    if 'mae' in metrics:
        print(f'\nRegression Metrics:')
        print(f'  MAE: {metrics["mae"]:.4f}')
        print(f'  RMSE: {metrics["rmse"]:.4f}')
        print(f'  R² Score: {metrics["r2"]:.4f}')

    if 'exact_match' in metrics:
        print(f'\nStep Count Accuracy:')
        print(f'  Exact Match: {metrics["exact_match"]:.4f} ({metrics["exact_match"]*100:.1f}%)')
        print(f'  Total True Steps: {metrics["total_true"]}')
        print(f'  Total Predicted Steps: {metrics["total_pred"]}')
        print(f'  Total Error: {metrics["total_error"]} steps')

    print(f'{"="*60}\n')


def load_normalization_params(normalization_path):
    """
    Load normalization parameters from saved file.

    Args:
        normalization_path: Path to normalization .npz file

    Returns:
        mean, std, channel_names
    """

    norm_data = np.load(normalization_path, allow_pickle=True)
    mean = norm_data['mean']
    std = norm_data['std']
    channel_names = norm_data['channel_names']

    return mean, std, channel_names


def normalize_data(X, mean, std):
    """
    Normalize data using provided mean and std.

    Args:
        X: Data to normalize
        mean: Mean values per channel
        std: Std values per channel

    Returns:
        Normalized data
    """

    return (X - mean) / std


def denormalize_data(X_norm, mean, std):
    """
    Denormalize data using provided mean and std.

    Args:
        X_norm: Normalized data
        mean: Mean values per channel
        std: Std values per channel

    Returns:
        Original scale data
    """

    return X_norm * std + mean
