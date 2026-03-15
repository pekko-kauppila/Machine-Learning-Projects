"""
Training script for Step Counter CNN models.

This script handles:
- Loading preprocessed data
- Model initialization (ShallowCNN or DeepCNN)
- Training with early stopping and learning rate scheduling
- Saving trained models and results

Usage:
    python src/train.py --model shallow_cnn --epochs 100 --lr 0.001
    python src/train.py --model deep_cnn --batch_size 32 --n_filters 128
"""

import argparse
import copy
import json
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.shallow_cnn import ShallowCNN
from models.deep_cnn import DeepCNN


def train_model(model, train_loader, val_loader, config, device):
    """
    Train a model with early stopping and learning rate scheduling.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Configuration dictionary
        device: torch device (cpu or cuda)

    Returns:
        history: Training history dictionary
        model: Trained model
    """

    model_name = config.get('model_type', 'Model').replace('_', ' ').title()

    print(f'\n{"="*60}')
    print(f'Training {model_name}')
    print(f'Experiment: {config["experiment_name"]}')
    print(f'{"="*60}')

    # Move model to device
    model = model.to(device)

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    # Loss function for regression
    criterion = nn.MSELoss()
    print(f'\nTask: Regression (step counting)')
    print(f'Loss function: MSELoss')

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # Training history
    history = {
        'loss': [],
        'val_loss': [],
        'mae': [],
        'val_mae': [],
        'rmse': [],
        'val_rmse': []
    }

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    # Training loop
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(batch_y.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        # Calculate regression metrics
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        train_mae = np.abs(train_preds - train_targets).mean()
        val_mae = np.abs(val_preds - val_targets).mean()
        train_rmse = np.sqrt(((train_preds - train_targets) ** 2).mean())
        val_rmse = np.sqrt(((val_preds - val_targets) ** 2).mean())

        history['mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)

        print(f'Epoch {epoch+1}/{config["epochs"]} - '
              f'loss: {train_loss:.4f} - mae: {train_mae:.4f} - '
              f'val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}')

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())

            # Save best checkpoint in experiment directory
            checkpoint_path = config['experiment_dir'] / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                print(f'Best model was at epoch {best_epoch}')
                break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Add training metadata to history
    history['best_epoch'] = best_epoch
    history['total_epochs'] = epoch + 1

    print(f'\nTraining complete!')
    print(f'Best epoch: {best_epoch}')
    print(f'Best validation loss: {best_val_loss:.4f}')

    return history, model


def main(args):
    """Main training pipeline."""

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # Configuration
    CONFIG = {
        'data_dir': Path(args.data_dir),
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'patience': args.patience,
        'n_filters': args.n_filters,
        'dropout_rate': args.dropout,
        'save_dir': Path(args.save_dir),
        'experiment_name': f'{args.model}_fold{args.fold}_{datetime.now().strftime("%Y%m%d_%H%M%S")}' if args.fold is not None else f'{args.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'notes': args.notes,
        'model_type': args.model,
        'fold': args.fold
    }

    # Create experiment directory
    experiment_dir = CONFIG['save_dir'] / CONFIG['experiment_name']
    experiment_dir.mkdir(parents=True, exist_ok=True)
    CONFIG['experiment_dir'] = experiment_dir

    print(f'\nConfiguration:')
    print(json.dumps({k: str(v) for k, v in CONFIG.items()}, indent=2))

    # Load data
    print(f'\nLoading data from {CONFIG["data_dir"]}...')

    if args.fold is not None:
        # Load fold-specific data
        print(f'Using GroupKFold cross-validation, Fold {args.fold}')
        train_data = np.load(CONFIG['data_dir'] / f'cnn_fold{args.fold}_train.npz')
        X_train = train_data['X']
        y_train = train_data['y_count'].astype(np.float32)

        # For fold-based training, use 10% of training data as validation
        # Set fold-specific random seed for reproducibility
        fold_random_state = 42 + args.fold
        np.random.seed(fold_random_state)

        val_split_idx = int(len(X_train) * 0.9)
        indices = np.random.permutation(len(X_train))
        train_indices = indices[:val_split_idx]
        val_indices = indices[val_split_idx:]

        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]

        print(f'  Created internal train/val split from fold training data (90/10, seed={fold_random_state})')

        # Reset random seed for training
        np.random.seed(42)
        torch.manual_seed(42)
    else:
        # Legacy: load separate train/val files
        print('Using legacy train/val split')
        train_data = np.load(CONFIG['data_dir'] / 'cnn_train_data.npz')
        X_train = train_data['X']
        y_train = train_data['y_count'].astype(np.float32)

        val_data = np.load(CONFIG['data_dir'] / 'cnn_val_data.npz')
        X_val = val_data['X']
        y_val = val_data['y_count'].astype(np.float32)

    # Convert to PyTorch tensors (PyTorch expects channels first: N, C, L)
    X_train_tensor = torch.FloatTensor(X_train).permute(0, 2, 1)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val).permute(0, 2, 1)
    y_val_tensor = torch.FloatTensor(y_val)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    print(f'Data loaded:')
    print(f'  X_train: {X_train.shape} -> Tensor: {X_train_tensor.shape}')
    print(f'  X_val: {X_val.shape} -> Tensor: {X_val_tensor.shape}')
    print(f'  Batch size: {CONFIG["batch_size"]}')
    print(f'  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')

    # Create model
    print(f'\nCreating {args.model} model...')
    if args.model == 'shallow_cnn':
        model = ShallowCNN(
            input_channels=X_train.shape[2],
            sequence_length=X_train.shape[1],
            num_filters=CONFIG['n_filters'],
            dropout_rate=CONFIG['dropout_rate']
        )
    elif args.model == 'deep_cnn':
        model = DeepCNN(
            input_channels=X_train.shape[2],
            sequence_length=X_train.shape[1],
            num_filters=CONFIG['n_filters'],
            dropout_rate=CONFIG['dropout_rate']
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}. Choose 'shallow_cnn' or 'deep_cnn'")

    print(f'Model created: {args.model}')
    print(f'Input shape: (window_size={X_train.shape[1]}, n_channels={X_train.shape[2]})')
    print(f'Filters: {CONFIG["n_filters"]}, Dropout: {CONFIG["dropout_rate"]}')

    # Train model
    history, model = train_model(model, train_loader, val_loader, CONFIG, device)

    # Evaluate on validation set
    print(f'\n{"="*60}')
    print('VALIDATION SET RESULTS')
    print(f'{"="*60}')

    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).squeeze()
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(batch_y.numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    val_mae = np.abs(val_preds - val_targets).mean()
    val_rmse = np.sqrt(((val_preds - val_targets) ** 2).mean())
    val_r2 = 1 - (np.sum((val_targets - val_preds) ** 2) / np.sum((val_targets - val_targets.mean()) ** 2))

    criterion = nn.MSELoss()
    val_loss = criterion(torch.FloatTensor(val_preds), torch.FloatTensor(val_targets)).item()

    print(f'\nMetrics:')
    print(f'  Loss (MSE): {val_loss:.4f}')
    print(f'  MAE: {val_mae:.4f}')
    print(f'  RMSE: {val_rmse:.4f}')
    print(f'  R² Score: {val_r2:.4f}')

    # Save model
    model_path = CONFIG['experiment_dir'] / 'final_model.pth'

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'input_shape': (X_train.shape[1], X_train.shape[2]),
        'task_type': 'regression',
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'val_loss': val_loss,
        'training_history': history
    }, model_path)

    print(f'\nModel saved to: {model_path}')

    # Save validation results
    val_results = {
        'task_type': 'regression',
        'val_loss': float(val_loss),
        'val_mae': float(val_mae),
        'val_rmse': float(val_rmse),
        'val_r2': float(val_r2)
    }

    val_results_path = CONFIG['experiment_dir'] / 'val_results.json'
    with open(val_results_path, 'w') as f:
        json.dump(val_results, f, indent=2)
    print(f'Validation results saved to: {val_results_path}')

    # Save configuration
    config_path = CONFIG['experiment_dir'] / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({k: str(v) for k, v in CONFIG.items()}, f, indent=2)
    print(f'Configuration saved to: {config_path}')

    # Track experiment in CSV
    experiments_csv = CONFIG['save_dir'] / 'experiments.csv'

    all_columns = [
        'experiment_name', 'timestamp', 'n_filters', 'dropout_rate',
        'learning_rate', 'batch_size', 'epochs_trained', 'best_epoch', 'notes',
        'val_mae', 'val_rmse', 'val_r2', 'val_loss'
    ]

    experiment_record = {
        'experiment_name': CONFIG['experiment_name'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'n_filters': CONFIG['n_filters'],
        'dropout_rate': CONFIG['dropout_rate'],
        'learning_rate': CONFIG['learning_rate'],
        'batch_size': CONFIG['batch_size'],
        'epochs_trained': history['total_epochs'],
        'best_epoch': history['best_epoch'],
        'notes': CONFIG.get('notes', ''),
        'val_mae': float(val_mae),
        'val_rmse': float(val_rmse),
        'val_r2': float(val_r2),
        'val_loss': float(val_loss)
    }

    file_exists = experiments_csv.exists()
    with open(experiments_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(experiment_record)

    print(f'Experiment tracked in: {experiments_csv}')

    print('\n' + '=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f'\nExperiment directory: {CONFIG["experiment_dir"]}')
    print(f'Model path: {model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Step Counter CNN model')

    # Model selection
    parser.add_argument('--model', type=str, default='shallow_cnn',
                        choices=['shallow_cnn', 'deep_cnn'],
                        help='Model architecture to train')

    # Data paths
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--save_dir', type=str, default='models/saved',
                        help='Directory to save trained models')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold number for cross-validation (0-4). If not specified, uses legacy train/val split')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')

    # Model hyperparameters
    parser.add_argument('--n_filters', type=int, default=64,
                        help='Number of filters in convolutional layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')

    # Experiment tracking
    parser.add_argument('--notes', type=str, default='',
                        help='Notes about this experiment')

    args = parser.parse_args()
    main(args)
