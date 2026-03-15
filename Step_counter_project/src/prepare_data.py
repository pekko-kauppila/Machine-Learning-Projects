"""
Data preparation script for Step Counter CNN training.

This script processes raw accelerometer data into windowed format suitable for CNN training.
It handles:
- Windowing continuous signals into fixed-size segments
- Creating step count labels for regression
- Train/validation/test splitting by participant
- Data normalization and saving processed arrays

Usage:
    python src/prepare_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings('ignore')


# Configuration
SAMPLING_RATE = 100  # Hz
WINDOW_SIZE_SEC = 2.0  # seconds
WINDOW_SIZE = int(WINDOW_SIZE_SEC * SAMPLING_RATE)  # samples

# Overlap configuration
TRAIN_OVERLAP = 0.5  # 50% overlap for training
TEST_OVERLAP = 0.0  # 0% overlap for test (non-overlapping windows)

TRAIN_STEP = int(WINDOW_SIZE * (1 - TRAIN_OVERLAP))
TEST_STEP = int(WINDOW_SIZE * (1 - TEST_OVERLAP))

# Cross-validation configuration
N_FOLDS = 5  # Number of folds for GroupKFold

# Channel configuration
USE_SMV_CHANNEL = True  # Add SMV as 4th channel

# Paths
DATA_PATH = Path('data/OxWalk_Dec2022/Hip_100Hz')
METADATA_PATH = Path('data/OxWalk_Dec2022/metadata.csv')
OUTPUT_DIR = Path('data/processed')


def process_participant_cnn(participant_id, data_path, window_size=WINDOW_SIZE,
                            step_size=TRAIN_STEP, use_smv=USE_SMV_CHANNEL):
    """
    Process a single participant's data and extract raw windowed data for CNN.

    Args:
        participant_id: Participant ID (e.g., 'P01')
        data_path: Path to data directory
        window_size: Size of window in samples
        step_size: Step size for sliding window (controls overlap)
        use_smv: Whether to add SMV as 4th channel

    Returns:
        windows: numpy array of shape (n_windows, window_size, n_channels)
        labels_dict: dictionary with different label types
        metadata_list: list of metadata dicts for each window
    """

    # Load data
    file_path = data_path / f'{participant_id}_hip100.csv'
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Calculate derived channels if needed
    if use_smv:
        df['smv'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

    windows_list = []
    labels_count = []
    metadata_list = []

    # Sliding window
    for start_idx in range(0, len(df) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window = df.iloc[start_idx:end_idx]

        # Extract raw accelerometer data
        channels = []
        channels.append(window['x'].values)
        channels.append(window['y'].values)
        channels.append(window['z'].values)

        if use_smv:
            channels.append(window['smv'].values)

        # Stack channels: (window_size, n_channels)
        window_data = np.stack(channels, axis=-1)
        windows_list.append(window_data)

        # Create labels
        annotations = window['annotation'].values
        step_count = annotations.sum()
        labels_count.append(step_count)

        # Store metadata
        metadata_list.append({
            'participant': participant_id,
            'window_start': start_idx,
            'window_end': end_idx,
            'timestamp_start': window['timestamp'].iloc[0],
            'timestamp_end': window['timestamp'].iloc[-1]
        })

    # Convert to numpy arrays
    windows = np.array(windows_list)  # (n_windows, window_size, n_channels)
    labels_dict = {
        'count': np.array(labels_count)  # (n_windows,)
    }

    return windows, labels_dict, metadata_list


def prepare_data(data_path=DATA_PATH, metadata_path=METADATA_PATH, output_dir=OUTPUT_DIR):
    """
    Main data preparation pipeline using GroupKFold cross-validation.

    Args:
        data_path: Path to raw data directory
        metadata_path: Path to metadata CSV
        output_dir: Path to save processed data
    """

    print('=' * 70)
    print('STEP COUNTER - DATA PREPARATION WITH GROUPKFOLD CROSS-VALIDATION')
    print('=' * 70)
    print(f'\nWindow size: {WINDOW_SIZE} samples ({WINDOW_SIZE_SEC} seconds)')
    print(f'Train step size: {TRAIN_STEP} samples ({TRAIN_STEP/SAMPLING_RATE} seconds, {TRAIN_OVERLAP*100}% overlap)')
    print(f'Test step size: {TEST_STEP} samples ({TEST_STEP/SAMPLING_RATE} seconds, {TEST_OVERLAP*100}% overlap)')
    print(f'Add SMV channel: {USE_SMV_CHANNEL}')
    print(f'Number of folds: {N_FOLDS}\n')

    # Load metadata
    metadata = pd.read_csv(metadata_path)
    unique_participants = metadata['participant'].unique()
    print(f'Total participants: {len(unique_participants)}\n')

    # Process all participants with TRAIN overlap for training folds
    print('Processing all participants (50% overlap for training)...')
    train_windows_by_participant = {}
    train_labels_by_participant = {}

    for idx, row in metadata.iterrows():
        pid = row['participant']
        try:
            windows, labels, _ = process_participant_cnn(
                pid, data_path, step_size=TRAIN_STEP
            )

            train_windows_by_participant[pid] = windows
            train_labels_by_participant[pid] = labels['count']

            print(f'  {pid}: {len(windows)} windows (train overlap)')
        except Exception as e:
            print(f'  {pid}: Error - {e}')

    # Process all participants with TEST overlap (no overlap) for test folds
    print('\nProcessing all participants (0% overlap for testing)...')
    test_windows_by_participant = {}
    test_labels_by_participant = {}

    for idx, row in metadata.iterrows():
        pid = row['participant']
        try:
            windows, labels, _ = process_participant_cnn(
                pid, data_path, step_size=TEST_STEP
            )

            test_windows_by_participant[pid] = windows
            test_labels_by_participant[pid] = labels['count']

            print(f'  {pid}: {len(windows)} windows (test overlap)')
        except Exception as e:
            print(f'  {pid}: Error - {e}')

    # Create GroupKFold splits
    print(f'\nCreating {N_FOLDS}-fold GroupKFold splits...')
    gkf = GroupKFold(n_splits=N_FOLDS)

    # Create dummy X and y for split (we only need participant groups)
    dummy_X = np.arange(len(unique_participants))
    dummy_y = np.zeros(len(unique_participants))
    groups = unique_participants

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save each fold
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(dummy_X, dummy_y, groups)):
        print(f'\n--- Fold {fold_idx} ---')

        train_pids = unique_participants[train_idx]
        test_pids = unique_participants[test_idx]

        print(f'  Train participants ({len(train_pids)}): {sorted(train_pids)}')
        print(f'  Test participants ({len(test_idx)}): {sorted(test_pids)}')

        # Combine training data for this fold
        X_train_list = []
        y_train_list = []
        train_participants_list = []

        for pid in train_pids:
            if pid in train_windows_by_participant:
                X_train_list.append(train_windows_by_participant[pid])
                y_train_list.append(train_labels_by_participant[pid])
                train_participants_list.extend([pid] * len(train_windows_by_participant[pid]))

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        train_participants = np.array(train_participants_list)

        # Combine test data for this fold
        X_test_list = []
        y_test_list = []
        test_participants_list = []

        for pid in test_pids:
            if pid in test_windows_by_participant:
                X_test_list.append(test_windows_by_participant[pid])
                y_test_list.append(test_labels_by_participant[pid])
                test_participants_list.extend([pid] * len(test_windows_by_participant[pid]))

        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        test_participants = np.array(test_participants_list)

        print(f'  Train: {X_train.shape}, {y_train.sum():.0f} total steps, {(y_train>0).sum()/len(y_train)*100:.1f}% with steps')
        print(f'  Test: {X_test.shape}, {y_test.sum():.0f} total steps, {(y_test>0).sum()/len(y_test)*100:.1f}% with steps')

        # Normalize using training fold statistics
        train_mean = X_train.mean(axis=(0, 1))
        train_std = X_train.std(axis=(0, 1))

        X_train_norm = (X_train - train_mean) / train_std
        X_test_norm = (X_test - train_mean) / train_std

        # Save fold data
        np.savez_compressed(
            output_dir / f'cnn_fold{fold_idx}_train.npz',
            X=X_train_norm,
            y_count=y_train,
            participants=train_participants
        )

        np.savez_compressed(
            output_dir / f'cnn_fold{fold_idx}_test.npz',
            X=X_test_norm,
            y_count=y_test,
            participants=test_participants
        )

        np.savez(
            output_dir / f'cnn_fold{fold_idx}_normalization.npz',
            mean=train_mean,
            std=train_std
        )

        print(f'  Saved: fold{fold_idx}_train.npz, fold{fold_idx}_test.npz, fold{fold_idx}_normalization.npz')

    # Save fold participant mapping
    channel_names = ['X', 'Y', 'Z']
    if USE_SMV_CHANNEL:
        channel_names.append('SMV')

    np.savez(
        output_dir / 'cnn_channel_info.npz',
        channel_names=channel_names
    )

    # Also create legacy single-split files for notebooks (for backward compatibility)
    print('\n' + '=' * 70)
    print('Creating legacy single-split files for notebooks...')
    print('=' * 70)

    from sklearn.model_selection import train_test_split

    # Use same random state as original notebook for reproducibility
    train_val_pids, test_pids_legacy = train_test_split(unique_participants, test_size=0.2, random_state=42)
    train_pids_legacy, val_pids_legacy = train_test_split(train_val_pids, test_size=0.25, random_state=42)

    print(f'Legacy split: Train={len(train_pids_legacy)}, Val={len(val_pids_legacy)}, Test={len(test_pids_legacy)} participants')

    # Create legacy train set
    X_train_legacy_list = []
    y_train_legacy_list = []
    train_participants_legacy_list = []

    for pid in train_pids_legacy:
        if pid in train_windows_by_participant:
            X_train_legacy_list.append(train_windows_by_participant[pid])
            y_train_legacy_list.append(train_labels_by_participant[pid])
            train_participants_legacy_list.extend([pid] * len(train_windows_by_participant[pid]))

    X_train_legacy = np.concatenate(X_train_legacy_list, axis=0)
    y_train_legacy = np.concatenate(y_train_legacy_list, axis=0)

    # Create legacy val set
    X_val_legacy_list = []
    y_val_legacy_list = []
    val_participants_legacy_list = []

    for pid in val_pids_legacy:
        if pid in train_windows_by_participant:
            X_val_legacy_list.append(train_windows_by_participant[pid])
            y_val_legacy_list.append(train_labels_by_participant[pid])
            val_participants_legacy_list.extend([pid] * len(train_windows_by_participant[pid]))

    X_val_legacy = np.concatenate(X_val_legacy_list, axis=0)
    y_val_legacy = np.concatenate(y_val_legacy_list, axis=0)

    # Create legacy test set
    X_test_legacy_list = []
    y_test_legacy_list = []
    test_participants_legacy_list = []

    for pid in test_pids_legacy:
        if pid in test_windows_by_participant:
            X_test_legacy_list.append(test_windows_by_participant[pid])
            y_test_legacy_list.append(test_labels_by_participant[pid])
            test_participants_legacy_list.extend([pid] * len(test_windows_by_participant[pid]))

    X_test_legacy = np.concatenate(X_test_legacy_list, axis=0)
    y_test_legacy = np.concatenate(y_test_legacy_list, axis=0)

    # Normalize using legacy train statistics
    legacy_train_mean = X_train_legacy.mean(axis=(0, 1))
    legacy_train_std = X_train_legacy.std(axis=(0, 1))

    X_train_legacy_norm = (X_train_legacy - legacy_train_mean) / legacy_train_std
    X_val_legacy_norm = (X_val_legacy - legacy_train_mean) / legacy_train_std
    X_test_legacy_norm = (X_test_legacy - legacy_train_mean) / legacy_train_std

    # Save legacy files
    np.savez_compressed(
        output_dir / 'cnn_train_data.npz',
        X=X_train_legacy_norm,
        y_count=y_train_legacy,
        participants=np.array(train_participants_legacy_list)
    )

    np.savez_compressed(
        output_dir / 'cnn_val_data.npz',
        X=X_val_legacy_norm,
        y_count=y_val_legacy,
        participants=np.array(val_participants_legacy_list)
    )

    np.savez_compressed(
        output_dir / 'cnn_test_data.npz',
        X=X_test_legacy_norm,
        y_count=y_test_legacy,
        participants=np.array(test_participants_legacy_list)
    )

    np.savez(
        output_dir / 'cnn_normalization.npz',
        mean=legacy_train_mean,
        std=legacy_train_std
    )

    print(f'  Saved legacy files: cnn_train_data.npz, cnn_val_data.npz, cnn_test_data.npz')

    print('\n' + '=' * 70)
    print('DATA PREPARATION COMPLETE!')
    print('=' * 70)
    print(f'\nFiles saved to: {output_dir.absolute()}')
    print(f'\nCreated:')
    print(f'  - {N_FOLDS}-fold GroupKFold splits (cnn_fold*.npz) for production training')
    print(f'  - Legacy single split (cnn_train/val/test_data.npz) for notebooks')
    print(f'\nEach fold has ~{len(unique_participants)//N_FOLDS} test participants.')
    print('\nDataset configuration:')
    print('  - Train: 50% overlap (more training data)')
    print('  - Test: 0% overlap (fair evaluation, no double-counting)')
    print(f'\nTo train on all folds, run:')
    print(f'  for fold in {{0..{N_FOLDS-1}}}; do python src/train.py --fold $fold; done')


if __name__ == '__main__':
    prepare_data()
