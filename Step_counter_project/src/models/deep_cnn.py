"""
Deeper 1D CNN architecture with more layers.

Increased capacity to learn complex temporal patterns in accelerometer data.
"""

import torch
import torch.nn as nn


class DeepCNN(nn.Module):
    """
    Deeper 1D CNN with 5 convolutional blocks for step count regression.

    Architecture:
        - 5 Conv1D blocks with batch norm and pooling
        - Global average pooling
        - Dense layers with dropout
        - Output layer

    Args:
        input_channels: Number of input channels (default: 4)
        sequence_length: Length of input sequence (default: 200)
        num_filters: Base number of filters (progressively increased)
        dropout_rate: Dropout rate for regularization
    """

    def __init__(self, input_channels, sequence_length, num_filters=64, dropout_rate=0.5):
        super(DeepCNN, self).__init__()

        # Note: sequence_length is accepted for API consistency but not used
        # since we use adaptive pooling

        # Conv Block 1
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout_conv1 = nn.Dropout(dropout_rate * 0.5)  # Lower dropout in early conv layers

        # Conv Block 2
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout_conv2 = nn.Dropout(dropout_rate * 0.5)

        # Conv Block 3
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(num_filters * 4)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout_conv3 = nn.Dropout(dropout_rate * 0.5)

        # Conv Block 4
        self.conv4 = nn.Conv1d(num_filters * 4, num_filters * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(num_filters * 8)
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        self.dropout_conv4 = nn.Dropout(dropout_rate * 0.5)

        # Conv Block 5
        self.conv5 = nn.Conv1d(num_filters * 8, num_filters * 8, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(num_filters * 8)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Dense layers
        self.dense1 = nn.Linear(num_filters * 8, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dense3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layer
        self.output = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout_conv1(x)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout_conv2(x)

        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout_conv3(x)

        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = self.dropout_conv4(x)

        # Conv Block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.global_pool(x)

        # Flatten
        x = x.squeeze(-1)

        # Dense layers
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.dense3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        # Output
        x = self.output(x)

        return x.squeeze(-1)
