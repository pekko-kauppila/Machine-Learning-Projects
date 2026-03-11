"""
Shallow 1D CNN architecture for step counting.

Simple baseline with 1 convolutional layer followed by dense layers.
Architecture: Conv1D → ReLU → MaxPool → Dropout → Dense → Dense
"""

import torch
import torch.nn as nn


class ShallowCNN(nn.Module):
    def __init__(self, input_channels, sequence_length, num_filters=64, kernel_size=5, pool_size=2,
                 output_type='binary', dropout_rate=0.5):
        super(ShallowCNN, self).__init__()

        self.output_type = output_type

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate the size after conv and pooling
        conv_output_length = sequence_length // pool_size
        flattened_size = num_filters * conv_output_length

        # Dense layers
        self.linear1 = nn.Linear(flattened_size, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layers
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)

        # Apply activation based on task type
        if self.output_type == 'binary':
            x = self.sigmoid(x)
        # For regression: no activation (model can output any value)

        return x.squeeze(-1)
