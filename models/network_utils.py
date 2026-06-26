from typing import Sequence
import torch
import torch.nn as nn

def build_mlp(
        input_dim: int,
        hidden_sizes: Sequence[int],
        output_dim: int,
        activation=nn.ReLU,
        output_activation=nn.Identity        
        ) -> nn.Sequential:
    
        layers = []
        curr_layer = input_dim

        for hid_size in hidden_sizes:
            layers.extend([nn.Linear(curr_layer, hid_size), activation()])
            curr_layer = hid_size

        layers.extend([nn.Linear(curr_layer, output_dim), output_activation()])

        return nn.Sequential(*layers)

def cnn_block(
            input_channels: int,
            output_channels: int,
        ):
    
        return [
            
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
      ]

def build_cnn(
          input_channels: int,
          out_channels: int,
    ):
    blocks = []
    current_in = input_channels
    for i in reversed(range(3)):
        current_out = max(current_in, out_channels // (2 ** i))
        blocks.extend(cnn_block(input_channels=current_in, output_channels=current_out))
        current_in = current_out
    return nn.Sequential(*blocks)


def build_cnn1d(input_channels: int, out_channels: int) -> nn.Sequential:
    """1-D CNN for ray observations (B, C, num_rays). Global-average-pools to (B, out_channels)."""
    return nn.Sequential(
        nn.Conv1d(input_channels, out_channels // 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(out_channels // 2, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),   # (B, out_channels, 1)
    )




