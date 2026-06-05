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
            layers.extend(nn.Linear(curr_layer, hid_size), activation())
            curr_layer = hid_size

        layers.extend(nn.Linear(curr_layer, output_dim),output_activation())

        return nn.Sequential(*layers)




