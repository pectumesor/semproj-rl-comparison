"""The grid cell network architecture (Extended Data
Fig. 1) consists of three layers: a recurrent layer, a linear layer, and an output layer.
The single recurrent layer is an LSTM (long short-term memory) that projects
to place and head direction units via the linear layer. The linear layer implements
regularization through dropout. The recurrent LSTM layer consists of one cell
of 128 hidden units, with no peephole connections.

Described on the paper: Vector-based navigation using grid-like representations in artificial agents

"""

import torch
import torch.nn as nn


class GridCellNetwork(nn.Module):
    def __init__(self, dropout: float, inital_pos: torch.Tensor, initial_head_dir: torch.Tensor,
                 place_cell_size: int, head_dir_cell_size: int,
                batch_size: int, linear_sizes : list[int] = [512, 256, 2], hidden_size: int = 128):
        super().__init__()
    

        self.initial_lstm_state = (
            torch.zeros((1, batch_size, hidden_size), dtype=torch.float),
            torch.zeros((1, batch_size, hidden_size), dtype=torch.float)
        )

        self.l0 = inital_pos
        self.m0 = initial_head_dir

        self.linear_l0_place = nn.Linear(place_cell_size, hidden_size, bias=False)
        self.linear_l0_head = nn.Linear(head_dir_cell_size, hidden_size, bias=False)

        self.linear_m0_place = nn.Linear(place_cell_size, hidden_size, bias=False)
        self.linear_m0_head = nn.Linear(head_dir_cell_size, hidden_size, bias=False)

        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, num_layers=1) # Receives [v, sin(phi_t), cos(phi_t)]

        current_size = hidden_size
        layers = []
        for size in linear_sizes:
            layers += [
                nn.Linear(current_size, size),
                nn.Dropout(p=dropout)
            ]
        
        self.linear_decoder = nn.Sequential(*layers)

    def initialize_lstm(self, init_place_cell: torch.Tensor, init_head_cell: torch.Tensor):

        l0 = self.linear_l0_place(init_place_cell) + self.linear_l0_head(init_head_cell)
        m0 = self.linear_m0_place(init_place_cell) + self.linear_m0_head(init_head_cell)
        self.initial_lstm_state = (l0, m0)


    def forward(self, x):

        hidden_state, cell_state = self.lstm(x)

        
