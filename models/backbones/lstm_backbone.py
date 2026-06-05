from typing import Sequence
import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    """
    Simple LSTM backbone that receives observation embeddings and outputs features for the heads

    Args:
        input_dim: Dimension size of the input features: (B, input_dim)
        feature_dim: Dimenson of the LSTM hidden units feature size and also the feature dimension fed into heads: (B, feature_dim)
    """

    def __init__(self, input_dim: int, feature_dim: int, num_layers: int):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_dim,
                                hidden_size=feature_dim,
                                num_layers=num_layers,
                                )
        
    # --- INSPIRATION: Code seen on https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/,
    # on how stable-baselines3 implemented LSTM layers
    def forward(self, input_feats: torch.Tensor, lstm_state: tuple[torch.Tensor, torch.Tensor], done: torch.Tensor):

        batch_size = lstm_state[0].shape[1]
        hidden = input_feats.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h,d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0), # Shape: (1, batch_size, feature_dim)
                (
                (1.0 - d).view(1,-1,1) * lstm_state[0],
                (1.0 - d).view(1,-1,1) * lstm_state[1],
                ),
            )
            new_hidden += [h]

        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state
    

class NAVA3C(nn.Module):

    """
    LSTM Backbone for the NAVA3C architecture from the paper:

        Learning to Navigate in Complex Environments: https://arxiv.org/abs/1611.03673

    """
    def __init__(self,
                input_dim: int,
                feature_dim: Sequence[int],
                num_layers: Sequence[int],
                velocity_dim: int,
                last_action_dim: int
                ):
        
        super().__init__() 
        
        # input_dim + 1 since it receives observation embeddings + last reward
        self.first_lstm = SimpleLSTM(input_dim=input_dim + 1, feature_dim=feature_dim[0], num_layers=num_layers[0])

        # Takes in observation embeddings, hidden state of the first lstm, current velocity and last action
        self.second_lstm = SimpleLSTM(input_dim=input_dim + feature_dim[0] + velocity_dim + last_action_dim,
                                      feature_dim=feature_dim[1],
                                      num_layers=num_layers[1])
        
    def forward(self,
                    input_feats: torch.Tensor,
                    lstm_states: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
                    dones: torch.Tensor,
                    last_rewards: torch.Tensor,
                    velocities: torch.Tensor,
                    last_actions: torch.Tensor
            ):

            lstm_state_1, lstm_state_2 = lstm_states

            first_input = torch.cat([input_feats, last_rewards], dim=-1)

            new_hidden, lstm_state_1 = self.first_lstm(first_input, lstm_state_1, dones)

            second_input = torch.cat([input_feats, new_hidden, velocities, last_actions], dim=-1)

            new_hidden, lstm_state_2 = self.second_lstm(second_input, lstm_state_2, dones)

            return new_hidden, (lstm_state_1, lstm_state_2)
            








        



