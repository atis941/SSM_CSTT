import torch
import torch.nn as nn
from SSM_TU.src.model.sequence_layer import SequenceLayer
import torch.nn.functional as F

###### SSM model #####
class SSMPhonemeModel(nn.Module):
    def __init__(self, 
                 d_state: list[int], 
                 d_in: list[int],
                 d_out: list[int],
                 num_of_ssm_layers: int,
                 linear_in_features: list[int],
                 linear_out_features: list[int],
                 num_of_linear_layers: int,
                 vocab_size: int = 62, 
                 ssm_dropout: float = 0.1,
                 linear_dropout: float = 0.1):
        super().__init__()
        
        # checking dimensional and list value requirements for the input parameters
        assert num_of_ssm_layers == len(d_in), "d_in dimension must match the num_of_ssm_layers parameter"
        assert num_of_ssm_layers == len(d_state), "d_state dimension must match the num_of_ssm_layers parameter"
        assert num_of_ssm_layers == len(d_out), "d_out dimension must match the num_of_ssm_layers parameter"

        assert num_of_linear_layers == len(linear_in_features), "linear_in_features length must match the num_of_linear_layers parameter"
        assert num_of_linear_layers == len(linear_out_features), "linear_out_features length must match the num_of_linear_layers parameter"

        self.ssm_layers = nn.ModuleList([
            SequenceLayer(
                d_in=d_in,
                d_state=d_state,
                d_out=d_out,
                norm=True,
                norm_type='bn',
                dropout=ssm_dropout,
                act='LeakyRELu',
                trainable_SkipLayer=False
            ) for d_in, d_state, d_out in zip(d_in, d_state, d_out)
        ])

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=lin_f_in,
                      out_features=lin_f_out) for lin_f_in, lin_f_out in zip(linear_in_features, linear_out_features) 
        ])

        self.lin_dropout1 = nn.Dropout(linear_dropout)
        # classifier over phoneme vocabulary
        # applied at every timestep
        self.classifier = nn.Linear(in_features=linear_out_features[-1], 
                                    out_features=vocab_size)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward operation

        Parameters
        ----------
        x: torch.Tensor
            shape (batch, seq_len, d_in)

        Returns
        -------
        logits: torch.Tensor
            logits of shape (batch, seq_len, vocab_size)
        """
        for linear_projection, ssm_layer in zip(self.linear_layers, self.ssm_layers):
            x = ssm_layer(x)
            x = linear_projection(x)
            x = F.relu(x)

        x = self.lin_dropout1(x) # using a dropout for the last linear layer
        logits = self.classifier(x)   # (batch, seq_len, vocab_size)
        return logits