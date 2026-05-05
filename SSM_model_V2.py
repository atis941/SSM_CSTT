import torch
import torch.nn as nn
from SSM_TU.src.model.sequence_layer import SequenceLayer

###### SSM model #####
class SSMPhonemeModel(nn.Module):
    def __init__(self, 
                 d_state: int, 
                 d_in: int,
                 vocab_size: int = 62, 
                 dropout: float = 0.1,
                 num_layers: int = 3):
        super().__init__()

        # input is (batch, seq_len, d_in) => so d_in = n_mels in mel spectrogram
        # d_out = 80 as well, so residual connections work naturally
        self.num_layers = num_layers

        self.ssm_layers = nn.ModuleList([
            SequenceLayer(
                d_in=d_in,
                d_state=d_state,
                d_out=d_in,
                norm=True,
                norm_type='bn',
                dropout=dropout,
                act='LeakyRELu',
                trainable_SkipLayer=False
            )
            for _ in range(num_layers)
        ])

        # classifier over phoneme vocabulary
        # applied at every timestep
        self.classifier = nn.Linear(in_features=d_in, 
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
        for layer in self.ssm_layers:
            x = layer(x)

        logits = self.classifier(x)   # (batch, seq_len, vocab_size)
        return logits