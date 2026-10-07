import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Wav2VecConvFeatureEncoder(nn.Module):
    """
    Wav2vec-style convolutional feature encoder.

    The first Conv1d layer is followed by GroupNorm.
    The remaining Conv1d layers do not use normalization.

    Input shape
    -----------
    [B, T, input_channels]

    For mono waveform input:
    [B, T, 1]

    Output shape
    ------------
    [B, T_reduced, conv_out_channels[-1]]
    """

    def __init__(
        self,
        input_channels: int,
        conv_out_channels: list[int],
        conv_kernel_sizes: list[int],
        conv_strides: list[int],
        conv_paddings: list[int],
        conv_dilations: list[int],
        conv_bias: bool = False,
        conv_dropout: float = 0.0,
    ):
        super().__init__()

        # Example:
        #
        # input_channels = 1
        # conv_out_channels = [64, 128, 256]
        #
        # conv_in_channels becomes:
        # [1, 64, 128]
        conv_in_channels = [
            input_channels,
            *conv_out_channels[:-1]
        ]

        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.conv_paddings = conv_paddings
        self.conv_dilations = conv_dilations

        self.conv_layers = nn.ModuleList()

        for layer_index, (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation
        ) in enumerate(
            zip(
                conv_in_channels,
                conv_out_channels,
                conv_kernel_sizes,
                conv_strides,
                conv_paddings,
                conv_dilations
            )
        ):

            convolution = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=conv_bias
            )

            # The first convolutional layer uses GroupNorm.
            if layer_index == 0:
                conv_block = nn.Sequential(
                    convolution,

                    nn.GroupNorm(
                        num_groups=out_channels,
                        num_channels=out_channels
                    ),

                    nn.GELU(),

                    nn.Dropout(conv_dropout)
                )

            # All remaining layers use no normalization.
            else:
                conv_block = nn.Sequential(
                    convolution,
                    nn.GELU(),
                    nn.Dropout(conv_dropout)
                )

            self.conv_layers.append(conv_block)

    def calculate_output_lengths(
        self,
        input_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the valid sequence lengths after all Conv1d layers.

        Parameters
        ----------
        input_lengths:
            Original unpadded waveform lengths.

            Shape:
            [B]

        Returns
        -------
        output_lengths:
            Valid lengths after the convolutional encoder.

            Shape:
            [B]
        """

        output_lengths = input_lengths.clone().long()

        for kernel_size, stride, padding, dilation in zip(
            self.conv_kernel_sizes,
            self.conv_strides,
            self.conv_paddings,
            self.conv_dilations
        ):
            output_lengths = (
                output_lengths
                + 2 * padding
                - dilation * (kernel_size - 1)
                - 1
            ) // stride + 1

        return output_lengths

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Raw waveform tensor.

            Shape:
            [B, T, input_channels]

        Returns
        -------
        x:
            Learned acoustic feature sequence.

            Shape:
            [B, T_reduced, conv_out_channels[-1]]
        """

        # Conv1d expects [B, channels, sequence_length].
        #
        # [B, T, 1] -> [B, 1, T]
        x = x.transpose(1, 2)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Return to sequence representation.
        #
        # [B, channels, T_reduced]
        # ->
        # [B, T_reduced, channels]
        x = x.transpose(1, 2)

        return x

# used for the transforer version
class PositionalConvEmbedding(nn.Module):
    """
    Wav2Vec 2.0 positional convolutional embedding.

    Adds learnable positional information to the feature vectors
    produced by the convolutional feature encoder.

    Input
    -----
    [batch_size, sequence_length, embedding_dimension]

    Output
    ------
    [batch_size, sequence_length, embedding_dimension]
    """

    def __init__(
        self,
        embedding_dimension: int,
        kernel_size: int = 128,
        groups: int = 16,
        use_weight_norm: bool = True,
    ):
        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.kernel_size = kernel_size
        self.groups = groups

        # ---------------------------------------------------------
        # Positional convolution
        # ---------------------------------------------------------
        conv = nn.Conv1d(
            in_channels=embedding_dimension,
            out_channels=embedding_dimension,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=groups,
            bias=True,
        )

        # Optional Weight Normalization
        if use_weight_norm:
            conv = weight_norm(conv)

        self.conv = conv

        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Shape:
                [batch_size, sequence_length, embedding_dimension]

        Returns
        -------
        torch.Tensor
            Shape:
                [batch_size, sequence_length, embedding_dimension]
        """

        # Save the original features for the residual connection.
        residual = x

        # Conv1d expects:
        # [batch_size, embedding_dimension, sequence_length]
        x = x.transpose(1, 2)

        x = self.conv(x)

        # ---------------------------------------------------------
        # Even kernel sizes (e.g. 128) produce one additional
        # timestep because of the symmetric padding.
        #
        # Remove the last timestep so that the output sequence
        # length matches the input sequence length.
        # ---------------------------------------------------------
        if self.kernel_size % 2 == 0:
            x = x[:, :, :-1]

        x = self.activation(x)

        # Back to
        # [batch_size, sequence_length, embedding_dimension]
        x = x.transpose(1, 2)

        # Residual connection
        x = x + residual

        return x



# used for the transformer version
class Wav2VecFeatureProjection(nn.Module):
    """
    Projects the output of the convolutional feature encoder into
    the embedding dimension expected by the Transformer.

    Input:
        [B, T_reduced, conv_output_dimension]

    Output:
        [B, T_reduced, transformer_embedding_dimension]
    """

    def __init__(
        self,
        conv_output_dimension: int,
        transformer_embedding_dimension: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(
            conv_output_dimension
        )

        self.projection = nn.Linear(
            in_features=conv_output_dimension,
            out_features=transformer_embedding_dimension
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)

        return x

class Wav2VecTransformerEncoder(nn.Module):
    """
    Stack of Transformer encoder layers for contextualizing the
    acoustic feature sequence.

    Input and output:
        [B, T_reduced, embedding_dimension]
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_layers: int,
        number_of_attention_heads: int,
        feedforward_dimension: int,
        dropout: float = 0.1
    ):
        super().__init__()

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=number_of_attention_heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=number_of_layers,
            norm=nn.LayerNorm(embedding_dimension)
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        x = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        return x

