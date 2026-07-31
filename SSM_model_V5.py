import torch
import torch.nn as nn

from SSM_TU.src.model.sequence_layer import SequenceLayer


class SSMPhonemeModel(nn.Module):
    def __init__(
        self,
        d_state: list[int],
        d_in: list[int],
        d_out: list[int],
        num_of_ssm_layers: int,
        vocab_size: int = 62,
        ssm_dropout: float = 0.1,
        linear_dropout: float = 0.1,
        norm: bool = True,
        norm_type: str = "bn",
        act: str = "LeakyRELu",
        trainable_SkipLayer: bool = True,

        # Convolutional frontend parameters
        num_conv_layers: int = 0,
        raw_input_channel: int = 1, # the input channel of the first convolutional layer
        conv_out_channels: list[int] | None = None,
        conv_kernel_sizes: list[int] = [10],
        conv_strides: list[int] = [2],
        conv_paddings: list[int] = [0],
        conv_dilations: list[int] = [1],
        conv_bias: bool = False,
        conv_batch_norm: bool = True,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Validate SSM configuration
        # ---------------------------------------------------------
        assert num_of_ssm_layers == len(d_in), (
            "Length of d_in must match num_of_ssm_layers."
        )
        assert num_of_ssm_layers == len(d_state), (
            "Length of d_state must match num_of_ssm_layers."
        )
        assert num_of_ssm_layers == len(d_out), (
            "Length of d_out must match num_of_ssm_layers."
        )

        assert num_conv_layers >= 0, (
            "num_conv_layers must be greater than or equal to zero."
        )

        assert len(conv_out_channels) == num_conv_layers, (
            f"conv_out_channels contains {len(conv_out_channels)} elements, "
            f"but num_conv_layers is {num_conv_layers}. "
            "The number of output channels must be specified for every Conv1d layer."
        )

        assert len(conv_kernel_sizes) == num_conv_layers, (
            f"conv_kernel_sizes contains {len(conv_kernel_sizes)} elements, "
            f"but num_conv_layers is {num_conv_layers}. "
            "A kernel size must be specified for every Conv1d layer."
        )

        assert len(conv_strides) == num_conv_layers, (
            f"conv_strides contains {len(conv_strides)} elements, "
            f"but num_conv_layers is {num_conv_layers}. "
            "A stride must be specified for every Conv1d layer."
        )

        assert len(conv_paddings) == num_conv_layers, (
            f"conv_paddings contains {len(conv_paddings)} elements, "
            f"but num_conv_layers is {num_conv_layers}. "
            "A padding value must be specified for every Conv1d layer."
        )

        assert len(conv_dilations) == num_conv_layers, (
            f"conv_dilations contains {len(conv_dilations)} elements, "
            f"but num_conv_layers is {num_conv_layers}. "
            "A dilation value must be specified for every Conv1d layer."
        )

        self.num_conv_layers = num_conv_layers
        self.raw_input_channel = raw_input_channel
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.conv_paddings = conv_paddings
        self.conv_dilations = conv_dilations

        # ---------------------------------------------------------
        # Convolutional frontend
        # ---------------------------------------------------------
        self.conv_frontend = nn.ModuleList()

        if num_conv_layers > 0:
            if conv_out_channels is None:
                raise ValueError(
                    "conv_out_channels must be provided when "
                    "num_conv_layers is greater than zero."
                )

            if len(conv_out_channels) != num_conv_layers:
                raise ValueError(
                    "The length of conv_out_channels must equal "
                    "num_conv_layers."
                )

            # define the input channels based on the output channels
            self.conv_in_channels = [
                self.raw_input_channel,
                *self.conv_out_channels[:-1]
            ] # if conv_out_channels = [32,64,128,256] then conv_in_channels=[1,32,64,128]

            for in_channel, out_channel, kernel_size, stride, padding, dilation in zip(
                self.conv_in_channels,
                self.conv_out_channels,
                self.conv_kernel_sizes,
                self.conv_strides,
                self.conv_paddings,
                self.conv_dilations,

            ):
                block = nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        bias=conv_bias,
                    ),

                    nn.BatchNorm1d(out_channel)
                    if conv_batch_norm
                    else nn.Identity(),

                    nn.LeakyReLU(),
                )

                self.conv_frontend.append(block) # holds the nn.Sequential layers of Conv1d, BacthNorm1d and LeakyRelu for each defined input and output channel

            frontend_output_size = self.conv_out_channels[-1]

        else:
            # No convolutional frontend.
            frontend_output_size = raw_input_channel

        # The last convolutional output dimension must equal the input dimension of the first SSM layer.
        if d_in[0] != frontend_output_size:
            raise ValueError(
                f"The first SSM d_in value is {d_in[0]}, but the "
                f"convolutional frontend produces "
                f"{frontend_output_size} features."
            )

        # ---------------------------------------------------------
        # SSM layers
        # ---------------------------------------------------------
        self.ssm_layers = nn.ModuleList([
            SequenceLayer(
                d_in=current_d_in,
                d_state=current_d_state,
                d_out=current_d_out,
                norm=norm,
                norm_type=norm_type,
                dropout=ssm_dropout,
                act=act,
                trainable_SkipLayer=trainable_SkipLayer,
            )
            for current_d_in, current_d_state, current_d_out
            in zip(d_in, d_state, d_out)
        ])

        # ---------------------------------------------------------
        # Final classifier
        # ---------------------------------------------------------
        self.linear_dropout = nn.Dropout(linear_dropout)

        self.classifier = nn.Linear(
            in_features=d_out[-1],
            out_features=vocab_size,
        )

    def calculate_output_lengths(
        self,
        input_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate sequence lengths after all Conv1d layers.

        Parameters
        ----------
        input_lengths:
            Original waveform lengths, shape [batch].

        Returns
        -------
        output_lengths:
            Lengths after convolutional downsampling, shape [batch].
        """
        output_lengths = input_lengths.clone()

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
            Raw waveform tensor with shape:

                [batch, sequence_length, raw_input_channels]

            For mono waveform input:

                [batch, sequence_length, 1]

        Returns
        -------
        logits:
            Shape:

                [batch, reduced_sequence_length, vocab_size]
        """

        # Conv1d expects:
        # [batch, channels, sequence_length]
        x = x.transpose(1, 2)

        for conv_layer in self.conv_frontend:
            x = conv_layer(x)

        # SequenceLayer expects:
        # [batch, sequence_length, features]
        x = x.transpose(1, 2)

        for ssm_layer in self.ssm_layers:
            x = ssm_layer(x)

        x = self.linear_dropout(x)
        logits = self.classifier(x)

        return logits