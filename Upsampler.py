
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class Upsampler:
    """
    Upsamples batched CTC log probabilities along the temporal axis.

    Input:
        log_probs:            [B, T, C]
        valid_output_lengths: [B]

    Output:
        upsampled_log_probs:  [B, T_new, C]
        new_valid_lengths:    [B]
    """

    def __init__(self, 
                 method: str = "linear"):
        self.method = method.lower()

        self.methods = {
            "linear": self._linear_interpolation,
            "linear_pytorch": self._linear_interpolation_pytorch
        }

        if self.method not in self.methods:
            raise ValueError(
                f"Unknown upsampling method: {method}. "
                f"Available methods: {list(self.methods)}"
            )

    def __call__(
        self,
        log_probs: torch.Tensor,
        valid_output_lengths: torch.Tensor,
        factor: int = 2
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
            raise ValueError("Upsampling factor must be an integer >= 1")

        if log_probs.ndim != 3:
            raise ValueError("log_probs must have shape [B, T, C]")

        B, T, C = log_probs.shape

        if valid_output_lengths.ndim != 1 or valid_output_lengths.numel() != B:
            raise ValueError("valid_output_lengths must have shape [B]")

        lengths = valid_output_lengths.to(
            device=log_probs.device,
            dtype=torch.long
        )

        if torch.any(lengths < 1) or torch.any(lengths > T):
            raise ValueError("Valid output lengths must be between 1 and T")

        if factor == 1:
            return log_probs, valid_output_lengths

        upsampled = self.methods[self.method](
            log_probs=log_probs,
            valid_lengths=lengths,
            factor=factor
        )

        # Interpolation of log probabilities does not generally
        # preserve normalization. Normalize across classes again.
        upsampled = F.log_softmax(upsampled, dim=-1)

        new_valid_lengths = valid_output_lengths * factor

        return upsampled, new_valid_lengths

    def _linear_interpolation(
        self,
        log_probs: torch.Tensor,
        valid_lengths: torch.Tensor,
        factor: int
    ) -> torch.Tensor:
        """
        Padding-aware, vectorized linear interpolation.

        Each recording is interpolated independently according
        to its valid temporal length, without a Python batch loop.

        Uses endpoint-aligned interpolation (align_corners=True
        convention), with special handling for length-one inputs.
        """
        print("interpolation STARTED...")
        B, T, C = log_probs.shape

        new_T = T * factor
        new_lengths = valid_lengths * factor

        # [1, new_T]
        output_positions = torch.arange(
            new_T,
            device=log_probs.device,
            dtype=torch.float32
        ).unsqueeze(0)

        # [B, 1]
        old_last = (valid_lengths - 1).unsqueeze(1)
        new_last = (new_lengths - 1).unsqueeze(1)

        # Calculate the source position for each output timestep.
        # Every recording uses its own valid temporal length.
        source_positions = (
            output_positions
            * old_last.to(torch.float32)
            / new_last.clamp(min=1).to(torch.float32)
        )

        # Prevent reading beyond each recording's valid region.
        source_positions = torch.minimum(
            source_positions,
            old_last.to(torch.float32)
        )

        # Neighboring source indices.
        left_indices = source_positions.floor().long()

        right_indices = torch.minimum(
            left_indices + 1,
            old_last
        )

        # Interpolation weights.
        alpha = (
            source_positions - left_indices.to(torch.float32)
        ).to(dtype=log_probs.dtype)

        # Expand indices for all C phoneme classes.
        left_indices = left_indices.unsqueeze(-1).expand(-1, -1, C)
        right_indices = right_indices.unsqueeze(-1).expand(-1, -1, C)

        # Gather neighboring log-probability vectors.
        left_values = torch.gather(
            log_probs,
            dim=1,
            index=left_indices
        )

        right_values = torch.gather(
            log_probs,
            dim=1,
            index=right_indices
        )

        # Linear interpolation.
        upsampled = (
            (1.0 - alpha.unsqueeze(-1)) * left_values
            + alpha.unsqueeze(-1) * right_values
        )

        # Set padded output positions to zero.
        valid_mask = (
            output_positions < new_lengths.unsqueeze(1)
        )

        upsampled = upsampled.masked_fill(
            ~valid_mask.unsqueeze(-1),
            0.0
        )
        print("interpolation ENDED...")

        return upsampled

    
    def _linear_interpolation_pytorch(
        self,
        log_probs: torch.Tensor,
        valid_lengths: torch.Tensor,
        factor: int
    ) -> torch.Tensor:
        """
        Upsample each recording independently using PyTorch's
        built-in linear interpolation.

        Parameters
        ----------
        log_probs : torch.Tensor
            Shape [B, T, C], including padding.

        valid_lengths : torch.Tensor
            Shape [B], valid temporal length of each recording.

        factor : int
            Upsampling factor.

        Returns
        -------
        torch.Tensor
            Shape [B, T_new_max, C], padded to the longest
            upsampled recording.
        """

        upsampled_sequences = []

        for i, length in enumerate(valid_lengths):

            length = int(length.item())

            # Extract only the valid part of the recording.
            sample = log_probs[i, :length, :]  # [T_valid, C]

            # PyTorch interpolation expects [N, C, T].
            sample = sample.transpose(0, 1).unsqueeze(0)

            # Upsample temporal dimension.
            upsampled = F.interpolate(
                sample,
                scale_factor=factor,
                mode="linear",
                align_corners=True
            )

            # Convert back to [T_new, C].
            upsampled = upsampled.squeeze(0).transpose(0, 1)

            upsampled_sequences.append(upsampled)

        # Pad all upsampled recordings to equal temporal length.
        upsampled_batch = pad_sequence(
            upsampled_sequences,
            batch_first=True,
            padding_value=0.0
        )

        return upsampled_batch


