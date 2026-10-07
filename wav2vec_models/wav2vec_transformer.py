import torch
import torch.nn as nn

from wav2vec_layers import (
    Wav2VecConvFeatureEncoder,
    Wav2VecFeatureProjection,
    Wav2VecTransformerEncoder,
    PositionalConvEmbedding,
)


class Wav2VecTransformerEncoderModel(nn.Module):
    """
    Shared wav2vec-inspired Transformer encoder.

    Architecture
    ------------
    Raw waveform
        -> convolutional feature encoder
        -> feature projection
        -> optional feature masking
        -> positional convolutional embedding
        -> Transformer encoder

    This class does not contain:
        - a phoneme classifier
        - CTC-specific output logic
        - a contrastive pretraining head
    """

    def __init__(
        self,

        # Convolutional feature encoder
        input_channels: int,
        conv_out_channels: list[int],
        conv_kernel_sizes: list[int],
        conv_strides: list[int],
        conv_paddings: list[int],
        conv_dilations: list[int],
        conv_bias: bool = False,
        conv_dropout: float = 0.0,

        # Feature projection
        transformer_embedding_dimension: int = 256,
        feature_projection_dropout: float = 0.1,

        # Positional convolution
        positional_kernel_size: int = 128,
        positional_groups: int = 16,
        positional_use_weight_norm: bool = True,

        # Transformer
        number_of_transformer_layers: int = 4,
        number_of_attention_heads: int = 4,
        feedforward_dimension: int = 1024,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()

        self.transformer_embedding_dimension = transformer_embedding_dimension

        # ---------------------------------------------------------
        # Convolutional feature encoder
        # ---------------------------------------------------------
        self.feature_encoder = Wav2VecConvFeatureEncoder(
            input_channels=input_channels,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            conv_paddings=conv_paddings,
            conv_dilations=conv_dilations,
            conv_bias=conv_bias,
            conv_dropout=conv_dropout,
        )

        conv_output_dimension = conv_out_channels[-1]

        # ---------------------------------------------------------
        # Feature projection
        # ---------------------------------------------------------
        self.feature_projection = Wav2VecFeatureProjection(
            conv_output_dimension=conv_output_dimension,
            transformer_embedding_dimension=(
                transformer_embedding_dimension
            ),
            dropout=feature_projection_dropout,
        )

        # ---------------------------------------------------------
        # Positional convolutional embedding
        # ---------------------------------------------------------
        self.positional_embedding = PositionalConvEmbedding(
            embedding_dimension=transformer_embedding_dimension,
            kernel_size=positional_kernel_size,
            groups=positional_groups,
            use_weight_norm=positional_use_weight_norm,
        )

        # ---------------------------------------------------------
        # Transformer encoder
        # ---------------------------------------------------------
        self.transformer_encoder = Wav2VecTransformerEncoder(
            embedding_dimension=transformer_embedding_dimension,
            number_of_layers=number_of_transformer_layers,
            number_of_attention_heads=number_of_attention_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=transformer_dropout,
        )

    def calculate_output_lengths(
        self,
        waveform_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate valid sequence lengths after the convolutional
        feature encoder.
        """

        return self.feature_encoder.calculate_output_lengths(
            waveform_lengths
        )

    @staticmethod
    def create_padding_mask(
        sequence_lengths: torch.Tensor,
        maximum_sequence_length: int
    ) -> torch.Tensor:
        """
        Create a boolean Transformer padding mask.

        False:
            Valid feature timestep.

        True:
            Padded timestep that should be ignored.
        """

        time_indices = torch.arange(
            maximum_sequence_length,
            device=sequence_lengths.device
        ).unsqueeze(0)

        return time_indices >= sequence_lengths.unsqueeze(1)

    def extract_projected_features(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        """
        Run the convolutional encoder and feature projection, but
        stop before masking, positional embedding, and Transformer.

        Returns
        -------
        projected_features:
            Shape [B, T_reduced, transformer_embedding_dimension]

        output_lengths:
            Shape [B]

        padding_mask:
            Shape [B, T_reduced]
        """

        # [B, T_raw, 1]
        # ->
        # [B, T_reduced, conv_output_dimension]
        conv_features = self.feature_encoder(waveforms)

        output_lengths = self.calculate_output_lengths(
            waveform_lengths
        )

        output_lengths = torch.clamp(
            output_lengths,
            min=0,
            max=conv_features.shape[1]
        )

        # [B, T_reduced, conv_output_dimension]
        # ->
        # [B, T_reduced, transformer_embedding_dimension]
        projected_features = self.feature_projection(
            conv_features
        )

        padding_mask = self.create_padding_mask(
            sequence_lengths=output_lengths,
            maximum_sequence_length=projected_features.shape[1]
        )

        return (
            projected_features,
            output_lengths,
            padding_mask
        )

    def contextualize_features(
        self,
        projected_features: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Add positional information and process the sequence with
        the Transformer.
        """

        x = self.positional_embedding(
            projected_features
        )

        contextual_features = self.transformer_encoder(
            x,
            padding_mask=padding_mask
        )

        return contextual_features

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Standard unmasked encoder forward pass.

        Returns
        -------
        contextual_features:
            Shape [B, T_reduced, transformer_embedding_dimension]

        output_lengths:
            Shape [B]
        """

        (
            projected_features,
            output_lengths,
            padding_mask
        ) = self.extract_projected_features( # conv1D frontend -> feature projector
            waveforms,
            waveform_lengths
        )

        contextual_features = self.contextualize_features( # positional embedding -> padding mask -> transformer
            projected_features,
            padding_mask
        )

        return contextual_features, output_lengths


class Wav2VecTransformerFineTuningModel(nn.Module):
    """
    Wav2vec-inspired model for supervised phoneme recognition.

    Shared encoder
        -> dropout
        -> phoneme classifier
    """

    def __init__(
        self,
        encoder: Wav2VecTransformerEncoderModel,
        vocab_size: int,
        classifier_dropout: float = 0.1
    ):
        super().__init__()

        self.encoder = encoder

        self.classifier_dropout = nn.Dropout(
            classifier_dropout
        )

        self.classifier = nn.Linear(
            in_features=encoder.transformer_embedding_dimension,
            out_features=vocab_size
        )

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        contextual_features, output_lengths = self.encoder(
            waveforms,
            waveform_lengths
        )

        contextual_features = self.classifier_dropout(
            contextual_features
        )

        logits = self.classifier(
            contextual_features
        )

        return logits, output_lengths

class Wav2VecTransformerPretrainingModel(nn.Module):
    """
    Wav2vec-inspired Transformer pretraining model.

    This model:
        1. Extracts projected acoustic features.
        2. Randomly selects valid feature positions.
        3. Replaces selected positions with one learned mask embedding.
        4. Passes the masked sequence through the positional embedding
           and Transformer.
        5. Returns both the Transformer outputs and the original
           unmasked features.

    Important
    ---------
    This class implements the masking and contextualization stage,
    but not yet the complete wav2vec 2.0 objective.

    Still required:
        - vector quantizer
        - negative sampling
        - contrastive loss
        - diversity loss
    """

    def __init__(
        self,
        encoder: Wav2VecTransformerEncoderModel,
        mask_probability: float = 0.065,
    ):
        super().__init__()

        self.encoder = encoder
        self.mask_probability = mask_probability

        # One trainable vector used to replace every deliberately
        # masked feature vector.
        #
        # Shape:
        # [transformer_embedding_dimension]
        self.mask_embedding = nn.Parameter(
            torch.empty(
                encoder.transformer_embedding_dimension
            )
        )

        nn.init.uniform_(
            self.mask_embedding,
            a=-0.1,
            b=0.1
        )

    def create_feature_mask(
        self,
        output_lengths: torch.Tensor,
        maximum_sequence_length: int
    ) -> torch.Tensor:
        """
        Randomly select valid feature positions for masking.

        Parameters
        ----------
        output_lengths:
            Number of valid post-convolution timesteps for every
            sample.

            Shape:
            [B]

        maximum_sequence_length:
            Padded feature-sequence length.

        Returns
        -------
        feature_mask:
            Boolean tensor with shape [B, T_reduced].

            False:
                Keep the original feature vector.

            True:
                Replace this feature vector with mask_embedding.
        """

        # Create timestep numbers:
        #
        # [0, 1, 2, ..., T_reduced - 1]
        #
        # Shape after unsqueeze:
        # [1, T_reduced]
        timestep_indices = torch.arange(
            maximum_sequence_length,
            device=output_lengths.device
        ).unsqueeze(0)

        # True for real feature positions.
        # False for padding.
        #
        # Shape:
        # [B, T_reduced]
        valid_positions = (
            timestep_indices
            < output_lengths.unsqueeze(1)
        )

        # Generate one random number in [0, 1) for every position.
        #
        # Shape:
        # [B, T_reduced]
        random_values = torch.rand(
            valid_positions.shape,
            device=output_lengths.device
        )

        # Select positions according to mask_probability.
        randomly_selected_positions = (
            random_values < self.mask_probability
        )

        # A position is masked only when:
        #
        # 1. it was randomly selected, and
        # 2. it is a real feature, not padding.
        feature_mask = (
            randomly_selected_positions
            & valid_positions
        )

        return feature_mask

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        waveforms:
            Padded raw waveform batch.

            Shape:
            [B, T_raw, 1]

        waveform_lengths:
            Original unpadded waveform lengths.

            Shape:
            [B]

        Returns
        -------
        Dictionary containing:

        contextual_features:
            Transformer outputs created from the masked input.

            Shape:
            [B, T_reduced, D]

        target_features:
            Original projected features before deliberate masking.

            Shape:
            [B, T_reduced, D]

        masked_features:
            Projected features after the selected vectors were replaced
            by the learned mask embedding.

            Shape:
            [B, T_reduced, D]

        feature_mask:
            Boolean mask indicating which valid positions were
            deliberately hidden.

            Shape:
            [B, T_reduced]

        padding_mask:
            Boolean mask indicating which positions are padding.

            Shape:
            [B, T_reduced]

        output_lengths:
            Number of valid post-convolution timesteps per sample.

            Shape:
            [B]
        """

        # ---------------------------------------------------------
        # 1. Convolutional feature extraction and feature projection
        # ---------------------------------------------------------
        (
            projected_features,
            output_lengths,
            padding_mask
        ) = self.encoder.extract_projected_features(
            waveforms,
            waveform_lengths
        )

        # projected_features:
        # [B, T_reduced, D]
        #
        # output_lengths:
        # [B]
        #
        # padding_mask:
        # [B, T_reduced]

        # Keep the original projected features.
        #
        # These represent the unmasked acoustic information that the
        # pretraining objective will later use as the basis for its
        # targets.
        target_features = projected_features

        # ---------------------------------------------------------
        # 2. Select valid positions for deliberate masking
        # ---------------------------------------------------------
        feature_mask = self.create_feature_mask(
            output_lengths=output_lengths,
            maximum_sequence_length=projected_features.shape[1]
        )

        # ---------------------------------------------------------
        # 3. Replace selected vectors with the learned mask vector
        # ---------------------------------------------------------

        # Clone the tensor so we do not overwrite target_features.
        masked_features = projected_features.clone()

        # Boolean indexing selects complete D-dimensional vectors.
        #
        # Every selected vector is replaced with the same learned
        # mask embedding.
        masked_features[feature_mask] = (
            self.mask_embedding.to(
                dtype=masked_features.dtype
            )
        )

        # ---------------------------------------------------------
        # 4. Add positional information and run the Transformer
        # ---------------------------------------------------------
        contextual_features = (
            self.encoder.contextualize_features(
                projected_features=masked_features,
                padding_mask=padding_mask
            )
        )

        return {
            "contextual_features": contextual_features,
            "target_features": target_features,
            "masked_features": masked_features,
            "feature_mask": feature_mask,
            "padding_mask": padding_mask,
            "output_lengths": output_lengths,
        }