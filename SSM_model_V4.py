import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Identity

from SSM_TU.src.model.sequence_layer import SequenceLayer
from SSM_TU.src.model.random_zero_order_hold import RandomZeroOrderHold


class SSMPhonemeModel(nn.Module):

    def __init__(
        self,
        d_state: list[int],
        d_in: list[int],
        d_out: list[int],
        num_of_ssm_layers: int,
        input_bias: bool = False,
        output_bias: bool = False,
        complex_output: bool = False,
        B_C_init: str = "orthogonal",
        bias_init: str = "zero",
        stability: str = "abs",
        vocab_size: int = 62,
        ssm_dropout: float = 0.1,
        linear_dropout: float | None = None,
        norm: bool = True,
        norm_type: str = "bn",
        act: str = "LeakyRELu",
        trainable_SkipLayer: bool = True,
        rzoh_probabilities: list[float] | None = None,
        
    ):
        super().__init__()

        # checking dimensional and list value requirements
        assert num_of_ssm_layers == len(
            d_in
        ), "d_in dimension must match the num_of_ssm_layers parameter"

        assert num_of_ssm_layers == len(
            d_state
        ), "d_state dimension must match the num_of_ssm_layers parameter"

        assert num_of_ssm_layers == len(
            d_out
        ), "d_out dimension must match the num_of_ssm_layers parameter"

        # create the SSM layers
        self.ssm_layers = nn.ModuleList([
            SequenceLayer(
                d_in=d_in_i,
                d_state=d_state_i,
                d_out=d_out_i,
                input_bias=input_bias,
                output_bias=output_bias,
                complex_output=complex_output,
                B_C_init = B_C_init,
                bias_init = bias_init,
                stability = stability,
                norm=norm,
                norm_type=norm_type,
                dropout=ssm_dropout,
                act=act,
                trainable_SkipLayer=trainable_SkipLayer,
            )
            for d_in_i, d_state_i, d_out_i in zip(d_in, d_state, d_out)
        ])

        # create the Linear dropout layer
        if linear_dropout is None:
            self.lin_dropout1 = nn.Identity()
        else:
            self.lin_dropout1 = nn.Dropout(linear_dropout)

        # create the last Linear classifier layer
        self.classifier = nn.Linear(
            in_features=d_out[-1],
            out_features=vocab_size
        )

        # create the RZOH layers
        if rzoh_probabilities is None:
            rzoh_probabilities = [0.0] * num_of_ssm_layers

        assert len(rzoh_probabilities) == num_of_ssm_layers, ("rzoh_probabilities dimension must match num_of_ssm_layers")

        self.rzoh_layers = nn.ModuleList([
            RandomZeroOrderHold(prob=prob) if prob > 0.0 else nn.Identity() for prob in rzoh_probabilities # use Identity for security of 0.0 prob division in RZOH
        ])

        ###################################################
        # dictionary where SSM layer outputs will be stored
        self.feature_maps = {}

        ####################################################
        # keeps the hook handles so we can remove them later
        self.feature_hook_handles = []

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        """
        Perform the forward operation.

        Parameters
        ----------
        x: torch.Tensor
            shape (batch, seq_len, d_in)

        Returns
        -------
        logits: torch.Tensor
            shape (batch, seq_len, vocab_size)
        """

        for ssm_layer, rzoh_layer in zip(self.ssm_layers, self.rzoh_layers):
            x = ssm_layer(x)
            x = rzoh_layer(x)

        x = self.lin_dropout1(x)

        logits = self.classifier(x)

        return logits

    def set_step_scale(
        self,
        step_scales,
    ):
        # create the previous_step_scales as a list of [1, first 3 elements of step_scales]
        previous_step_scales = [1, *step_scales[:-1]]

        assert len(step_scales) == len(self.ssm_layers)

        self.step_scales = step_scales
        self.previous_step_scales = previous_step_scales

        for previous_step_scale, step_scale, layer in zip(
            previous_step_scales,
            step_scales,
            self.ssm_layers
        ):
            layer.set_step_scale(
                step_scale,
                previous_step_scale
            )

    def get_output_lengths(self, input_lengths):
        """Returns the output length, of the subsampled output, when it downsampling happens with step_scale during inference"""

        output_lengths = input_lengths.clone()

        for step_scale, previous_step_scale in zip(
            self.step_scales,
            self.previous_step_scales
        ):

            factor = int(step_scale / previous_step_scale)

            if factor > 1:
                output_lengths = (
                    output_lengths + factor - 1
                ) // factor

        return output_lengths

    def get_layer_output_lengths(self, input_lengths):
        """get the output length for each layer's output feature map in the case of subsampling"""

        current_lengths = input_lengths.clone()
        layer_output_lengths = []

        for step_scale, previous_step_scale in zip(
            self.step_scales,
            self.previous_step_scales
        ):
            factor = int(
                step_scale / previous_step_scale
            )

            if factor > 1:
                current_lengths = (
                    current_lengths + factor - 1
                ) // factor

            layer_output_lengths.append(
                current_lengths.clone()
            )

        return layer_output_lengths

    #################################################################
    # the callback function to register on the forward hook operation
    def _get_feature_hook(self, name):

        def hook(module, input, output): # the hook function to call when the the particular layer "module" finishes its forward pass

            self.feature_maps[name] = output.detach() # through detach i can get the actual tensor data of the output

        return hook

    ##################################################################################
    # the function to call on the model object, to register the forward hook callbacks 
    def register_feature_hooks(self):
        """Registers the forward hooks, to be able to observe the output activations"""

        # clear previous feature maps
        self.feature_maps.clear()

        # remove old hooks if they already exist
        self.remove_feature_hooks()

        # register hook function to the ssm_layers
        for i, ssm_layer in enumerate(self.ssm_layers):

            handle = ssm_layer.register_forward_hook( # handle is like a reference/pointer to that hook function
                self._get_feature_hook(f"ssm_layer_{i}")
            )

            self.feature_hook_handles.append(handle)

        # register hook function to the last linear classification layer
        classifier_handle = self.classifier.register_forward_hook(
            self._get_feature_hook(f"classifier")
        )

        self.feature_hook_handles.append(classifier_handle)

    ###############################################################################################
    # remove the feature hooks, not to register more hooks at once to each layer (safety)
    def remove_feature_hooks(self):
        """Removes the feature hooks"""

        for handle in self.feature_hook_handles:
            handle.remove()

        self.feature_hook_handles.clear()