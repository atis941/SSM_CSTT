import torch
import tensorflow as tf

class CTCDecoder:

    def __init__(
        self,
        blank_idx: int,
        beam_width: int,
        top_paths: int,
        method: str = "greedy", # base definition for training and evaluation
    ):
        self.method = method
        self.blank_idx = blank_idx
        self.beam_width = beam_width
        self.top_paths = top_paths

        assert self.blank_idx == 0, "TensorFlow beam search currently supports blank_idx=0 only"

        # select the decoding method
        if self.method not in ["greedy", "beam_search"]:
            raise ValueError(
                f"Unknown decoding method: {self.method}"
            )
        
    def __call__(self, 
                 logits: torch.Tensor,
                 valid_output_lengths: torch.Tensor) -> list[list[int]]:
        """
        Automatically calls the selected decoding method.

        Parameters
        ----------
        logits: torch.Tensor [B, T, C]
            the predicted logits -> necessary for beam search
        
        pred_ids: torch.Tensor
            the indices of the most probable phonemes at each timestep -> necessary for ctc_greedy_decode

        Returns
        -------
        Output:
            Decoded phoneme indices.
        """

        if self.method == "greedy":
            return self.perform_ctc_greedy_decode(logits=logits,
                                                  valid_output_lengths=valid_output_lengths)

        elif self.method == "beam_search":
            
            
            # returns:
            # -------
            # decoded: list[tf.SparseTensor], length=top_paths
            #   contains the decoded phoneme sequences for every recording -> [0] is the best 
            # log_probabilities: tf.Tensor - float32
            return self.perform_ctc_beam_search(logits_predicted=logits,
                                                valid_output_lengths=valid_output_lengths)

    def perform_ctc_greedy_decode(self,
                                  logits: torch.Tensor,
                                  valid_output_lengths: torch.Tensor):
        """
        Performs the ctc greedy decoding

        Parameters
        """
        # Argmax does not need gradients.
        with torch.no_grad():
            pred_ids_batch = torch.argmax(logits.detach(),
                                          dim=-1)

            pred_ids_batch = pred_ids_batch.cpu()
            lengths = valid_output_lengths.cpu().tolist()

            predictions = []

            for sample, current_input_length in enumerate(lengths):

                pred_ids = pred_ids_batch[sample, :current_input_length].tolist()

                decoded_pred = self.ctc_greedy_decode(pred_ids,
                                                      blank_idx=self.blank_idx)

                predictions.append(decoded_pred)

            return predictions

    def perform_ctc_beam_search(self, 
                                logits_predicted: torch.Tensor,
                                valid_output_lengths: torch.Tensor):
        """
        Performs the beam search decoding for the predicted phoneme indices -> ctc_beam_search comes from tensorflow
        
        Parameters
        ----------
        logits_predicted: torch.Tensor
            the logits predicted by the SSM model

        Returns
        -------
        list[list[int]]
            Decoded phoneme indices for each recording in the batch.
        """
        # Move blank to the last position -> tensorflow's beam search needs it
        reordered_logits = torch.cat([logits_predicted[..., 1:], logits_predicted[..., :1]],
                                     dim=-1)

        # Convert to TensorFlow format [T, B, C] -> tensorflow's beam search needs it
        logits_tf = tf.convert_to_tensor(reordered_logits.permute(1, 0, 2).detach().cpu().numpy(),
                                         dtype=tf.float32)

        # Convert valid output lengths from pytorch tensor to tensorflow tensor
        valid_output_lengths_tf = tf.convert_to_tensor(valid_output_lengths.detach().cpu().numpy(),
                                                       dtype=tf.int32)

        # Perform CTC beam search
        # decoded: decoded phoneme sequences
        decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
            inputs=logits_tf,
            sequence_length=valid_output_lengths_tf,
            beam_width=self.beam_width,
            top_paths=self.top_paths
        )

        # Convert sparse output to dense
        # Each sample have a different number of predicted phonemes -> each decoded phoneme can have different lengths
        # Therefore we need to convert it into a rectangular tensor, with padding. in this case the
        # padding value is -1
        # decoded[0] contains the best decoded hypothesis sequence for each recording in the batch
        # contains predicted phoneme indices
        decoded_dense = tf.sparse.to_dense(
            decoded[0],
            default_value=-1
        ).numpy()

        # Restore original phoneme indices
        # 1. remove the -1 padding values
        # 2. Convert the tensorflow class indices back to the original vocabulary indices -> because of moved blank idx for tensorflow
        prediction_indices = [
            [int(idx) + 1 for idx in row if idx != -1]
            for row in decoded_dense
        ]

        return prediction_indices

    def ctc_greedy_decode(self,
                          pred_ids:list, 
                          blank_idx=0):
        """
        Collapse repeats and remove blank symbols from a single predicted sequence.
        pred_ids: list[int]
        """
        decoded = []
        prev = None

        for idx in pred_ids:
            if idx != blank_idx and idx != prev:
                decoded.append(idx)
            prev = idx

        return decoded