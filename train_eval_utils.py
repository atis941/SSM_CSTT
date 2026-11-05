import time
from typing import Any

import torch

from utils import compute_batch_per_from_log_probs, decode_batch_to_phonemes, log_message
import pandas as pd
import os

from Decoder import CTCDecoder

def train_one_batch(X_train, 
                    X_train_length, 
                    y_train, 
                    y_train_length, 
                    device, 
                    ssm_model, 
                    loss_fn, 
                    true_train_batch_loss):
    """
    Trains one batch and calculates loss
    backward propagation excluded
    """
    # send tensors to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # forward pass
    logits = ssm_model(X_train)                    # [B, T, C]
    log_probs = logits.log_softmax(dim=-1)        # [B, T, C]

    #print("logits finite:", torch.isfinite(logits).all().item())
    #print("NaNs in logits:", torch.isnan(logits).sum().item())
    #print("+Inf in logits:", torch.isposinf(logits).sum().item())
    #print("-Inf in logits:", torch.isneginf(logits).sum().item())

    #print("log_probs finite:", torch.isfinite(log_probs).all().item())
    #print("NaNs in log_probs:", torch.isnan(log_probs).sum().item())
    #print("+Inf in log_probs:", torch.isposinf(log_probs).sum().item())
    #print("-Inf in log_probs:", torch.isneginf(log_probs).sum().item())
    log_probs_ctc = log_probs.transpose(0, 1)     # [T, B, C]

    #checking exploding gradients for the log probabilities
    check_finite_tensor(log_probs, "log_probs")

    # loss
    loss = loss_fn(log_probs_ctc, y_train, X_train_length, y_train_length)

    # checking exploding gradients for the loss
    check_finite_tensor(loss, "loss")

    # accumulate weighted epoch loss
    true_train_batch_loss += loss.item() * len(X_train)

    return log_probs, log_probs_ctc, true_train_batch_loss, loss

def evaluate_one_batch(X_val: any,
                       X_val_length: any,
                       y_val: any,
                       y_val_length: any,
                       device: any,
                       ssm_model: any,
                       loss_fn: any,
                       true_val_batch_loss: any,
                       test_subsampling: bool = False):
    """Evaluates one batch during training"""

    # send tensors to device
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # forward pass
    logits = ssm_model(X_val)                  # [B, T, C] -> raw model scores, NOT PROBABILITIES
    log_probs = logits.log_softmax(dim=-1)    # [B, T, C]  ->  turn the raw model scores into probabilities
    log_probs_ctc = log_probs.transpose(0, 1) # [T, B, C] -> ctc loss function needs to have the input in this shape

    # in case of subsampling the temporal dimension of the log_probs change by the step_scale factors
    # for ctc we need to define the subsampled temporal length(if subsampling was applied)
    if test_subsampling:
        valid_output_lengths = ssm_model.get_output_lengths(X_val_length)
    else:
        valid_output_lengths = X_val_length

    # loss
    loss = loss_fn(log_probs_ctc, y_val, valid_output_lengths, y_val_length)

    # accumulate weighted epoch loss
    true_val_batch_loss += loss.item() * len(X_val)

    return log_probs, log_probs_ctc, valid_output_lengths, true_val_batch_loss, loss

def calculate_and_update_epoch_loss(true_batch_loss: any,
                                    dataset: any,
                                    epoch_losses_list: any):
    epoch_loss = true_batch_loss / len(dataset)
    epoch_losses_list.append(epoch_loss)

    return epoch_loss


def stop_early(
    use_early_stopping,
    best_epoch,
    best_val_loss,
    epoch_val_loss,
    early_stopping_min_delta,
    epoch,
    epochs_without_improvement,
    early_stopping_patience,
    ssm_model,
    best_model_path,
    log_file_loss_and_PER
):
    should_stop_early = False

    if use_early_stopping:
        improvement = best_val_loss - epoch_val_loss

        if improvement > early_stopping_min_delta:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                ssm_model.state_dict(),
                best_model_path
            )

            log_message(
                f"Validation loss improved to {best_val_loss:.6f}. "
                f"Best model saved at epoch {epoch + 1}.",
                log_file_loss_and_PER
            )

        else:
            epochs_without_improvement += 1

            log_message(
                f"No sufficient validation improvement for "
                f"{epochs_without_improvement}/{early_stopping_patience} epochs.",
                log_file_loss_and_PER
            )

            if epochs_without_improvement >= early_stopping_patience:
                should_stop_early = True

    return (
        best_val_loss,
        best_epoch,
        epochs_without_improvement,
        should_stop_early
    )

def log_epoch_info(compute_per: bool,
                   epoch: int,
                   epoch_per_calc_division: int,
                   log_file_loss_and_PER,
                   epoch_train_loss: any,
                   epoch_train_per: any,
                   epoch_val_loss: any,
                   epoch_val_per):
    """log the training and validation results for the current epoch"""
    log_message("-------------------------------------------------------------------------------------------------------", log_file_loss_and_PER)
    
    if compute_per and (epoch == 0 or (epoch + 1) % epoch_per_calc_division == 0):
        log_message(
            f"Epoch: {epoch + 1} | Training loss: {epoch_train_loss:.6f} | Training PER: {epoch_train_per * 100:.2f}%",
            log_file_loss_and_PER
        )
        log_message(
            f"Epoch: {epoch + 1} | Validation loss: {epoch_val_loss:.6f} | Validation PER: {epoch_val_per * 100:.2f}%",
            log_file_loss_and_PER
        )
    else:
        log_message(
            f"Epoch: {epoch + 1} | Training loss: {epoch_train_loss:.6f}",
            log_file_loss_and_PER
        )
        log_message(
            f"Epoch: {epoch + 1} | Validation loss: {epoch_val_loss:.6f}",
            log_file_loss_and_PER
        )

    log_message("-------------------------------------------------------------------------------------------------------", log_file_loss_and_PER)

def log_early_stopping(epoch: int,
                       best_epoch: int,
                       best_val_loss: any,
                       log_file_loss_and_PER: any,
                       log_file_predictions: any):
    log_message(
        f"Early stopping triggered after epoch {epoch + 1}. "
        f"Best epoch was {best_epoch + 1} "
        f"with validation loss {best_val_loss:.6f}.",
        log_file_loss_and_PER
    )

    log_message(
        f"Early stopping triggered after epoch {epoch + 1}. "
        f"Best epoch was {best_epoch + 1}.",
        log_file_predictions
    )

def restore_best_model(best_model_path: any,
                       device:any,
                       ssm_model: any,
                       best_epoch:any,
                       best_val_loss:any):
    best_state_dict = torch.load(
        best_model_path,
        map_location=device
    )
    
    ssm_model.load_state_dict(best_state_dict)

    print(
        f"Best model restored from epoch {best_epoch + 1} "
        f"with validation loss {best_val_loss:.6f}"
    )

def create_checkpoint(epoch: int,
                      use_scheduler: bool,
                      train_epoch_losses: list,
                      val_epoch_losses: list,
                      train_epoch_PERs: list,
                      val_epoch_PERs: list,
                      ssm_model: any,
                      optimizer: any,
                      scheduler: any,
                      best_val_loss:any,
                      best_epoch:any,
                      epochs_without_improvement:any) -> None:
    """Creates a Checkpoint to save the current status and informations about the trained model
    
    Parameters
    ----------
    epoch: int
        The current epoch
    
    use_scheduler: bool
        To decide whether scheduler was used or not
    
    train_epoch_losses: list
        The list of the training epoch losses until this point
    
    val_epoch_losses: list
        The list of the validation epoch losses until this point

    Returns
    -------
    None
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": ssm_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if use_scheduler else None,
        "train_epoch_losses": train_epoch_losses,
        "val_epoch_losses": val_epoch_losses,
        "train_epoch_PERs": train_epoch_PERs,
        "val_epoch_PERs": val_epoch_PERs,
        # Early stopping state
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement
    }

    torch.save(checkpoint, f"checkpoint.pt")

def log_training_time(training_start_time: any,
                      log_file_loss_and_PER: any,
                      log_file_predictions: any):
    training_end_time = time.time()
    elapsed_seconds = training_end_time - training_start_time
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)

    if hours > 0:
        elapsed_time_str = f"{hours} hours {minutes} minutes"
    else:
        elapsed_time_str = f"{minutes} minutes"

    log_message(
        f"Total training and validation time is: {elapsed_time_str}",
        log_file_loss_and_PER
    )

    log_message(
        f"Total training and validation time is: {elapsed_time_str}",
        log_file_predictions
    )

def qualitative_predictions(epoch: int,
                            predict_every_epoch: int,
                            ssm_model:any,
                            log_file_predictions: any,
                            val_dataloader: any,
                            device: any,
                            idx_to_phoneme_vocab: any,
                            blank_idx: any):
    if (epoch + 1) % predict_every_epoch == 0:
        log_message(f"########## QUALITATIVE VALIDATION CHECK AFTER EPOCH {epoch + 1} ##########", log_file_predictions)

        ssm_model.eval()
        with torch.inference_mode():
            # take one batch from validation loader
            X_obs, X_obs_length, y_obs, y_obs_length = next(iter(val_dataloader))

            X_obs = X_obs.to(device)
            y_obs = y_obs.to(device)

            logits_obs = ssm_model(X_obs)
            log_probs_obs = logits_obs.log_softmax(dim=-1)

            decoded_examples = decode_batch_to_phonemes(
                log_probs=log_probs_obs,
                input_lengths=X_obs_length,
                targets=y_obs,
                target_lengths=y_obs_length,
                idx_to_phoneme=idx_to_phoneme_vocab,
                blank_idx=blank_idx
            )

            # show only a few examples
            for i in range(min(5, len(decoded_examples))):
                pred_phonemes, tgt_phonemes = decoded_examples[i]

                log_message(f"Example {i + 1}", log_file_predictions)
                log_message(f"PRED: {' '.join(pred_phonemes)}", log_file_predictions)
                log_message(f"TGT : {' '.join(tgt_phonemes)}", log_file_predictions)
                log_message("--------------------------------------------------", log_file_predictions)

def apply_scheduler(scheduler_type: str,
                    scheduler: any,
                    epoch_val_loss: any,
                    optimizer: any):
    """activate scheduler if necessary"""

    if scheduler_type == "StepLR":
        scheduler.step()
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler.step(epoch_val_loss)
    print("LR:", optimizer.param_groups[0]["lr"])

def calculate_PER_for_batch(log_probs: any,
                            X_length: any,
                            y: any,
                            y_length: any,
                            total_edit_distance: any,
                            total_target_length: any,
                            decoder: CTCDecoder) -> list[float, int, int]:
    # calculates the PER value for one batch
    # log_probs: torch.Tensor - floating point, [B, T_subsampled, vocabulary_size], predicted log probability of every phoneme class at every output time step for every recording in the batch
    # X_lengths: torch.tensor - integer, [B], the unpadded valid timesteps of each recordings AFTER THE SUBSAMPLING (if it was applied)
    # y: torch.Tensor - integer, [N], Ground truth PHONEME INDICES concatenated across the batch,
    # y_length: torch.Tensor - integer, [B], number of GROUND-TRUTH phonemes in each recording, -> y_val_length[0] is the number of target phonemes of the first recording
    # blank_idx: integer, 0, the index of the blank character in the vocabulary
    batch_per, batch_edits, batch_targets = compute_batch_per_from_log_probs(
        log_probs=log_probs,
        output_lengths=X_length,
        targets=y,
        target_lengths=y_length,
        decoder=decoder
    )
    # batch_per: float, the calculated PER value for the batch
    # batch_edits: int, the calculated edit distance for the batch
    # batch_targets: int, the total target length of the batch

    # update the edit_distance and target_length for the WHOLE DATASET
    total_edit_distance += batch_edits
    total_target_length += batch_targets

    return batch_per, total_edit_distance, total_target_length

def log_batch_loss_and_PER(batch: int,
                           compute_per: bool,
                           epoch: int,
                           epoch_per_calc_division: int,
                           dataloader_obj: any,
                           loss: any,
                           batch_per: any,
                           log_file_loss_and_PER: any,
                           phase: str):
    if compute_per and (epoch == 0 or (epoch + 1) % epoch_per_calc_division == 0):
        log_message(
            f"{phase} batch: {batch + 1}/{len(dataloader_obj)} | "
            f"Batch loss: {loss.item():.4f} | "
            f"Batch PER: {batch_per * 100:.2f}%",
            log_file_loss_and_PER
        )
    else:
        log_message(
            f"{phase} batch: {batch + 1}/{len(dataloader_obj)} | "
            f"Batch loss: {loss.item():.4f}",
            log_file_loss_and_PER
        )


def check_gradients_for_inf(ssm_model: any):
    """Checks the gradients for infinite values before clipping"""

    for name, param in ssm_model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():

                num_nan = torch.isnan(param.grad).sum().item()
                num_inf = torch.isinf(param.grad).sum().item()

                raise RuntimeError(
                    f"NON-FINITE GRADIENT BEFORE CLIPPING: "
                    f"{name} | NaNs={num_nan} | Infs={num_inf}"
                )

def check_finite_tensor(tensor, name: str):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"Non-finite values detected in {name}. "
            f"min={tensor.nan_to_num().min().item():.4f}, "
            f"max={tensor.nan_to_num().max().item():.4f}"
        )

def save_test_results_to_csv(
    test_results: list[dict[str, Any]],
    test_save_path: str
) -> None:
    """
    Saves the testing results of all evaluated step-scale configurations
    to a CSV file.

    Each dictionary in test_results represents one evaluated configuration
    and becomes one row in the CSV file.

    Parameters:
    -----------
    test_results (list[dict[str, Any]]):
        List containing the testing results of all step-scale
        configurations. Each dictionary contains the configuration,
        CTC loss, PER, and optionally computational metrics.

    test_save_path (str):
        Directory where the CSV file should be saved.

    Returns:
    --------
        None
    """

    # Create the directory if necessary
    os.makedirs(test_save_path, exist_ok=True)

    # Convert the results list into a DataFrame
    df_test = pd.DataFrame(test_results)

    # Define output file path
    csv_path = os.path.join(
        test_save_path,
        f"test_results.csv",
    )

    # Save results
    df_test.to_csv(
        csv_path,
        index=False
    )

def calculate_test_dataset_metrics(true_val_batch_loss: any,
                                   val_dataloader: any,
                                   total_edit_distance: any,
                                   total_target_length: any):
    
    dataset_loss = (
        true_val_batch_loss / len(val_dataloader.dataset)
    )

    dataset_per = (
        total_edit_distance / total_target_length
        if total_target_length > 0
        else 0.0
    )

    return dataset_loss, dataset_per

def append_test_results(test_results: list,
                        step_scales: any,
                        dataset_loss: any,
                        dataset_per: any,
                        total_edit_distance: any,
                        total_target_length: any,
                        invalid_ctc_samples: any,
                        macs: float,
                        macs_percent: float,
                        params: any,
                        trainable_params: any,
                        decoder: str):


    test_results.append({
        "step_scale": step_scales.copy(),
        "test_loss": dataset_loss,
        "test_PER": dataset_per,
        "total_edit_distance": total_edit_distance,
        "total_target_length": total_target_length,
        "invalid_ctc_samples": invalid_ctc_samples,
        "macs": macs,
        "macs_percent": macs_percent,
        "params": params,
        "trainable_params": trainable_params,
        "decoder": decoder
    })

    print(
        f"Configuration {step_scales} | "
        f"Loss: {dataset_loss:.4f} | "
        f"PER: {dataset_per * 100:.2f}%"
    )

def count_invalid_ctc_samples(
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    output_lengths: torch.Tensor
) -> int:

    invalid_samples = 0
    offset = 0

    for target_length, output_length in zip(
        target_lengths.tolist(),
        output_lengths.tolist()
    ):
        target_length = int(target_length)
        output_length = int(output_length)

        target = targets[offset:offset + target_length]

        repeated_pairs = (
            target[1:] == target[:-1]
        ).sum().item()

        minimum_output_length = (
            target_length + repeated_pairs
        )

        if output_length < minimum_output_length:
            invalid_samples += 1

        offset += target_length

    return invalid_samples

def log_test_phoneme_predictions(
    log_probs: torch.Tensor,
    valid_output_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    decoder,
    idx_to_phoneme_vocab: dict,
    log_file,
    batch_idx: int,
    recording_offset: int
) -> int:
    """
    Decode and log the predicted and target phonemes
    for every recording in the current test batch.

    Parameters
    ----------
    log_probs: torch.Tensor [B, T, C]
        Predicted log probabilities.

    valid_output_lengths: torch.Tensor [B]
        Valid output lengths after subsampling.

    targets: torch.Tensor [N]
        Concatenated ground-truth phoneme indices.

    target_lengths: torch.Tensor [B]
        Number of ground-truth phonemes per recording.

    decoder: CTCDecoder
        Decoder used for prediction.

    idx_to_phoneme_vocab: dict
        Mapping from phoneme indices to phoneme strings.

    log_file:
        Open text file for writing predictions.

    batch_idx: int
        Current batch index, starting from zero.

    recording_offset: int
        Number of recordings processed in previous batches.

    Returns
    -------
    int
        Updated recording offset.
    """

    # Decode the predictions for the entire batch.
    # Output: list[list[int]]
    predicted_ids_batch = decoder(
        logits=log_probs,
        valid_output_lengths=valid_output_lengths
    )

    # Convert target lengths and indices to Python lists.
    target_lengths_list = target_lengths.cpu().tolist()
    targets_list = targets.cpu().tolist()

    # Position in the concatenated target tensor.
    target_offset = 0

    for sample_idx, predicted_ids in enumerate(predicted_ids_batch):

        # Retrieve the ground-truth phoneme indices.
        current_target_length = target_lengths_list[sample_idx]

        target_ids = targets_list[
            target_offset:target_offset + current_target_length
        ]

        target_offset += current_target_length

        # Convert predicted indices into phoneme strings.
        predicted_phonemes = [
            idx_to_phoneme_vocab[idx]
            for idx in predicted_ids
        ]

        # Convert target indices into phoneme strings.
        target_phonemes = [
            idx_to_phoneme_vocab[idx]
            for idx in target_ids
        ]

        # Global recording number.
        recording_number = recording_offset + sample_idx + 1

        # Log the prediction.
        log_file.write(
            f"Batch {batch_idx + 1} - Recording {recording_number}\n"
        )

        log_file.write(
            f"PRED: {' '.join(predicted_phonemes)}\n"
        )

        log_file.write(
            f"TGT : {' '.join(target_phonemes)}\n"
        )

        log_file.write(
            "--------------------------------------------------\n"
        )

    # Return the total number of processed recordings.
    return recording_offset + len(predicted_ids_batch)

def initialize_test_predictions_log(
    path_to_test_save: str,
    step_scales: list[int],
    decoding_method: str
):
    """
    Initialize the phoneme prediction log file for a test configuration.

    Returns
    -------
    log_file_predictions:
        Open file object for writing predictions.

    log_file_path: str
        Path to the created log file.
    """

    # Create the directory for testing results
    print("path_to_test_save:", path_to_test_save)
    print("step_scales:", step_scales)
    os.makedirs(path_to_test_save, exist_ok=True)

    # Create a unique filename for the current step-scale configuration
    step_scales_string = "_".join(str(scale) for scale in step_scales)

    log_file_path = os.path.join(
        path_to_test_save,
        f"aligned_phn_preds_{step_scales_string}.txt"
    )

    # Open the log file
    log_file_predictions = open(
        log_file_path,
        "w",
        encoding="utf-8"
    )

    # Write header information
    log_file_predictions.write(
        "########## TEST PHONEME PREDICTIONS ##########\n"
    )

    log_file_predictions.write(
        f"Step scales: {step_scales}\n"
    )

    log_file_predictions.write(
        f"Decoder: {decoding_method}\n"
    )

    log_file_predictions.write(
        "==================================================\n\n"
    )

    return log_file_predictions, log_file_path, step_scales_string