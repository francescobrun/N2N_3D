import numpy as np
import torch
import json
import tifffile
import logging
import psutil
import time
from datetime import datetime, timedelta
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Tuple

# Import custom modules and external dependencies
from _model import (
    create_model,
    setup_logging,
    zscore_normalize,
    NORM_EPS,
    maybe_compile,
    resolve_compile,
    resolve_device,
    resolve_mixed_precision,
    _amp_autocast,
    _cuda_device_arg,
    _safe_torch_load,
)
from monai.inferers import sliding_window_inference  # MONAI's sliding window inference
from tqdm import tqdm  # Progress bar utility

import warnings

warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
)

# Output formats this pipeline can write. Anything else is rejected at config
# parse time rather than silently reinterpreted as a per-slice dump.
SUPPORTED_OUTPUT_SUFFIXES = {".tif", ".tiff"}


def _load_and_preprocess_volume(
    volume_path: str,
    device: torch.device,
    mean_std_norm: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, float, float, np.dtype]:
    """
    Load a 3D volume from disk and apply preprocessing for inference.

    Args:
        volume_path: File path to the volume file
        device: Device the volume must be placed on. This has to be the same
                device the model lives on -- a bare .cuda() would ignore the
                --cuda_device selection and strand the volume on cuda:0.
        mean_std_norm: Optional tuple of (mean, std) for z-score normalization.
                      If None, computes mean and std from the volume.

    Returns:
        Tuple of (preprocessed_tensor, mean, std, source_dtype) where:
        - tensor: Preprocessed tensor of shape (1, 1, depth, height, width) ready for inference
        - mean: Mean value used for normalization
        - std: Standard deviation used for normalization
        - source_dtype: dtype of the file on disk, so the output can be written
          back in the same format the input used

    Raises:
        FileNotFoundError: If the volume file doesn't exist
        RuntimeError: If volume loading or preprocessing fails
    """
    # Verify the volume file exists before attempting to load
    if not Path(volume_path).exists():
        raise FileNotFoundError(f"Volume file not found: {volume_path}")

    try:
        # Load the volume from TIFF file and convert to float32 for precision.
        # The on-disk dtype is kept so save_output can restore it.
        raw = tifffile.imread(volume_path)
        source_dtype = raw.dtype
        volume = raw.astype(np.float32)
        del raw

        # Apply z-score normalization (shared helper, so training and inference
        # cannot drift apart on the epsilon guard)
        if mean_std_norm is not None:
            mean, std = mean_std_norm
            volume = zscore_normalize(volume, mean, std)
            logging.info(
                f"    Applied normalization with stored mean = {mean:.6f}, std = {std:.6f}"
            )
        else:
            # Compute mean and std from the volume itself
            mean = volume.mean()
            std = volume.std()
            volume = zscore_normalize(volume, mean, std)
            logging.info(
                f"    Applied normalization with computed mean = {mean:.6f}, std = {std:.6f}"
            )

        # Convert numpy array to PyTorch tensor
        tensor = torch.from_numpy(volume)

        # Add batch and channel dimensions: (D, H, W) -> (1, 1, D, H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        # Move tensor to the model's device. Explicit index: with
        # --cuda_device auto (the default) the selected GPU is whichever has
        # the most free memory, which is frequently not cuda:0.
        if device.type == "cuda":
            tensor = tensor.to(device)
            logging.info(f"    Moved volume tensor to {device}")

        return tensor, mean, std, source_dtype

    except Exception as e:
        raise RuntimeError(f"Failed to load or preprocess volume: {e}")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """
    Load pre-trained model weights from a checkpoint file.

    This function loads a model checkpoint that was saved during training,
    containing the model state dictionary and optimizer state.

    Args:
        model: The model architecture to load weights into
        checkpoint_path: Path to the checkpoint file (.pth or .pt)

    Returns:
        The model with loaded weights, set to evaluation mode

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails (corrupted file, wrong format, etc.)
    """
    # Verify checkpoint file exists before attempting to load
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        # Load weights onto the model's current device. weights_only=True is
        # safe here (checkpoints contain only state_dict + optimizer tensors)
        # and silences the PyTorch 2.6+ FutureWarning; on older PyTorch
        # (pre-1.13) the keyword doesn't exist, so _safe_torch_load adapts.
        state = _safe_torch_load(
            checkpoint_path,
            map_location=next(model.parameters()).device,
        )
        state_dict = state["state_dict"]

        # Handle DataParallel key mismatch:
        # If checkpoint keys don't have 'module.' prefix but model is wrapped in DataParallel,
        # we need to add the prefix. Conversely, remove it if loading non-DataParallel into DataParallel.
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        model_is_dataparallel = isinstance(model, torch.nn.DataParallel)

        if not has_module_prefix and model_is_dataparallel:
            # Add 'module.' prefix to all keys
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        elif has_module_prefix and not model_is_dataparallel:
            # Remove 'module.' prefix from all keys
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        # Load state dict with the corrected keys
        model.load_state_dict(state_dict)

        # Set model to evaluation mode
        model.eval()

        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def save_output(
    volume: np.ndarray,
    output_dir: Path,
    network_params: dict,
    overlap: float,
    batch_size: int,
    cuda_device: int,
    compression: bool = True,
    multilayer: bool = True,
    filename: str = "output_multilayer.tif",
    input_json_path: str = None,
    checkpoint_file: str = None,
) -> None:
    """
    Save the denoised 3D volume as either a single multi-layer TIFF file or separate 2D TIFF slices with metadata.

    This function takes a 3D numpy array and saves it in one of two formats:
    - Multi-layer TIFF: Single file containing all slices as layers (default)
    - Separate files: Individual TIFF files for each slice with zero-padding
    Both options include processing metadata in the TIFF tags for traceability.

    Args:
        volume: 3D numpy array of shape (depth, height, width) containing the denoised volume
        output_dir: Directory where the TIFF file(s) will be saved
        network_params: Dictionary containing training parameters and model configuration
                       to be embedded in the TIFF metadata
        compression: Whether to use lossless compression (default: True)
        multilayer: Whether to save as single multi-layer TIFF (True) or separate files (False)
                   (default: True)
        filename: Custom filename for the output file (default: "output_multilayer.tif")

    Raises:
        OSError: If unable to create output directory or save files
    """
    try:
        # Create output directory if it doesn't exist (including parent directories)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare metadata for TIFF tags
        # Convert network_params paths to forward slashes
        clean_network_params = {}
        for key, value in network_params.items():
            if isinstance(value, str) and "\\" in value:
                clean_network_params[key] = value.replace("\\", "/")
            else:
                clean_network_params[key] = value

        # Add input JSON file path if provided
        if input_json_path:
            input_json_path = str(Path(input_json_path)).replace("\\", "/")
        else:
            input_json_path = None

        # Add checkpoint file path if provided
        if checkpoint_file:
            checkpoint_file = str(Path(checkpoint_file)).replace("\\", "/")
        else:
            checkpoint_file = None

        # Create flat metadata dictionary (no nested JSON)
        metadata = {
            "Software": "N2N_3D Denoising Pipeline v.1.0",
            # Processing info
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_json": input_json_path,
            "checkpoint_file": checkpoint_file,
            "inference_batch_size": batch_size,
            "inference_cuda_device": cuda_device,
            "overlap": overlap,
        }

        # Add all network_params automatically (forward slash conversion already applied)
        metadata.update(clean_network_params)

        if multilayer:
            # Save as single multi-layer TIFF file
            output_path = output_dir / filename

            logging.info(f"    Saving as multi-layer TIFF: {output_path}")

            if compression:
                # Use deflate compression (built-in, reliable)
                tifffile.imwrite(
                    str(output_path), volume, metadata=metadata, compression="deflate"
                )
                logging.info("    Used deflate lossless compression")
            else:
                tifffile.imwrite(str(output_path), volume, metadata=metadata)

            logging.info(
                f"Successfully saved multi-layer TIFF with {volume.shape[0]} slices"
            )
        else:
            # Save as separate TIFF files
            logging.info("    Saving as separate TIFF files...")

            slice_stem = Path(filename).stem

            for j in tqdm(range(volume.shape[0]), desc="Saving output slices"):
                # Extract the 2D slice (height x width)
                img_np = volume[j, :, :]

                # Create output filename with zero-padding for proper sorting.
                # Derived from the requested filename so the caller's name is
                # honored rather than silently replaced.
                img_path = output_dir / f"{slice_stem}_{j:05d}.tif"

                # Prepare slice-specific metadata
                slice_metadata = metadata.copy()
                slice_metadata["slice_index"] = j

                # Save the slice as a TIFF file with embedded metadata
                if compression:
                    # Use deflate compression (built-in, reliable)
                    tifffile.imwrite(
                        str(img_path),
                        img_np,
                        metadata=slice_metadata,
                        compression="deflate",
                    )
                else:
                    tifffile.imwrite(str(img_path), img_np, metadata=slice_metadata)

            logging.info(
                f"Successfully saved {volume.shape[0]} separate slices to {output_dir}"
            )

    except Exception as e:

        raise OSError(f"Failed to save output: {e}")


def load_training_params(checkpoint_path: str) -> dict:
    """
    Load training hyperparameters from the checkpoint directory.

    During training, hyperparameters like model architecture, patch size, and
    other settings are saved to a params.json file. This function loads those
    parameters to ensure the inference setup matches the training configuration.

    Args:
        checkpoint_path: Path to the model checkpoint file

    Returns:
        Dictionary containing training parameters with keys:
            - nb_blocks: Number of encoder/decoder blocks in the U-Net
            - nb_first_filters: Number of filters in the first convolution layer
            - train_patch_size: Patch size used during training
            - other training-specific parameters

    Raises:
        FileNotFoundError: If params.json file doesn't exist in checkpoint directory
        RuntimeError: If the parameters file cannot be loaded or parsed
    """
    # Construct path to the parameters file (should be in same directory as checkpoint)
    params_path = Path(checkpoint_path).parent / "params.json"

    if not params_path.exists():
        raise FileNotFoundError(f"Training parameters file not found: {params_path}")

    try:
        # Load and parse the JSON parameters file. Logging of the load is
        # done by the caller as part of a consolidated params/norm line.
        with open(params_path, "r") as f:
            params = json.load(f)
        return params
    except Exception as e:
        raise RuntimeError(f"Failed to load training parameters: {e}")


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Find the latest checkpoint file in the specified directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        Path to the latest checkpoint file

    Raises:
        FileNotFoundError: If no checkpoint files are found in the directory
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Find all checkpoint files matching the pattern
    checkpoint_files = list(checkpoint_dir.glob("weights_epoch_*.torch"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    # Sort by epoch number (extracted from filename)
    def get_epoch(f):
        return int(f.stem.split("_")[-1])

    latest_checkpoint = max(checkpoint_files, key=get_epoch)
    return str(latest_checkpoint)


def load_normalization_stats(checkpoint_path: str) -> Optional[Tuple[float, float]]:
    """
    Load normalization statistics from the params.json file in the checkpoint directory.

    If the model was trained with normalization enabled, the mean and standard
    deviation are saved to the params.json file along with other training parameters.
    This function loads those exact statistics to apply the same normalization during inference.

    Args:
        checkpoint_path: Path to the model checkpoint file

    Returns:
        Tuple of (mean, std) if normalization statistics exist in params.json, otherwise None

    Raises:
        RuntimeError: If the params.json file cannot be loaded or parsed
    """
    # Logging of the load (or fallback) is done by the caller as part of a
    # consolidated params/norm line.
    params_path = Path(checkpoint_path).parent / "params.json"

    if not params_path.exists():
        return None

    try:
        with open(params_path, "r") as f:
            params = json.load(f)

        # Check if normalization statistics are present
        if "normalization_mean" in params and "normalization_std" in params:
            return (params["normalization_mean"], params["normalization_std"])
        return None
    except Exception as e:
        raise RuntimeError(f"Failed to load normalization statistics: {e}")


def _run_inference(
    model: torch.nn.Module,
    test_volume: torch.Tensor,
    batch_size: int,
    train_patch_size: Tuple[int, int, int],
    overlap: float,
    use_amp: bool,
    gpu_aggregation: bool,
    padding: bool,
) -> np.ndarray:
    """
    Run inference on a preloaded volume using a trained denoising model.

    Args:
        model: Pre-trained PyTorch model loaded with weights
        test_volume: Preprocessed input tensor of shape (1, 1, D, H, W)
        batch_size: Number of patches processed simultaneously in sliding window inference
        train_patch_size: Patch size used during training for sliding window inference
        overlap: Overlap ratio between patches for sliding window inference

    Returns:
        3D numpy array of shape (depth, height, width) containing the denoised volume

    Raises:
        RuntimeError: If inference fails at any stage (processing or model execution)
    """
    try:
        # Calculate padding (half patch size on each side). When padding is
        # enabled, every original-volume voxel ends up at or beyond the center
        # of some patch, where the network's prediction is most reliable.
        # When disabled (--no_padding), edge voxels are predicted from one-
        # sided patch context, which can be slightly OOD for the network but
        # saves ~1.5-2x in inference time on large volumes.
        if padding:
            pad_d, pad_h, pad_w = [s // 2 for s in train_patch_size]
        else:
            pad_d = pad_h = pad_w = 0

        with torch.no_grad():

            # Log overlap ratio
            logging.info(
                f"    Overlap: {overlap:.3f} ({int(overlap*100)}% overlap between patches)"
            )

            if padding:
                logging.info(
                    f"    Padding: {pad_d}×{pad_h}×{pad_w} per side (replicate mode)"
                )
                test_volume = torch.nn.functional.pad(
                    test_volume,
                    (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d),
                    mode="replicate",
                )
            else:
                logging.info("    Padding: disabled (--no_padding)")

            def _forward(x):
                # Autocast wraps the forward pass; the output is cast back to
                # fp32 so the sliding-window aggregation stays numerically clean.
                with _amp_autocast(use_amp):
                    out = model(x)
                return out.float()

            # The sliding-window aggregation buffer can live on CPU (default,
            # safest for low-VRAM setups) or on GPU (faster — eliminates the
            # per-patch GPU->CPU sync — but adds ~2 * D * H * W * 4 bytes of
            # VRAM for the running sum and weight map).
            agg_device = (
                next(model.parameters()).device
                if gpu_aggregation
                else torch.device("cpu")
            )
            logging.info(f"    Aggregation buffer device: {agg_device}")

            # Common sliding window inference
            pred_volume = sliding_window_inference(
                inputs=test_volume,
                roi_size=train_patch_size,
                sw_batch_size=batch_size,
                predictor=_forward,
                overlap=overlap,
                mode="gaussian",
                padding_mode="replicate",
                sw_device=next(model.parameters()).device,
                device=agg_device,
                progress=True,
            )

            # Crop padding from output
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                pred_volume = (
                    pred_volume[
                        0,
                        0,
                        pad_d : -pad_d if pad_d > 0 else pred_volume.shape[2],
                        pad_h : -pad_h if pad_h > 0 else pred_volume.shape[3],
                        pad_w : -pad_w if pad_w > 0 else pred_volume.shape[4],
                    ]
                    .cpu()
                    .numpy()
                )
            else:
                pred_volume = pred_volume[0, 0].cpu().numpy()

            return pred_volume

    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")


def main(args) -> None:
    """
    Main execution function for the inference script.

    This function orchestrates the entire inference pipeline:
    1. Load configuration from JSON file
    2. Load training parameters and create the model
    3. Load the trained weights from checkpoint
    4. Run inference on the specified dataset
    5. Save the denoised output

    The function includes comprehensive error handling and logging throughout
    the process to provide feedback and aid in debugging.
    """
    # Initialize logging for the entire process
    setup_logging()

    # Capture wall-clock start so the end-of-run summary can report total time.
    start_time = time.time()

    try:
        # Load configuration from JSON file
        with open(args.input_json, "r") as f:
            config = json.load(f)

        # Extract paths from config
        test_volume_path = config["test_volume_file"]
        checkpoint_dir = config["checkpoint_path"]
        output_path = Path(config["output_file"])

        # Reject unsupported output formats up front. Writing is TIFF-only, and
        # silently reinterpreting e.g. "result.nii" as a directory of numbered
        # .tif slices would hand back something the user never asked for.
        if output_path.suffix.lower() not in SUPPORTED_OUTPUT_SUFFIXES:
            raise ValueError(
                f"'output_file' must end in one of "
                f"{sorted(SUPPORTED_OUTPUT_SUFFIXES)} (got {output_path.name!r}). "
                f"This pipeline writes multi-layer TIFF only."
            )

        # Find the latest checkpoint in the directory
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)

        # Log the inference configuration for user reference. fp16 and
        # torch.compile are intentionally omitted from this banner because
        # they are GPU-capability-gated and shown with full context in the
        # consolidated hardware-detection block below.
        logging.info("Starting inference with configuration:")
        logging.info(f"    Input JSON: {args.input_json}")
        logging.info(f"    Test volume: {test_volume_path}")
        logging.info(f"    Checkpoint: {checkpoint_path}")
        logging.info(f"    Output: {output_path}")
        logging.info(f"    Batch size: {args.batch_size}")
        logging.info(f"    CUDA device: {args.cuda_device}")
        logging.info(f"    GPU aggregation: {args.gpu_aggregation}")
        logging.info(f"    Overlap ratio: {args.overlap}")
        logging.info(f"    Edge padding: {args.padding}")
        logging.info(f"    Compression: {args.compression}")

        # Phase separator
        logging.info("")

        # Load training parameters and normalization stats. Both come from the
        # same params.json file; report them in a single combined line.
        network_params = load_training_params(checkpoint_path)
        norm_stats = load_normalization_stats(checkpoint_path)
        params_path = Path(checkpoint_path).parent / "params.json"
        if norm_stats is not None:
            mean, std = norm_stats
            logging.info(
                f"Loaded training params from {params_path} "
                f"(norm: mean={mean:.6f}, std={std:.6f})"
            )
        else:
            logging.info(
                f"Loaded training params from {params_path} "
                f"(norm: not saved, will compute from volume)"
            )

        # Phase separator
        logging.info("")

        # Create the model architecture and load trained weights
        logging.info("Creating and loading model...")

        # Determine CUDA device + resolve every GPU-capability-gated decision
        # (fp16, torch.compile) in one block so adjacent log lines tell the
        # user exactly what's actually enabled on this hardware.
        # ('auto' picks the GPU with the most free memory.)
        device, cuda_device, compute_major, compute_minor = resolve_device(
            args.cuda_device
        )
        args.cuda_device = cuda_device
        use_amp = resolve_mixed_precision(
            args.half, device, compute_major, compute_minor
        )
        use_compile = resolve_compile(
            args.compile, device, compute_major, "first inference call"
        )

        # Create model (with architecture from training parameters).
        # The norm_division_factor is read from the checkpoint's params.json
        # so inference matches the trained architecture. Falls back to 1
        # (instance norm) for legacy checkpoints that predate the field;
        # any checkpoint produced by the current code records it explicitly.
        norm_division_factor = network_params.get("norm_division_factor", 1)
        logging.info(f"    Using norm_division_factor: {norm_division_factor}")

        # num_res_units must match training so the architecture (and thus the
        # state_dict keys) line up. Falls back to 0 (plain conv block per level)
        # for legacy checkpoints that predate the field.
        num_res_units = network_params.get("num_res_units", 0)
        logging.info(f"    Using num_res_units: {num_res_units}")

        # unet_depth / unet_stride must match training so the architecture (and
        # thus the state_dict keys) line up. Fall back to the original
        # (56,112,224,448) stride-2 geometry for legacy checkpoints that predate
        # these fields.
        unet_depth = network_params.get("unet_depth", 4)
        unet_stride = network_params.get("unet_stride", 2)
        logging.info(f"    Using unet_depth: {unet_depth}, unet_stride: {unet_stride}")

        model = create_model(
            device=cuda_device if torch.cuda.is_available() else "cpu",
            norm_division_factor=norm_division_factor,
            num_res_units=num_res_units,
            unet_depth=unet_depth,
            unet_stride=unet_stride,
        )

        # Load the trained weights from checkpoint
        model = load_checkpoint(model, checkpoint_path)

        # Apply the already-resolved torch.compile decision.
        model = maybe_compile(model, use_compile)

        # Phase separator
        logging.info("")

        # Load and preprocess the volume (with normalization stats if available)
        logging.info("Loading and preprocessing volume...")
        test_volume, norm_mean, norm_std, source_dtype = _load_and_preprocess_volume(
            test_volume_path, device, norm_stats
        )

        # Phase separator
        logging.info("")

        # Run inference
        logging.info("Running inference...")
        pred_volume = _run_inference(
            model,
            test_volume,
            args.batch_size,
            tuple(network_params["train_patch_size"]),
            args.overlap,
            use_amp,
            args.gpu_aggregation,
            args.padding,
        )
        output_shape = pred_volume.shape

        # GPU RAM memory monitoring. Reserved (allocator-level) is the headline
        # figure: it is what nvidia-smi reports and what determines whether the
        # run fits on a given card. Allocated (live tensors only) is shown
        # alongside it because the gap indicates allocator fragmentation.
        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_reserved() / 1024**3
            allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
            logging.info(
                f"GPU memory peak: {peak_vram_gb:.2f} GB reserved "
                f"({allocated_gb:.2f} GB allocated)"
            )
            torch.cuda.reset_peak_memory_stats()

        # RAM memory monitoring:
        process = psutil.Process()
        peak_ram_gb = process.memory_info().rss / 1024**3
        logging.info(f"RAM memory peak: {peak_ram_gb:.2f} GB")

        # Denormalize the output volume back to original gray level range.
        # zscore_normalize divides by (std + NORM_EPS), so invert with the same
        # denominator rather than std alone.
        pred_volume = pred_volume * (norm_std + NORM_EPS) + norm_mean
        logging.info(
            f"Denormalized output volume using mean={norm_mean:.6f}, std={norm_std:.6f}"
        )

        # Restore the source dtype so downstream tools receive the format they
        # supplied. Integer sources are clipped to the type's range and rounded;
        # without the clip, values the network pushed slightly past the range
        # would wrap around on cast. --output_float32 keeps the raw float
        # prediction for further numerical processing.
        if args.output_float32:
            logging.info("    Output dtype: float32 (--output_float32)")
        elif np.issubdtype(source_dtype, np.integer):
            info = np.iinfo(source_dtype)
            clipped = np.clip(pred_volume, info.min, info.max)
            n_clipped = int(np.count_nonzero(clipped != pred_volume))
            pred_volume = np.rint(clipped).astype(source_dtype)
            msg = f"    Output dtype: restored to {source_dtype}"
            if n_clipped:
                msg += (
                    f" ({n_clipped} voxel(s) clipped to " f"[{info.min}, {info.max}])"
                )
            logging.info(msg)
        else:
            pred_volume = pred_volume.astype(source_dtype)
            logging.info(f"    Output dtype: restored to {source_dtype}")

        # Phase separator
        logging.info("")

        # Save the denoised volume
        logging.info("Saving output...")
        output_dir = output_path.parent

        save_output(
            pred_volume,
            output_dir,
            network_params,
            args.overlap,
            args.batch_size,
            cuda_device,
            args.compression,
            # Output format is validated to be TIFF at config-parse time.
            True,
            output_path.name,
            args.input_json,
            checkpoint_path,
        )

        # End-of-run summary: total wall-clock, output shape, peak memory,
        # and the output file size on disk. Collapses what used to be three
        # or four separate trailing lines into one scannable record.
        elapsed = str(timedelta(seconds=int(time.time() - start_time)))
        try:
            output_size_bytes = output_path.stat().st_size
            if output_size_bytes >= 1024**3:
                size_str = f"{output_size_bytes / 1024**3:.2f} GB"
            else:
                size_str = f"{output_size_bytes / 1024**2:.1f} MB"
        except OSError:
            size_str = "size unknown"
        shape_str = "×".join(str(s) for s in output_shape)

        # Phase separator
        logging.info("")
        logging.info(
            f"Inference complete: {shape_str} voxels in {elapsed}, "
            f"peak VRAM {peak_vram_gb:.2f} GB, peak RAM {peak_ram_gb:.2f} GB. "
            f"Output: {output_path.name} ({size_str})"
        )

    except Exception as e:
        # Log any errors that occur during the process
        logging.error(f"Inference failed: {e}")
        # Re-raise the exception to ensure non-zero exit code
        raise


if __name__ == "__main__":

    # Parse and validate command line arguments
    parse = ArgumentParser(
        description="Load a model trained with Noise2Inverse, and use it to denoise a volume"
    )

    # Required arguments
    parse.add_argument(
        "input_json",
        help="Path to JSON file containing input/output paths and checkpoint directory",
    )

    # Optional arguments with default values
    parse.add_argument(
        "--batch_size", default=4, type=int, help="The number of patches per batch"
    )
    parse.add_argument(
        "--cuda_device",
        default="auto",
        type=_cuda_device_arg,
        help="CUDA device to use: a non-negative integer or 'auto' (picks the GPU with the most free memory via nvidia-smi). Default: auto.",
    )
    parse.add_argument(
        "--overlap",
        default=0.5,
        type=float,
        help="Overlap ratio between patches for sliding window inference (default: 0.5). Patch count -- and so runtime -- scales roughly as 1/(1-overlap)^3, so 0.85 costs about 8x more than 0.5. Gaussian-weighted blending suppresses patch-boundary seams well at the default, but on data with strong low-frequency structure seams can still appear; raise the overlap if you see a patch grid in the output.",
    )
    parse.add_argument(
        "--no_compression",
        action="store_true",
        help="Disable compression in output TIFF files (default: enabled)",
    )
    parse.add_argument(
        "--output_float32",
        action="store_true",
        help="Write the output as float32 instead of restoring the input volume's dtype (default: restore). Use when the denoised volume feeds further numerical processing and you do not want it quantized back to the source bit depth.",
    )
    parse.add_argument(
        "--no_half",
        action="store_true",
        help="Disable fp16 mixed precision inference (default: enabled when CUDA is available)",
    )
    parse.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (default: disabled). Gives a ~1.2-1.5x speedup after a one-time compilation on the first inference call, but requires PyTorch 2.0+, a GPU with compute capability >= 7.0, and -- on Windows -- an MSVC toolchain + Windows SDK on PATH to build the kernels. Only enable it if your toolchain is set up; otherwise Triton prints repeated 'Failed to find MSVC' warnings and falls back to eager mode.",
    )
    parse.add_argument(
        "--no_padding",
        action="store_true",
        help="Disable replicate-padding of the test volume by half-patch on each side (default: enabled). Padding ensures every output voxel is predicted from well-conditioned patch-center context; disabling it cuts inference time roughly 1.5-2x at the cost of slightly degraded predictions in the outermost ~half-patch of the volume.",
    )
    parse.add_argument(
        "--gpu_aggregation",
        action="store_true",
        help="Keep the sliding-window aggregation buffer on GPU instead of CPU (default: CPU). Faster (eliminates per-patch sync) but costs ~2 * D * H * W * 4 bytes of extra VRAM; safe only on cards with enough headroom for the volume size.",
    )

    args = parse.parse_args()

    # Handle flag logic (default to True, disable if flag is set)
    args.compression = not args.no_compression
    args.half = not args.no_half
    args.padding = not args.no_padding

    main(args)
