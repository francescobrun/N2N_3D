import numpy as np
import torch
import json
import tifffile
import logging
import psutil
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Tuple

# Import custom modules and external dependencies
from _model import create_model  # Model creation from model.py
from monai.inferers import sliding_window_inference  # MONAI's sliding window inference
from tqdm import tqdm  # Progress bar utility

import warnings
warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated"
)

def setup_logging() -> None:
    """
    Configure logging settings for the inference script.
    
    This function sets up the logging module to provide informative output
    during the inference process, including timestamps.
    """
    logging.basicConfig(
        level=logging.INFO,  # Show INFO level and above messages
        format='%(asctime)s - %(message)s',  # Include timestamp but remove log level
        datefmt='%Y-%m-%d %H:%M:%S'  # Timestamp format
    )

def _load_and_preprocess_volume(volume_path: str, mean_std_norm: Optional[Tuple[float, float]] = None) -> Tuple[torch.Tensor, float, float]:
    """
    Load a 3D volume from disk and apply preprocessing for inference.
    
    Args:
        volume_path: File path to the volume file
        mean_std_norm: Optional tuple of (mean, std) for z-score normalization.
                      If None, computes mean and std from the volume.
    
    Returns:
        Tuple of (preprocessed_tensor, mean, std) where:
        - tensor: Preprocessed tensor of shape (1, 1, depth, height, width) ready for inference
        - mean: Mean value used for normalization
        - std: Standard deviation used for normalization
    
    Raises:
        FileNotFoundError: If the volume file doesn't exist
        RuntimeError: If volume loading or preprocessing fails
    """
    # Verify the volume file exists before attempting to load
    if not Path(volume_path).exists():
        raise FileNotFoundError(f"Volume file not found: {volume_path}")
    
    try:
        # Load the volume from TIFF file and convert to float32 for precision
        volume = tifffile.imread(volume_path).astype(np.float32)
        
        # Apply z-score normalization 
        if mean_std_norm is not None:
            mean, std = mean_std_norm
            volume = (volume - mean) / std
            logging.info(f"    Applied normalization with stored mean = {mean:.6f}, std = {std:.6f}")
        else: 
            # Compute mean and std from the volume itself
            mean = volume.mean()
            std = volume.std()
            volume = (volume - mean) / std
            logging.info(f"    Applied normalization with computed mean = {mean:.6f}, std = {std:.6f}")
        
        # Convert numpy array to PyTorch tensor
        tensor = torch.from_numpy(volume)
        
        # Add batch and channel dimensions: (D, H, W) -> (1, 1, D, H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        # Move tensor to GPU if available for faster inference
        if torch.cuda.is_available():
            tensor = tensor.cuda()
            logging.info("    Moved volume tensor to GPU")
        
        logging.info("Loaded and preprocessed volume")
        return tensor, mean, std

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
        # Load weights
        state = torch.load(checkpoint_path)
        state_dict = state["state_dict"]
        
        # Handle DataParallel key mismatch:
        # If checkpoint keys don't have 'module.' prefix but model is wrapped in DataParallel,
        # we need to add the prefix. Conversely, remove it if loading non-DataParallel into DataParallel.
        has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
        model_is_dataparallel = isinstance(model, torch.nn.DataParallel)
        
        if not has_module_prefix and model_is_dataparallel:
            # Add 'module.' prefix to all keys
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif has_module_prefix and not model_is_dataparallel:
            # Remove 'module.' prefix from all keys
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        
        # Load state dict with the corrected keys
        model.load_state_dict(state_dict)
        
        # Set model to evaluation mode
        model.eval()
        
        logging.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

def save_output(volume: np.ndarray, output_dir: Path, network_params: dict, 
                use_tta: bool, overlap: float, batch_size: int, cuda_device: int,
                compression: bool = True, multilayer: bool = True, filename: str = "output_multilayer.tif",
                input_json_path: str = None, checkpoint_file: str = None) -> None:
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
            if isinstance(value, str) and '\\' in value:
                clean_network_params[key] = value.replace('\\', '/')
            else:
                clean_network_params[key] = value
        
        # Add input JSON file path if provided
        if input_json_path:
            input_json_path = str(Path(input_json_path)).replace('\\', '/')
        else:
            input_json_path = None
            
        # Add checkpoint file path if provided
        if checkpoint_file:
            checkpoint_file = str(Path(checkpoint_file)).replace('\\', '/')
        else:
            checkpoint_file = None
        
        # Create flat metadata dictionary (no nested JSON)
        metadata = {
            'Software': 'N2N_3D Denoising Pipeline v.1.0',
            # Processing info
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_json': input_json_path,
            'checkpoint_file': checkpoint_file,
            'inference_batch_size': batch_size,
            'inference_cuda_device': cuda_device,
            'use_tta': use_tta,
            'overlap': overlap
        }
        
        # Add all network_params automatically (forward slash conversion already applied)
        metadata.update(clean_network_params)
        
        if multilayer:
            # Save as single multi-layer TIFF file
            output_path = output_dir / filename
            
            logging.info(f"    Saving as multi-layer TIFF: {output_path}")
            
            if compression:
                # Use deflate compression (built-in, reliable)
                tifffile.imwrite(str(output_path), volume, metadata=metadata, compression='deflate')
                logging.info("    Used deflate lossless compression")
            else:
                tifffile.imwrite(str(output_path), volume, metadata=metadata)
            
            logging.info(f"Successfully saved multi-layer TIFF with {volume.shape[0]} slices")
        else:
            # Save as separate TIFF files
            logging.info("    Saving as separate TIFF files...")
            
            for j in tqdm(range(volume.shape[0]), desc="Saving output slices"):
                # Extract the 2D slice (height x width)
                img_np = volume[j, :, :] 
                
                # Create output filename with zero-padding for proper sorting
                img_path = output_dir / f"output_{j:05d}.tif"
                
                # Prepare slice-specific metadata
                slice_metadata = metadata.copy()
                slice_metadata['slice_index'] = j
                
                # Save the slice as a TIFF file with embedded metadata
                if compression:
                    # Use deflate compression (built-in, reliable)
                    tifffile.imwrite(str(img_path), img_np, metadata=slice_metadata, compression='deflate')
                else:
                    tifffile.imwrite(str(img_path), img_np, metadata=slice_metadata)
            
            logging.info(f"Successfully saved {volume.shape[0]} separate slices to {output_dir}")

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
        # Load and parse the JSON parameters file
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        logging.info(f"Loaded training parameters from {params_path}")
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
        return int(f.stem.split('_')[-1])
    
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
    params_path = Path(checkpoint_path).parent / "params.json"
    
    if not params_path.exists():
        logging.info("No params.json file found, will compute from volume")
        return None
    
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        # Check if normalization statistics are present
        if 'normalization_mean' in params and 'normalization_std' in params:
            mean = params['normalization_mean']
            std = params['normalization_std']
            logging.info(f"Loaded normalization statistics from {params_path}: mean={mean:.6f}, std={std:.6f}")
            return (mean, std)
        else:
            logging.info("No normalization statistics found in params.json, will compute from volume")
            return None
    except Exception as e:
        raise RuntimeError(f"Failed to load normalization statistics: {e}")


def _run_inference(model: torch.nn.Module, test_volume: torch.Tensor,
                   batch_size: int, train_patch_size: Tuple[int, int, int], 
                   use_tta: bool, overlap: float) -> np.ndarray:
    """
    Run inference on a preloaded volume using a trained denoising model.
    
    Args:
        model: Pre-trained PyTorch model loaded with weights
        test_volume: Preprocessed input tensor of shape (1, 1, D, H, W)
        batch_size: Number of patches processed simultaneously in sliding window inference
        train_patch_size: Patch size used during training for sliding window inference
        use_tta: Whether to use Test-Time Augmentation for improved predictions
        overlap: Overlap ratio between patches for sliding window inference
    
    Returns:
        3D numpy array of shape (depth, height, width) containing the denoised volume
    
    Raises:
        RuntimeError: If inference fails at any stage (processing or model execution)
    """
    try:
        device = next(model.parameters()).device
        
        # Calculate padding (half patch size on each side)
        pad_d, pad_h, pad_w = [s // 2 for s in train_patch_size]
        
        with torch.no_grad():
            
            # Log overlap ratio
            logging.info(f"    Overlap: {overlap:.3f} ({int(overlap*100)}% overlap between patches)")
            
            # Pad input volume to avoid edge artifacts
            logging.info(f"    Padding: {pad_d}×{pad_h}×{pad_w} per side (reflect mode)")
            test_volume = torch.nn.functional.pad(
                test_volume,
                (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d),
                mode='reflect'
            )
            
            if use_tta:
                logging.info("    Using Test-Time Augmentation...")
                # Create TTA predictor function
                def tta_predictor(x):
                    # TTA transforms
                    tta_transforms = [
                        ('none', None, None),           # 0. Identity
                        ('rot180', [3, 4], None),       # 1. 180° rotation H-W plane
                        ('flip', [2], None),            # 2. Flip D (depth)
                        ('flip', [3], None),            # 3. Flip H (height)
                        ('flip', [4], None),            # 4. Flip W (width)
                        ('rot180', [2, 3], None),       # 5. 180° rotation D-H plane
                        ('rot180', [2, 4], None),       # 6. 180° rotation D-W plane
                        ('flip', [3, 4], None),         # 7. Flip both H and W
                    ]
                    
                    def apply_transform(x, transform_type, dims, k):
                        if transform_type == 'none':
                            return x
                        elif transform_type == 'rot180':
                            return torch.rot90(x, k=2, dims=dims)
                        elif transform_type == 'flip':
                            return torch.flip(x, dims=dims)
                        return x
                    
                    def invert_transform(x, transform_type, dims, k):
                        if transform_type == 'none':
                            return x
                        elif transform_type == 'rot180':
                            return torch.rot90(x, k=2, dims=dims)
                        elif transform_type == 'flip':
                            return torch.flip(x, dims=dims)
                        return x
                    
                    preds = []
                    with torch.no_grad():
                        for transform_type, dims, k in tta_transforms:
                            x_aug = apply_transform(x, transform_type, dims, k)
                            pred = model(x_aug)
                            pred = invert_transform(pred, transform_type, dims, k)
                            preds.append(pred)
                    
                    return torch.stack(preds, dim=0).mean(dim=0)
                
                predictor = tta_predictor
            else:
                logging.info("    Test-Time Augmentation not applied...")
                predictor = model
            
            # Common sliding window inference
            pred_volume = sliding_window_inference(
                inputs=test_volume,
                roi_size=train_patch_size,
                sw_batch_size=batch_size,
                predictor=predictor,
                overlap=overlap,
                mode="gaussian",
                padding_mode="reflect",
                sw_device=next(model.parameters()).device, 
                device=torch.device("cpu"),
                progress=True,
            )
            
            if use_tta:
                logging.info(f"TTA completed with 8 transforms (all flips and 180° rotations)")
            
            # Crop padding from output
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                pred_volume = pred_volume[
                    0, 0,
                    pad_d:-pad_d if pad_d > 0 else pred_volume.shape[2],
                    pad_h:-pad_h if pad_h > 0 else pred_volume.shape[3],
                    pad_w:-pad_w if pad_w > 0 else pred_volume.shape[4]
                ].cpu().numpy()
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
    
    try:
        # Load configuration from JSON file
        with open(args.input_json, 'r') as f:
            config = json.load(f)
        
        # Extract paths from config
        test_volume_path = config['test_volume_file']
        checkpoint_dir = config['checkpoint_path']
        output_path = config['output_file']
        
        # Find the latest checkpoint in the directory
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        
        # Log the inference configuration for user reference
        logging.info("Starting inference with configuration:")
        logging.info(f"    Input JSON: {args.input_json}")
        logging.info(f"    Test volume: {test_volume_path}")
        logging.info(f"    Checkpoint: {checkpoint_path}")
        logging.info(f"    Output: {output_path}")
        logging.info(f"    Batch size: {args.batch_size}")
        logging.info(f"    CUDA device: {args.cuda_device}")
        logging.info(f"    Test-Time Augmentation: {args.use_tta}")
        logging.info(f"    Overlap ratio: {args.overlap}")
        logging.info(f"    Compression: {args.compression}")
        
        # Load training parameters and normalization stats
        network_params = load_training_params(checkpoint_path)
        
        # Try to load saved normalization statistics from training
        norm_stats = load_normalization_stats(checkpoint_path)
        if norm_stats is not None:
            logging.info(f"Using normalization statistics from training")
        else:
            logging.info(f"No saved normalization found, will compute from volume")
        
        # Create the model architecture and load trained weights
        logging.info("Creating and loading model...")
        
        # Determine CUDA device
        if torch.cuda.is_available():
            cuda_device = args.cuda_device
            logging.info(f"    Using GPU device: cuda:{cuda_device}")
        else:
            cuda_device = 0
            logging.info(f"    CUDA not available, using CPU")
        
        # Create model (with architecture from training parameters)
        # Use norm_division_factor from training parameters, default to 1 if not found
        norm_division_factor = network_params.get('norm_division_factor', 1)
        logging.info(f"    Using norm_division_factor: {norm_division_factor}")
        
        model = create_model(
            device=cuda_device if torch.cuda.is_available() else 'cpu',
            norm_division_factor=norm_division_factor
        )
        
        # Load the trained weights from checkpoint
        model = load_checkpoint(model, checkpoint_path)
   
        # Load and preprocess the volume (with normalization stats if available)
        logging.info("Loading and preprocessing volume...")
        test_volume, norm_mean, norm_std = _load_and_preprocess_volume(test_volume_path, norm_stats)
        
        # Run inference        
        logging.info("Running inference...")
        pred_volume = _run_inference(model, test_volume, args.batch_size, 
                                   tuple(network_params['train_patch_size']), 
                                   args.use_tta, args.overlap)

        # GPU RAM memory monitoring:
        if torch.cuda.is_available():
            max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
            logging.info(f"GPU memory peak: {max_memory_allocated:.2f} GB")          
            torch.cuda.reset_peak_memory_stats()
        
        # RAM memory monitoring:
        process = psutil.Process()
        peak_ram_gb = process.memory_info().rss / 1024**3
        logging.info(f"RAM memory peak: {peak_ram_gb:.2f} GB")
        
        # Denormalize the output volume back to original gray level range
        pred_volume = pred_volume * norm_std + norm_mean
        logging.info(f"Denormalized output volume using mean={norm_mean:.6f}, std={norm_std:.6f}")

        # Success message
        logging.info("Inference completed successfully")
        
        # Save the denoised volume
        logging.info("Saving output...")
        output_path = Path(output_path)
        output_dir = output_path.parent
        
        # Determine if output should be multilayer based on file extension
        multilayer = output_path.suffix.lower() == '.tif'
        
        save_output(pred_volume, output_dir, network_params, 
                   args.use_tta, args.overlap, args.batch_size, cuda_device,
                   args.compression, multilayer, output_path.name, 
                   args.input_json, checkpoint_path)   


    except Exception as e:
        # Log any errors that occur during the process
        logging.error(f"Inference failed: {e}")
        # Re-raise the exception to ensure non-zero exit code
        raise


if __name__ == "__main__":

    # Parse and validate command line arguments
    parse = ArgumentParser(description="Load a model trained with Noise2Inverse, and use it to denoise a volume")
    
    # Required arguments
    parse.add_argument('input_json', help='Path to JSON file containing input/output paths and checkpoint directory')
    
    # Optional arguments with default values
    parse.add_argument('--batch_size', default=4, type=int, help='The number of patches per batch')
    parse.add_argument('--cuda_device', default=0, type=int, help="CUDA device to use (default: 0)")
    parse.add_argument('--no_tta', action='store_true', help='Disable Test-Time Augmentation (default: enabled)')
    parse.add_argument('--overlap', default=0.8, type=float, help='Overlap ratio between patches for sliding window inference')
    parse.add_argument('--no_compression', action='store_true', help='Disable compression in output TIFF files (default: enabled)')
    
    args = parse.parse_args()
    
    # Handle flag logic (default to True, disable if flag is set)
    args.use_tta = not args.no_tta
    args.compression = not args.no_compression
    
    main(args)

