# N2N_3D: Noise2Noise 3D Volume Denoising

A PyTorch implementation for 3D volumetric image denoising using self-supervised deep learning. This repository provides a complete pipeline for training and inference with a 3D U-Net architecture, specifically designed for denoising  volumetric data without requiring clean target images. Instead, it only needs two replicas of the same noisy volume, as proposed in Noise2Noise or related variations such as e.g. Noise2Inverse, Half2Half, and similar self-supervised paradigms that produce two reconstructions with identical underlying content but independent noise. The pipeline is agnostic to how the split pair is produced — projection-view splitting, detector-side splitting, Poisson photon thinning (Half2Half), repeated independent acquisitions, or any other strategy that satisfies the Noise2Noise assumptions — as long as the two input volumes share the same shape, scale, and underlying scene.

## 📚 Citation

If you use this code in your research, please cite the following article: https://doi.org/10.1364/OE.471439

## 🌟 Features

- **3D U-Net Architecture**: Custom 3D U-Net with optimized architecture for volumetric data (using MONAI)
- **Memory Efficient**: Optimized data loading and processing pipelines enabling it to run on modest GPUs (two 512×512×512 voxel volumes can be processed on a GPU with less than 4GB of VRAM by reducing batch size)

## 📋 Installation

The only less common library used in this repository is MONAI (https://project-monai.github.io/). All other dependencies are trivial and are reported in [requirements.txt](requirements.txt).

```bash
# Clone the repository
git clone https://github.com/yourusername/N2I_3D.git
cd N2I_3D

# Install dependencies
pip install -r requirements.txt

# Note: For GPU acceleration, install PyTorch with CUDA support
# Visit https://pytorch.org/get-started/locally/ for CUDA-specific commands
```

## 🚀 Quick Start

### 1. Prepare Your Data

Prepare your three multi-layer TIFF files and choose an empty folder to store checkpoints during training. Then, create a JSON configuration file for your dataset where you also specify the name of the output file that will be created after inference, as follows:

```json
{
    "split1_volume_file": "path/to/your/volume_part1.tif",
    "split2_volume_file": "path/to/your/volume_part2.tif", 
    "checkpoint_path": "path/to/checkpoints/",
    "test_volume_file": "path/to/your/volume_full.tif", 
    "output_file": "path/to/output/denoised.tif"
}
```
**Note**: In principle, `test` can also be the same file as `split1` or `split2`.

**Note**: `split1` and `split2` must have identical dimensions (they're the two noisy copies of the same content). `test` can be a different size — for example, a cropped training pair with a full-volume test image.

**Note**: It is not required that the volumes are perfect cubes of N×N×N voxels.

**Optional ROI crop**: an additional `training_crop` entry can restrict
training to a bounding box within `split1`/`split2`. Useful when the volume
contains a region you do not want to spend network capacity on (e.g. skull
around a brain, air around a sample). Coordinates are half-open
`[start, end)` voxel ranges and follow the standard 3D imaging convention
for a multi-layer TIFF loaded as a `(n_slices, height, width)` array:
`z` → axis 0 (slice / depth), `y` → axis 1 (row), `x` → axis 2 (column).

**Optional circular mask**: an additional `training_circle_mask` entry can
further restrict training to a 2D circle in the y-x plane (extended through
every z slice as a cylinder). The natural shape for CT reconstructions
whose valid signal lives inside the inscribed circle of each square slice —
the four corners outside the circle are typically zero or reconstruction
artifacts and you don't want them in the normalization stats or in any
training patch. Coordinates are in **original (pre-crop) voxel space**
because the mask describes the geometry of the scan itself, independent of
the user's crop choice. `radius` is required and accepts either a positive
number or the string `"auto"` (use the inscribed circle of the original
slice, i.e. half of the smaller of the y and x extents of the uncropped
volume — the typical CT case). `center_y` and `center_x` are optional and
default to the geometric center of the original y and x axes. When
`training_crop` is also set, the crop is applied first and the circle
center is then translated to the cropped origin; patch sampling enforces
both constraints, so the effective training region is the intersection of
the crop's bounding box and the scan-valid circle. Inference is
unaffected — the test volume is processed as-is.

```json
{
    "split1_volume_file": "path/to/your/volume_part1.tif",
    "split2_volume_file": "path/to/your/volume_part2.tif",
    "checkpoint_path": "path/to/checkpoints/",
    "test_volume_file": "path/to/your/volume_full.tif",
    "output_file": "path/to/output/denoised.tif",
    "training_crop": {
        "z": [200, 800],
        "y": [50, 450],
        "x": [100, 500]
    },
    "training_circle_mask": {
        "radius": "auto"
    }
}
```

### 2. Train the Model

```bash
python train.py path/to/your/config.json 
```

**Note**: Training is designed to be executed overnight, which means that 50 epochs of training should take about 12 hours.

**Note**: Alongside the per-epoch checkpoints, training writes a `training_loss.csv` file in the checkpoint directory with one row per epoch (`epoch,mean_loss`). Useful for plotting the loss curve afterward and judging whether more epochs would help.

#### Optional Arguments:

- `--loaded_checkpoint_path`: Path to a checkpoint file to resume training from (default: None)
- `--nb_train_epoch`: Number of training epochs (default: 50)
- `--batch_size`: Number of patches per batch (default: 16)
- `--cuda_device`: CUDA device to use. A non-negative integer selects that specific GPU; the string `auto` (default) picks the GPU with the most free memory via `nvidia-smi`. Useful on shared multi-GPU machines to avoid colliding with other users. Falls back to GPU 0 if `nvidia-smi` is unavailable.
- `--norm_division_factor`: Division factor for group normalization (default: 56, i.e. layer normalization — num_groups=1). Empirically the best pairing with the residual learning wrapper because it preserves inter-channel structure that helps signal/noise discrimination. Setting it to 1 selects instance normalization (num_groups=56); intermediate divisors of 56 give true group normalization. Valid values: 1, 2, 4, 7, 8, 14, 28, 56.
- `--num_res_units`: Number of residual conv units per level inside the MONAI U-Net (default: 0, a plain conv block per level). This is MONAI's *intra-block* residual learning and is distinct from the image-level residual wrapper (`output = input + unet(input)`) that is always active. Values of `1` or `2` add deeper per-level blocks with internal skip connections, which can improve denoising fidelity and edge sharpness (helpful if outputs look over-smoothed) at the cost of more compute, memory, and parameters — so adjust batch size accordingly and **requires retraining**. The value is recorded in `params.json` and read back automatically at inference time so the reconstructed architecture matches the trained one.
- `--unet_depth`: Number of U-Net levels (default: 4). Channels start at 56 and double per level, so depth 4 = (56, 112, 224, 448) and depth 3 = (56, 112, 224); there are `unet_depth - 1` downsampling stages. **Fewer levels** keep detail at a finer resolution and reduce the smoothing that comes from the coarse bottleneck (helpful for sharpness) but shrink the receptive field; **more levels** widen spatial context at the cost of more downsampling. Must be >= 2 and **requires retraining**. Recorded in `params.json` and read back automatically at inference.
- `--unet_stride`: Downsampling factor applied uniformly at every U-Net stage (default: 2). **Set to 1** for a no-downsampling, full-resolution network (MSD-like): the sharpest option since no spatial information is lost, but dramatically more memory- and compute-hungry — reduce `--batch_size` accordingly. Must be >= 1 and **requires retraining**. Recorded in `params.json` and read back automatically at inference.
- `--no_half`: Disable fp16 mixed precision training (default: enabled on tensor-core GPUs only, i.e. compute capability >= 7.0). Automatically skipped on older cards (GTX 10-series / Pascal) where fp16 would be slower than fp32.
- `--no_compile`: Disable `torch.compile` (default: enabled when PyTorch 2.0+ is available **and** the GPU has compute capability >= 7.0). When enabled, the model is graph-compiled before training for a ~1.2-1.5x speedup; the first training step is slower while compilation runs. Automatically skipped on Pascal and earlier (compute < 7.0) where the Triton backend is unsupported, and falls back to eager mode if PyTorch is older than 2.0 or compilation raises. Checkpoints are written in the same format regardless, so inference can load them either way.
- `--keep_only_last`: Keep only the most recent epoch's checkpoint on disk; the previous epoch's `weights_epoch_NNN.torch` is deleted after each save (default: every epoch is preserved, useful for testing/ablation).
- `--loss`: Loss function used during training. Either `mse` (default) or `l1`. Both are valid for Noise2Noise on symmetric noise distributions: MSE recovers the conditional mean (the standard N2N choice, slightly over-smoothed output), while L1 recovers the conditional median (often visibly sharper edges, comparable flat-region quality). The chosen loss is persisted to `params.json` for traceability.

#### Examples:

```bash
# Basic training with default settings
python train.py config.json

# Training with custom batch size and more epochs
python train.py config.json --batch_size 16 --nb_train_epoch 100

# Training with instance normalization (the previous default; 56 groups)
python train.py config.json --norm_division_factor 1

# Training with intermediate group normalization (e.g. 14 groups)
python train.py config.json --norm_division_factor 4

# Resume training from checkpoint
python train.py config.json --loaded_checkpoint_path checkpoints/weights_epoch_020.torch
```

### 3. Run Inference

```bash
python inference.py path/to/your/config.json 
```

**Note**: Inference will automatically use the latest checkpoint available in the checkpoint directory.

**Note**: Inference runtime depends strongly on the `--overlap` setting and the test volume size, since patch count scales roughly as `1/(1-overlap)³`. At the default overlap of 0.5 on a large volume (e.g. ~400×500×1000 voxels), expect well under an hour on a consumer GPU; raising it (e.g. to 0.85) improves blending at the seams but multiplies runtime several-fold. On smaller volumes or with a larger `--batch_size`, runtime drops accordingly. fp16 mixed precision (enabled by default on tensor-core GPUs) further reduces it. The image-level residual wrapper keeps seams small even at low overlap, so 0.5 is a good speed/quality default.

#### Optional Arguments:

- `--batch_size`: Number of patches processed simultaneously (default: 4)
- `--cuda_device`: CUDA device to use. A non-negative integer selects that specific GPU; the string `auto` (default) picks the GPU with the most free memory via `nvidia-smi`. Useful on shared multi-GPU machines to avoid colliding with other users. Falls back to GPU 0 if `nvidia-smi` is unavailable.
- `--overlap`: Overlap ratio between patches for sliding window inference (default: 0.5). Higher values reduce patch-boundary seams but increase runtime sharply (patch count scales ~`1/(1-overlap)³`); lower values are faster with slightly more risk of visible seams.
- `--no_compression`: Disable compression in output TIFF files (default: enabled)
- `--no_half`: Disable fp16 mixed precision inference (default: enabled on tensor-core GPUs only). fp16 is automatically skipped on older GPUs without tensor cores (compute capability < 7.0, e.g. GTX 10-series / Pascal), where fp16 would be slower than fp32.
- `--no_compile`: Disable `torch.compile` (default: enabled when PyTorch 2.0+ is available **and** the GPU has compute capability >= 7.0). When enabled, the model is graph-compiled before inference for ~1.2-1.5x speedup; the first inference call is slower (typically 30-90s) while compilation runs. Automatically skipped on Pascal and earlier (GTX 10-series, compute < 7.0) because `torch.compile`'s Triton backend doesn't support those cards. Also falls back to eager mode if PyTorch is older than 2.0 or compilation raises.
- `--gpu_aggregation`: Keep the sliding-window aggregation buffer on GPU instead of CPU (default: CPU). Faster inference (~1.2-1.5x by eliminating the per-patch GPU→CPU sync) but uses roughly `2 * D * H * W * 4` bytes of additional VRAM (e.g. ~1.6 GB for a 400×500×1000 voxel volume). Recommended only on cards with ample free VRAM after the model and input volume are loaded.
- `--no_padding`: Disable replicate-padding of the test volume by half the training patch size on each side (default: enabled). Padding ensures every output voxel is predicted from well-conditioned patch-center context; disabling it cuts inference time roughly 1.5-2x at the cost of slightly degraded predictions in the outermost ~half-patch of the volume. With image-level residual learning the cost of disabling is small (the identity pathway preserves input values where the network is uncertain), so this is a reasonable speedup if you don't rely on the outer boundary voxels of the output.

**Note**: `norm_division_factor` is automatically loaded from the training parameters to ensure consistency with the trained model.

#### Examples:

```bash
# Basic inference with default settings
python inference.py config.json

# Inference with larger batch size (uses more memory)
python inference.py config.json --batch_size 8

# Inference with higher overlap (smoother seam blending but slower)
python inference.py config.json --overlap 0.85

# Inference without compression (larger output file size)
python inference.py config.json --no_compression
```

### Input/Output Formats
- **Multi-layer TIFF**: Single file with all slices (default)
- **Metadata**: Complete processing parameters embedded in TIFF tags

## 📊 Performance Tips

1. **GPU Memory**: Adjust batch size based on available GPU memory.
2. **norm_division_factor**: Higher values (e.g., 2 or 4) could improve the results but may cause oversmoothing. Batch size should be adjusted accordingly.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Code was restructured with heavy inspiration from the SSD_3D repository (https://github.com/xni-esrf/SSD_3D)

---

**Note**: This implementation is specifically designed for 3D images. For 2D images, please consider other repositories.




