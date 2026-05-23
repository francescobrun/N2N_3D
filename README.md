# N2N_3D: Noise2Noise (or Noise2Inverse) 3D Volume Denoising

A PyTorch implementation for 3D volumetric image denoising using self-supervised deep learning. This repository provides a complete pipeline for training and inference with a 3D U-Net architecture, specifically designed for denoising  volumetric data without requiring clean target images. Instead, it only needs two replicas of the same noisy volume, as proposed in Noise2Noise or related variations such as e.g. Noise2Inverse. 

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

**Optional ROI crop**: an additional `crop` entry can restrict training to a
bounding box within `split1`/`split2`. Useful when the volume contains a
region you do not want to spend network capacity on (e.g. skull around a
brain, air around a sample). Coordinates are half-open `[start, end)` voxel
ranges; axes are `x` → axis 0, `y` → axis 1, `z` → axis 2. Inference is
unaffected — the test volume is processed as-is.

```json
{
    "split1_volume_file": "path/to/your/volume_part1.tif",
    "split2_volume_file": "path/to/your/volume_part2.tif",
    "checkpoint_path": "path/to/checkpoints/",
    "test_volume_file": "path/to/your/volume_full.tif",
    "output_file": "path/to/output/denoised.tif",
    "crop": {
        "x": [100, 500],
        "y": [50, 450],
        "z": [200, 800]
    }
}
```

### 2. Train the Model

```bash
python train.py path/to/your/config.json 
```

**Note**: Training is designed to be executed overnight, which means that 50 epochs of training should take about 12 hours.

#### Optional Arguments:

- `--loaded_checkpoint_path`: Path to a checkpoint file to resume training from (default: None)
- `--nb_train_epoch`: Number of training epochs (default: 50)
- `--batch_size`: Number of patches per batch (default: 32)
- `--cuda_device`: CUDA device to use (default: 0)
- `--norm_division_factor`: Division factor for group normalization (default: 1, i.e. "instance")
- `--no_half`: Disable fp16 mixed precision training (default: enabled on tensor-core GPUs only, i.e. compute capability >= 7.0). Automatically skipped on older cards (GTX 10-series / Pascal) where fp16 would be slower than fp32.

#### Examples:

```bash
# Basic training with default settings
python train.py config.json

# Training with custom batch size and more epochs
python train.py config.json --batch_size 16 --nb_train_epoch 100

# Training with layer normalization
python train.py config.json --norm_division_factor 56

# Training with custom normalization (28 groups)
python train.py config.json --norm_division_factor 2

# Resume training from checkpoint
python train.py config.json --loaded_checkpoint_path checkpoints/weights_epoch_020.torch
```

### 3. Run Inference

```bash
python inference.py path/to/your/config.json 
```

**Note**: Inference will automatically use the latest checkpoint available in the checkpoint directory.

**Note**: Inference should take minutes.

#### Optional Arguments:

- `--batch_size`: Number of patches processed simultaneously (default: 4)
- `--cuda_device`: CUDA device to use (default: 0)
- `--tta`: Enable Test-Time Augmentation (default: disabled)
- `--overlap`: Overlap ratio between patches for sliding window inference (default: 0.8)
- `--no_compression`: Disable compression in output TIFF files (default: enabled)
- `--no_half`: Disable fp16 mixed precision inference (default: enabled on tensor-core GPUs only). fp16 is automatically skipped on older GPUs without tensor cores (compute capability < 7.0, e.g. GTX 10-series / Pascal), where fp16 would be slower than fp32.

**Note**: `norm_division_factor` is automatically loaded from the training parameters to ensure consistency with the trained model.

#### Examples:

```bash
# Basic inference with default settings
python inference.py config.json

# Inference with larger batch size (uses more memory)
python inference.py config.json --batch_size 8

# Inference with Test-Time Augmentation (slower but potentially higher quality)
python inference.py config.json --tta

# Inference with custom overlap (lower overlap = less quality but faster)
python inference.py config.json --overlap 0.5

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




