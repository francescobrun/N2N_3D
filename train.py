import json
import itertools
import torch
import gc
import psutil
import numpy as np
import tifffile
import torchvision
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
import torchio as tio

from _model import create_model


# ============================================================================
# TRAINING HYPERPARAMETERS (Hard-coded for optimal quality)
# ============================================================================

# Training data
TRAIN_PATCH_SIZE = [96, 96, 96]  # Patch size for training
NB_PATCH_PER_EPOCH = 17600  # Number of patches per epoch

# Optimization
LEARNING_RATE = 0.0005  # Learning rate for Adam optimizer
WEIGHT_DECAY = 0.0  # Weight decay coefficient for L2 regularization


# ============================================================================
# CUSTOM TRANSFORMS AND DATASET CLASSES
# ============================================================================

def _cube_rotation_group():
    """Return the 24 proper rotations of the cube as (axes_perm, sign_flips).

    Each element is a signed permutation matrix. Filtering signed permutations
    by determinant = +1 yields exactly the rotation group (chiral octahedral
    group, order 24). No duplicates, no reflections.
    """
    rotations = []
    for perm in itertools.permutations((0, 1, 2)):
        inversions = sum(
            1 for i in range(3) for j in range(i + 1, 3) if perm[i] > perm[j]
        )
        perm_sign = 1 if inversions % 2 == 0 else -1
        for signs in itertools.product((1, -1), repeat=3):
            if perm_sign * signs[0] * signs[1] * signs[2] == 1:
                rotations.append((perm, signs))
    assert len(rotations) == 24
    return rotations


CUBE_ROTATIONS = _cube_rotation_group()

class CubeSymmetryTransform(tio.Transform):
    """
    Complete set of 24 rotational symmetries of a cube.
    For each training pair, a random orientation is selected from the 24 possible 
    rotational symmetries of a cube. Finally, a horizontal flip is applied with 
    a 50% probability.
    
    Uses explicit permutation matrices for maximum robustness and performance.
    """
    
    def _apply_rotation(self, tensor, rotation_idx):
        """Apply a specific rotation using permutation indices."""
        axes_perm, flip_dirs = CUBE_ROTATIONS[rotation_idx]
        
        # Apply axis permutation
        tensor = tensor.permute(axes_perm)
        
        # Apply flips for each axis
        for i, flip in enumerate(flip_dirs):
            if flip == -1:
                tensor = tensor.flip(i)
        
        return tensor
    
    def _geom_transform(self, tensors_to_transform):
        """Apply one of 24 rotational symmetries + optional horizontal flip."""
        # Randomly select one of 24 rotations
        rotation_idx = torch.randint(0, 24, [1]).item()
        # Random horizontal flip with 50% probability
        cur_h_flip = torch.randint(0, 2, [1]).item()
        
        for i in range(len(tensors_to_transform)):
            tensors_to_transform[i] = torch.squeeze(tensors_to_transform[i])
            # Shape: (D, H, W)
            
            # Apply selected rotation
            tensors_to_transform[i] = self._apply_rotation(tensors_to_transform[i], rotation_idx)
            
            # Randomly horizontally flip
            if cur_h_flip == 1:
                tensors_to_transform[i] = torchvision.transforms.functional.hflip(tensors_to_transform[i])
            
            tensors_to_transform[i] = torch.unsqueeze(tensors_to_transform[i], 0)
        
        return tensors_to_transform
    
    def apply_transform(self, subject):
        """
        Apply the selective symmetry transform to a TorchIO subject.
        """
        tensors = [
            subject['split1_volume'].data,
            subject['split2_volume'].data,
        ]

        t1, t2 = self._geom_transform(tensors)

        subject['split1_volume'].set_data(t1)
        subject['split2_volume'].set_data(t2)

        return subject


class N2IDataset(Dataset):
    """
    Noise2Inverse dataset that preloads both volumes into memory for fast training.
    Both split volumes are loaded during initialization for efficient patch access.
    """
    
    def __init__(self, dataset_name, training_patch_size, nb_patches, normalization=True):
        
        # Load dataset metadata       
        with open(dataset_name, 'r') as f:
            dataset_info = json.load(f)
        
        split1_path = dataset_info["split1_volume_file"]
        split2_path = dataset_info["split2_volume_file"]
        
        # Preload both volumes into memory
        print("Loading training volumes into memory...")
        self.split1_volume = tifffile.imread(split1_path).astype(np.float32)
        self.split2_volume = tifffile.imread(split2_path).astype(np.float32)
        
        self.volume_shape = self.split1_volume.shape
        
        # Always use full volume (no cropping)
        self.sampling_shape = self.volume_shape
        self.sampling_offset = (0, 0, 0)
        
        self.patch_size = training_patch_size
        self.nb_patches = nb_patches
        self.normalization = normalization
        
        # Pre-compute normalization statistics if needed
        self.mean = None
        self.std = None
        
        if self.normalization:
            print("Computing normalization statistics...")
            # Compute unified mean and std from both volumes
            combined_data = np.concatenate([self.split1_volume.flatten(), self.split2_volume.flatten()])
            self.mean = np.float32(combined_data.mean())
            self.std = np.float32(combined_data.std())
            del combined_data

        # Move volumes to shared memory so DataLoader workers do not duplicate them
        # across processes (important on Windows where spawn-mode workers would
        # otherwise pickle a full copy of each volume per worker).
        self.split1_volume = torch.from_numpy(self.split1_volume)
        self.split2_volume = torch.from_numpy(self.split2_volume)
        self.split1_volume.share_memory_()
        self.split2_volume.share_memory_()

        # Reused for every __getitem__ call; the transform is stateless (random
        # state is sampled fresh inside apply_transform).
        self.transform = CubeSymmetryTransform()

    def _load_patch(self, volume, start_coords):
        """Extract a patch from preloaded volume."""
        x, y, z = start_coords
        patch = volume[x:x+self.patch_size[0], y:y+self.patch_size[1], z:z+self.patch_size[2]].clone()
        return patch.unsqueeze(0)  # Add channel dimension
    
    def __len__(self):
        return self.nb_patches
    
    def __getitem__(self, idx):
        # Generate random patch coordinates within valid sampling region
        max_x = self.sampling_shape[0] - self.patch_size[0]
        max_y = self.sampling_shape[1] - self.patch_size[1]
        max_z = self.sampling_shape[2] - self.patch_size[2]
        
        start_x = np.random.randint(0, max_x + 1) + self.sampling_offset[0]
        start_y = np.random.randint(0, max_y + 1) + self.sampling_offset[1]
        start_z = np.random.randint(0, max_z + 1) + self.sampling_offset[2]
        
        # Load patches from preloaded volumes (returns float32 torch tensors)
        patch1 = self._load_patch(self.split1_volume, (start_x, start_y, start_z))
        patch2 = self._load_patch(self.split2_volume, (start_x, start_y, start_z))

        # Apply normalization if needed
        if self.normalization:
            patch1 = (patch1 - self.mean) / (self.std + 1e-7)
            patch2 = (patch2 - self.mean) / (self.std + 1e-7)

        # Apply cube symmetry transform
        subject = tio.Subject(
            split1_volume=tio.ScalarImage(tensor=patch1),
            split2_volume=tio.ScalarImage(tensor=patch2)
        )
        subject = self.transform(subject)
        
        return {
            'split1_volume': subject['split1_volume'],
            'split2_volume': subject['split2_volume']
        }



def _worker_init_fn(worker_id):
    """Seed numpy independently in each DataLoader worker.

    Without this, every worker inherits the same numpy RNG state and the random
    patch coordinates in __getitem__ end up correlated across workers. PyTorch
    already seeds its own and Python's RNGs per worker; we only need to forward
    that to numpy.
    """
    np.random.seed(torch.initial_seed() % 2**32)


def save_model(model, optimizer, epoch, save_path):
    """Save model checkpoint with PyTorch's built-in compression."""
    state = {
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }    
    # Use PyTorch's compression (recommended)
    torch.save(state, save_path, _use_new_zipfile_serialization=True)


def train_model(dl, model, loss_func, optimizer, 
                checkpoint_dir, loaded_checkpoint_path, nb_train_epoch, device):
    """Train the model with logic similar to train_old.py."""
    
    start_epoch_nb = 0
    
    # Load checkpoint if specified
    if loaded_checkpoint_path is not None:
        print("Loading weights...")
        state = torch.load(loaded_checkpoint_path, map_location=torch.device(device), weights_only=True)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch_nb = state['epoch']+1

    # Training loop    
    first_epoch_completed = False
    for epoch in range(start_epoch_nb, nb_train_epoch):
        epoch_loss = 0
        
        for batch in tqdm(dl, desc=f'Epoch {epoch+1}/{nb_train_epoch}'):
            # Per-sample: with 50% probability swap which noisy copy is input vs. target
            data1 = batch["split1_volume"][tio.DATA].to(device)
            data2 = batch["split2_volume"][tio.DATA].to(device)
            swap = (torch.rand(data1.shape[0], device=device) < 0.5).view(-1, 1, 1, 1, 1)
            input = torch.where(swap, data2, data1)
            target = torch.where(swap, data1, data2)
            
            # Proceed to a training step
            optimizer.zero_grad()
            pred = model(input)
            
            # Use MSE loss. Loss is scaled by 1000 to counteract Adam's eps=1e-8
            # damping the update when gradients are small (z-score-normalized
            # inputs + MSE produce tiny gradient magnitudes). Equivalent to
            # using eps=1e-11; the scaling form is kept for numerical safety.
            total_loss = loss_func(pred, target)
            loss_val = total_loss * 1000
            
            loss_val.backward()
            optimizer.step()
            epoch_loss += loss_val.item() / len(dl)

        print(f"Mean loss value of the epoch : {epoch_loss:.4f}")
        
        # Show memory monitoring only after first epoch
        if not first_epoch_completed and torch.cuda.is_available():
            max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
            print(f"GPU memory peak: {max_memory_allocated:.2f} GB")          
            torch.cuda.reset_peak_memory_stats()
        
        if not first_epoch_completed:
            process = psutil.Process()
            peak_ram_gb = process.memory_info().rss / 1024**3
            print(f"RAM memory peak: {peak_ram_gb:.2f} GB")
        
        first_epoch_completed = True    

        # Save checkpoint at each epoch
        print("Saving checkpoint for epoch n°{}...".format(epoch))
        save_model(model, optimizer, epoch, checkpoint_dir / f"weights_epoch_{epoch:03d}.torch")
        
        # Clear memory to prevent CUDA out of memory errors
        del epoch_loss
        gc.collect()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(params):
    """Main training function."""
   
    with open(params.input_json, 'r') as f:
        dataset_info = json.load(f)
    
    # Get checkpoint directory from JSON and create it if it doesn't exist
    checkpoint_dir = Path(dataset_info["checkpoint_path"])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{params.cuda_device}")
        print(f"Using GPU device: cuda:{params.cuda_device}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    # Initialize the model to be trained
    model = create_model(device=params.cuda_device if torch.cuda.is_available() else 'cpu',
                        norm_division_factor=getattr(params, 'norm_division_factor', 1))

    # Create the memory-efficient data loading pipeline
    print("Setting up data loader...")
    train_dataset = N2IDataset(
        params.input_json,
        TRAIN_PATCH_SIZE,
        NB_PATCH_PER_EPOCH,
        normalization=True 
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=params.num_workers,
        persistent_workers=params.num_workers > 0,
        worker_init_fn=_worker_init_fn if params.num_workers > 0 else None,
    )

    # Create loss function and optimizer
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=WEIGHT_DECAY, lr=LEARNING_RATE)

    # Save the training parameters including normalization statistics
    params_dict = dict(vars(params))
    params_dict.update({
        # Data parameters:  
        'normalization_mean': float(train_dataset.mean),
        'normalization_std': float(train_dataset.std),
        # Training parameters:
        'loaded_checkpoint_path': params.loaded_checkpoint_path,
        'loss_function': loss_func.__class__.__name__,  
        'optimizer': optimizer.__class__.__name__,  
        'learning_rate': LEARNING_RATE,  
        'weight_decay': WEIGHT_DECAY,        
        'train_patch_size': TRAIN_PATCH_SIZE,
        'nb_patch_per_epoch': NB_PATCH_PER_EPOCH,
        'nb_train_epoch': params.nb_train_epoch,
        'training_cuda_device': params.cuda_device,
        'training_batch_size': params.batch_size,
        # UNet model architecture parameters:
        'unet_in_channels': model.in_channels,
        'unet_out_channels': model.out_channels,
        'unet_channels': model.channels,
        'unet_strides': model.strides,
        'unet_kernel_size': model.kernel_size,
        'unet_up_kernel_size': model.up_kernel_size,
        'unet_num_res_units': model.num_res_units,
        'unet_act': model.act,
        'unet_norm': model.norm,
        'unet_dropout': model.dropout
    })
    with open(checkpoint_dir / "params.json", 'w') as par_file:
        json.dump(params_dict, par_file)
    print(f"Saved training parameters with normalization statistics: mean={train_dataset.mean:.6f}, std={train_dataset.std:.6f}")

    # Train model
    train_model(
        train_loader,
        model,
        loss_func,
        optimizer,
        checkpoint_dir,
        params.loaded_checkpoint_path,
        params.nb_train_epoch,
        device
    )


if __name__ == "__main__":

    parse = ArgumentParser(description="Train a model with Noise2Inverse, using 3d convolutions")
    parse.add_argument('input_json', help='Path to JSON file containing dataset information and processing paths')
    parse.add_argument('--loaded_checkpoint_path', default=None, help="If set, load the checkpoint located at the provided path")
    parse.add_argument('--nb_train_epoch', default=50, type=int, help="The number of training epochs")
    parse.add_argument('--batch_size', default=32, type=int, help="The number of patch per batch")
    parse.add_argument('--cuda_device', default=0, type=int, help="CUDA device to use (default: 0)")
    parse.add_argument('--norm_division_factor', default=1, type=int, help="Division factor for group normalization (1=instance norm, 56=layer norm)")
    parse.add_argument('--num_workers', default=4, type=int, help="Number of DataLoader worker processes (default: 4; use 0 on very low-RAM systems)")

    main(parse.parse_args())
