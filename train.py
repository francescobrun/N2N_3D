import json
import itertools
import logging
import time
import torch
import psutil
import numpy as np
import tifffile
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader

from _model import create_model


# Enable cuDNN autotuning and TF32. Training and inference both run fixed-shape
# convolutions (96^3 patches / fixed sliding-window roi), so cuDNN's benchmark
# mode pays its one-time autotune cost back immediately by selecting the fastest
# conv algorithm. TF32 (Ampere+; a no-op on older cards) speeds up the fp32
# fallback paths with precision that is irrelevant for denoising.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# AMP API compatibility: PyTorch 2.0+ exposes torch.amp.GradScaler / torch.autocast
# (device-agnostic), while older PyTorch (~1.6 to ~1.13) exposes torch.cuda.amp.*.
# Both implementations are functionally equivalent for our usage; we pick whichever
# the installed version provides so the pipeline runs on older toolchains too.
if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):

    def _make_gradscaler(enabled):
        return torch.amp.GradScaler("cuda", enabled=enabled)

else:

    def _make_gradscaler(enabled):
        return torch.cuda.amp.GradScaler(enabled=enabled)


if hasattr(torch, "autocast"):

    def _amp_autocast(enabled):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)

else:

    def _amp_autocast(enabled):
        return torch.cuda.amp.autocast(enabled=enabled)


# torch.load gained a `weights_only` keyword in PyTorch 1.13 (and the default
# flipped to True in 2.6 with a FutureWarning otherwise). On older PyTorch the
# argument doesn't exist and passing it raises TypeError. Detect once and call
# torch.load with or without the kwarg accordingly.
import inspect as _inspect

_TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = (
    "weights_only" in _inspect.signature(torch.load).parameters
)
del _inspect


def _safe_torch_load(path, map_location):
    if _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY:
        return torch.load(path, map_location=map_location, weights_only=True)
    return torch.load(path, map_location=map_location)


def setup_logging() -> None:
    """Configure logging settings for the training script.

    Mirrors the format used by inference.py so a back-to-back train/infer
    session has consistent timestamped output that can be cross-correlated
    by wall clock.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


class CubeSymmetryTransform:
    """
    Complete set of 24 rotational symmetries of a cube.
    A single random orientation is drawn from the 24 proper rotations of a cube
    and applied identically to every tensor in the pair, followed by a 50%
    horizontal flip — together covering the full 48-element octahedral group.

    Operates directly on (C, D, H, W) tensors. The same random orientation is
    applied to both members of the pair so the two noisy copies stay registered.
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

    def __call__(self, tensors):
        """Apply one of 24 rotational symmetries + optional horizontal flip to
        a list of (C, D, H, W) tensors, returning the transformed list."""
        # Randomly select one of 24 rotations
        rotation_idx = torch.randint(0, 24, [1]).item()
        # Random horizontal flip with 50% probability
        cur_h_flip = torch.randint(0, 2, [1]).item()

        for i in range(len(tensors)):
            t = torch.squeeze(tensors[i])  # (C, D, H, W) -> (D, H, W)

            # Apply selected rotation
            t = self._apply_rotation(t, rotation_idx)

            # Randomly horizontally flip (flips the last/W axis)
            if cur_h_flip == 1:
                t = t.flip(-1)

            tensors[i] = torch.unsqueeze(t, 0)  # restore channel dim

        return tensors


def _parse_crop(crop, volume_shape):
    """Validate and normalize the optional 'training_crop' entry from the JSON config.

    Accepts a dict of the form {"x": [start, end], "y": [start, end], "z":
    [start, end]} with half-open intervals in voxel coordinates (start
    inclusive, end exclusive). Follows the standard 3D imaging convention:
    z -> axis 0 (slice/depth), y -> axis 1 (row), x -> axis 2 (column).
    Returns either None (no crop requested) or a tuple of three (start, end)
    pairs in axis-0, axis-1, axis-2 order.
    """
    if crop is None:
        return None
    if not isinstance(crop, dict) or set(crop.keys()) != {"x", "y", "z"}:
        raise ValueError(
            f"'training_crop' must be a dict with exactly the keys 'x', 'y', 'z' (got {crop})"
        )
    # JSON key -> axis index (standard imaging convention)
    axis_for_key = {"z": 0, "y": 1, "x": 2}
    ranges = [None, None, None]
    for axis_name, axis in axis_for_key.items():
        rng = crop[axis_name]
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            raise ValueError(
                f"training_crop[{axis_name!r}] must be a 2-element [start, end] list (got {rng})"
            )
        start, end = int(rng[0]), int(rng[1])
        if start < 0 or end <= start:
            raise ValueError(
                f"training_crop[{axis_name!r}] must satisfy 0 <= start < end (got [{start}, {end}])"
            )
        if end > volume_shape[axis]:
            raise ValueError(
                f"training_crop[{axis_name!r}] end={end} exceeds volume size "
                f"{volume_shape[axis]} on axis {axis}"
            )
        ranges[axis] = (start, end)
    return tuple(ranges)


def _parse_circle_mask(mask, volume_shape, training_patch_size):
    """Validate and normalize the optional 'training_circle_mask' entry.

    Accepts a dict with a required 'radius' (float) and optional 'center_y'
    and 'center_x' (floats; default to the geometric center of the y and x
    axes of the (post-crop) volume). Coordinates are in post-crop voxel
    space; the mask is a 2D circle in the y-x plane extended through every
    z slice (a cylinder), matching the geometry of a typical CT
    reconstruction whose valid region is the inscribed circle of each slice.

    Returns either None (no mask) or a tuple (center_y, center_x, radius)
    of floats.
    """
    if mask is None:
        return None
    if not isinstance(mask, dict):
        raise ValueError(
            f"'training_circle_mask' must be a dict (got {type(mask).__name__})"
        )
    allowed = {"radius", "center_y", "center_x"}
    extra = set(mask.keys()) - allowed
    if extra:
        raise ValueError(
            f"training_circle_mask has unknown keys: {sorted(extra)} "
            f"(allowed: {sorted(allowed)})"
        )
    if "radius" not in mask:
        raise ValueError("training_circle_mask must include 'radius'")

    H, W = volume_shape[1], volume_shape[2]
    raw_radius = mask["radius"]
    if isinstance(raw_radius, str):
        if raw_radius != "auto":
            raise ValueError(
                f"training_circle_mask['radius'] must be a positive number or "
                f"the string 'auto' (got {raw_radius!r})"
            )
        # 'auto' = inscribed circle of the post-crop slice: half of the
        # smaller of the y and x extents. Matches the typical CT geometry
        # where the valid signal lives inside the inscribed circle of each
        # square reconstructed slice.
        radius = min(H, W) / 2.0
    else:
        radius = float(raw_radius)
    if radius <= 0:
        raise ValueError(f"training_circle_mask['radius'] must be > 0 (got {radius})")

    center_y = float(mask.get("center_y", (H - 1) / 2.0))
    center_x = float(mask.get("center_x", (W - 1) / 2.0))
    if not (0 <= center_y < H):
        raise ValueError(
            f"training_circle_mask['center_y']={center_y} must be in [0, {H})"
        )
    if not (0 <= center_x < W):
        raise ValueError(
            f"training_circle_mask['center_x']={center_x} must be in [0, {W})"
        )

    # Smallest radius that can hold at least one patch: a patch centered on
    # the circle center has its farthest corner at distance
    # sqrt((ps_y - 1)^2 + (ps_x - 1)^2) / 2 from the center.
    ps_y, ps_x = training_patch_size[1], training_patch_size[2]
    min_radius = ((ps_y - 1) ** 2 + (ps_x - 1) ** 2) ** 0.5 / 2.0
    if radius < min_radius:
        raise ValueError(
            f"training_circle_mask['radius']={radius:.2f} is too small to contain any "
            f"{ps_y}x{ps_x} training patch (minimum feasible radius: {min_radius:.2f})"
        )

    return (center_y, center_x, radius)


class N2IDataset(Dataset):
    """
    Noise2Inverse dataset that preloads both volumes into memory for fast training.
    Both split volumes are loaded during initialization for efficient patch access.
    """

    def __init__(
        self, dataset_name, training_patch_size, nb_patches, normalization=True
    ):

        # Load dataset metadata
        with open(dataset_name, "r") as f:
            dataset_info = json.load(f)

        split1_path = dataset_info["split1_volume_file"]
        split2_path = dataset_info["split2_volume_file"]

        # Preload both volumes into memory
        logging.info("Loading training volumes into memory...")
        self.split1_volume = tifffile.imread(split1_path).astype(np.float32)
        self.split2_volume = tifffile.imread(split2_path).astype(np.float32)

        if self.split1_volume.shape != self.split2_volume.shape:
            raise ValueError(
                f"split1 and split2 must have identical shapes, got "
                f"{self.split1_volume.shape} vs {self.split2_volume.shape}"
            )

        # Optional 2D circular mask in the y-x plane (extended through every
        # z slice as a cylinder). The natural shape for CT reconstructions
        # whose valid signal lives inside the inscribed circle of each slice.
        # Coordinates are in ORIGINAL (pre-crop) voxel space because the mask
        # describes the geometry of the scan itself -- independent of how the
        # user chooses to crop. 'auto' radius likewise uses the original
        # slice dimensions, not the cropped ones. When training_crop is also
        # set, the crop is applied first and the circle center is then
        # translated to the cropped origin (radius unchanged); patch sampling
        # enforces both constraints simultaneously.
        original_shape = self.split1_volume.shape
        self.circle_mask = _parse_circle_mask(
            dataset_info.get("training_circle_mask"),
            original_shape,
            training_patch_size,
        )

        # Optional ROI cropping. When the JSON config defines a "training_crop"
        # entry, both training volumes are sliced to that bounding box and
        # everything downstream (normalization stats, patch sampling, diagnostics)
        # sees only the cropped region. Inference is unaffected — it still runs
        # on whatever test volume the config points to.
        self.crop = _parse_crop(dataset_info.get("training_crop"), original_shape)
        if self.crop is not None:
            # Crop tuple is in axis order (axis-0, axis-1, axis-2); under the
            # standard imaging convention these are (z, y, x).
            (z0, z1), (y0, y1), (x0, x1) = self.crop
            self.split1_volume = self.split1_volume[z0:z1, y0:y1, x0:x1].copy()
            self.split2_volume = self.split2_volume[z0:z1, y0:y1, x0:x1].copy()
            logging.info(
                f"Applied training crop: z=[{z0},{z1}), y=[{y0},{y1}), x=[{x0},{x1}) "
                f"-> shape {self.split1_volume.shape}"
            )

        self.volume_shape = self.split1_volume.shape

        if any(vs < ps for vs, ps in zip(self.volume_shape, training_patch_size)):
            raise ValueError(
                f"Training volume shape {self.volume_shape} is smaller than the "
                f"training patch size {tuple(training_patch_size)} in at least "
                f"one dimension. Crop the volume less aggressively (every "
                f"dimension must be >= {training_patch_size[0]} voxels) or "
                f"reduce TRAIN_PATCH_SIZE in train.py."
            )

        # The circle is fixed in original (acquisition) coordinates. When a
        # crop has also been applied, the runtime patch-acceptance test
        # addresses cropped offsets, so the circle center is translated
        # internally by the crop offset for that bookkeeping. The physical
        # geometry of the circle is unchanged. self.circle_mask retains the
        # original-coord values for params.json traceability; the _circle_*
        # attributes hold the runtime (cropped-coord) values.
        if self.circle_mask is not None:
            cy_orig, cx_orig, r = self.circle_mask
            if self.crop is not None:
                _, (y0, _), (x0, _) = self.crop
                cy_run, cx_run = cy_orig - y0, cx_orig - x0
            else:
                cy_run, cx_run = cy_orig, cx_orig
            logging.info(
                f"Applied training circle mask: center=({cy_orig:.1f}, {cx_orig:.1f}), "
                f"radius={r:.1f} (in y-x plane, extended through z)"
            )
            self._circle_cy = cy_run
            self._circle_cx = cx_run
            self._circle_r2 = r * r
        else:
            self._circle_cy = self._circle_cx = self._circle_r2 = None

        # Diagnostic: report how heavily the volume is sampled per epoch.
        # Heavy reuse (high ratio) on tiny volumes or vanishing coverage on huge
        # volumes are both fine for N2N training but worth being aware of.
        unique_positions = 1
        for vs, ps in zip(self.volume_shape, training_patch_size):
            unique_positions *= vs - ps + 1
        logging.info(f"Volume size: {'×'.join(str(s) for s in self.volume_shape)}")
        logging.info(
            f"Unique patch start positions: {unique_positions:.2e} "
            f"(patches/epoch: {nb_patches}, coverage per epoch: "
            f"{nb_patches / unique_positions:.2e})"
        )

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
            logging.info("Computing normalization statistics...")
            if self.circle_mask is not None:
                # Exclude out-of-circle voxels from the stats so the corners
                # (zero/artifact in CT) don't skew the mean/std estimate.
                H, W = self.volume_shape[1], self.volume_shape[2]
                y_grid, x_grid = np.ogrid[:H, :W]
                yx_mask = (
                    (y_grid - self._circle_cy) ** 2 + (x_grid - self._circle_cx) ** 2
                ) <= self._circle_r2
                in_circle_1 = self.split1_volume[:, yx_mask]
                in_circle_2 = self.split2_volume[:, yx_mask]
                combined_data = np.concatenate(
                    [in_circle_1.ravel(), in_circle_2.ravel()]
                )
            else:
                combined_data = np.concatenate(
                    [self.split1_volume.flatten(), self.split2_volume.flatten()]
                )
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
        """Extract a patch from preloaded volume.

        start_coords is in axis order (axis-0, axis-1, axis-2), which under the
        standard imaging convention corresponds to (z, y, x).
        """
        z, y, x = start_coords
        patch = volume[
            z : z + self.patch_size[0],
            y : y + self.patch_size[1],
            x : x + self.patch_size[2],
        ].clone()
        return patch.unsqueeze(0)  # Add channel dimension

    def _patch_inside_circle(self, start_y, start_x):
        """Return True if all four xy corners of a patch starting at
        (start_y, start_x) lie inside the configured circle mask."""
        if self.circle_mask is None:
            return True
        cy, cx, r2 = self._circle_cy, self._circle_cx, self._circle_r2
        ps_y, ps_x = self.patch_size[1], self.patch_size[2]
        y0, y1 = start_y, start_y + ps_y - 1
        x0, x1 = start_x, start_x + ps_x - 1
        return (
            (y0 - cy) ** 2 + (x0 - cx) ** 2 <= r2
            and (y0 - cy) ** 2 + (x1 - cx) ** 2 <= r2
            and (y1 - cy) ** 2 + (x0 - cx) ** 2 <= r2
            and (y1 - cy) ** 2 + (x1 - cx) ** 2 <= r2
        )

    def __len__(self):
        return self.nb_patches

    def __getitem__(self, idx):
        # Generate random patch coordinates within valid sampling region.
        # Naming follows standard imaging convention: z = axis 0 (slice),
        # y = axis 1 (row), x = axis 2 (column). When a circle mask is
        # configured, fall back to rejection sampling: pick a candidate,
        # accept iff the entire patch fits inside the circle.
        max_z = self.sampling_shape[0] - self.patch_size[0]
        max_y = self.sampling_shape[1] - self.patch_size[1]
        max_x = self.sampling_shape[2] - self.patch_size[2]

        MAX_ATTEMPTS = 200
        for _ in range(MAX_ATTEMPTS):
            start_z = np.random.randint(0, max_z + 1) + self.sampling_offset[0]
            start_y = np.random.randint(0, max_y + 1) + self.sampling_offset[1]
            start_x = np.random.randint(0, max_x + 1) + self.sampling_offset[2]
            if self._patch_inside_circle(start_y, start_x):
                break
        else:
            raise RuntimeError(
                f"Could not find a valid patch position inside the circle mask "
                f"after {MAX_ATTEMPTS} attempts. The circle may be too small "
                f"relative to the patch size, or the circle center may be too "
                f"close to the volume edge."
            )

        # Load patches from preloaded volumes (returns float32 torch tensors)
        patch1 = self._load_patch(self.split1_volume, (start_z, start_y, start_x))
        patch2 = self._load_patch(self.split2_volume, (start_z, start_y, start_x))

        # Apply normalization if needed
        if self.normalization:
            patch1 = (patch1 - self.mean) / (self.std + 1e-7)
            patch2 = (patch2 - self.mean) / (self.std + 1e-7)

        # Apply cube symmetry transform (same random orientation to both
        # patches so the noisy pair stays registered). Returns plain tensors;
        # the default DataLoader collate stacks them into a (B, 1, D, H, W)
        # batch with no per-sample object-construction overhead.
        patch1, patch2 = self.transform([patch1, patch2])

        return {
            "split1_volume": patch1,
            "split2_volume": patch2,
        }


def _worker_init_fn(worker_id):
    """Seed numpy independently in each DataLoader worker.

    Without this, every worker inherits the same numpy RNG state and the random
    patch coordinates in __getitem__ end up correlated across workers. PyTorch
    already seeds its own and Python's RNGs per worker; we only need to forward
    that to numpy.
    """
    np.random.seed(torch.initial_seed() % 2**32)


def _cuda_device_arg(s):
    """argparse type for --cuda_device: accepts 'auto' or a non-negative int."""
    if s == "auto":
        return s
    try:
        value = int(s)
    except ValueError:
        import argparse

        raise argparse.ArgumentTypeError(
            f"--cuda_device must be 'auto' or an integer, got {s!r}"
        )
    if value < 0:
        import argparse

        raise argparse.ArgumentTypeError(f"--cuda_device must be >= 0, got {value}")
    return value


def _select_cuda_device(arg):
    """Resolve --cuda_device to a concrete GPU index.

    Returns a tuple (index, info_string). For an explicit integer the info
    string is empty. For successful 'auto' selection the info string
    summarizes the choice so the caller can fold it into a single line.
    Warnings are printed directly here and the index falls back to 0.

    Free memory is queried via torch.cuda.mem_get_info when available
    (PyTorch 1.11+), which respects CUDA_VISIBLE_DEVICES and only reports
    GPUs PyTorch can actually use. We fall back to nvidia-smi only when
    that API is missing, and refuse to trust nvidia-smi if its device
    count disagrees with what PyTorch sees -- because nvidia-smi reports
    physical GPUs, not the subset visible to PyTorch.
    """
    if isinstance(arg, int):
        return arg, ""
    if arg != "auto":
        return int(arg), ""

    if not torch.cuda.is_available():
        return 0, ""

    n_visible = torch.cuda.device_count()
    if n_visible == 0:
        return 0, ""

    # Preferred path: PyTorch's own per-device free-memory query.
    try:
        free = []
        for i in range(n_visible):
            free_bytes, _ = torch.cuda.mem_get_info(i)
            free.append(free_bytes // (1024**2))
        idx = max(range(len(free)), key=lambda i: free[i])
        return idx, f"auto-selected, {free[idx]} MiB free; free per GPU: {free}"
    except AttributeError:
        pass  # PyTorch < 1.11: mem_get_info not available, fall through

    # Fallback: nvidia-smi. Only trust it if its device count matches
    # PyTorch's view; otherwise CUDA_VISIBLE_DEVICES (or similar) is
    # restricting PyTorch and the indices would be mis-mapped.
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        free = [int(line.strip()) for line in out.strip().splitlines() if line.strip()]
        if len(free) != n_visible:
            logging.warning(
                f"    --cuda_device auto: nvidia-smi reports {len(free)} GPUs but "
                f"PyTorch sees {n_visible} (likely CUDA_VISIBLE_DEVICES is set); "
                f"falling back to cuda:0 for safety."
            )
            return 0, ""
        idx = max(range(len(free)), key=lambda i: free[i])
        return idx, f"auto-selected, {free[idx]} MiB free; free per GPU: {free}"
    except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e:
        logging.warning(
            f"    --cuda_device auto: nvidia-smi failed ({e}); falling back to 0"
        )
        return 0, ""


def save_model(model, optimizer, epoch, save_path):
    """Save model checkpoint with PyTorch's built-in compression.

    Saves the underlying (uncompiled) module's state_dict so checkpoints are
    identical whether or not torch.compile wrapped the model. A compiled module
    (OptimizedModule) otherwise prefixes every key with '_orig_mod.', which the
    inference checkpoint loader does not expect.
    """
    base_model = getattr(model, "_orig_mod", model)
    state = {
        "epoch": int(epoch),
        "state_dict": base_model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    # Use PyTorch's compression (recommended)
    torch.save(state, save_path, _use_new_zipfile_serialization=True)


def train_model(
    dl,
    model,
    loss_func,
    optimizer,
    checkpoint_dir,
    loaded_checkpoint_path,
    nb_train_epoch,
    device,
    use_amp,
    keep_only_last,
):
    """Train the model with logic similar to train_old.py."""

    start_epoch_nb = 0

    # Load checkpoint if specified
    if loaded_checkpoint_path is not None:
        logging.info("Loading weights...")
        state = _safe_torch_load(
            loaded_checkpoint_path, map_location=torch.device(device)
        )
        # Load into the underlying module so clean (unprefixed) checkpoint keys
        # work whether or not the model has been wrapped by torch.compile.
        getattr(model, "_orig_mod", model).load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch_nb = state["epoch"] + 1

    # GradScaler prevents fp16 gradient underflow during backward. When
    # use_amp=False it's a no-op (passes through scale/step/update calls).
    scaler = _make_gradscaler(use_amp)

    # Per-epoch loss log, appended live so the curve survives interruptions
    # and is recoverable for plateau-detection / plotting after training.
    # On resume (start_epoch_nb > 0) we append to the existing file; on a
    # fresh run we (re)write the header.
    loss_csv_path = checkpoint_dir / "training_loss.csv"
    write_header = (
        start_epoch_nb == 0
        or not loss_csv_path.exists()
        or loss_csv_path.stat().st_size == 0
    )
    loss_csv = open(loss_csv_path, "w" if write_header else "a")
    if write_header:
        loss_csv.write("epoch,mean_loss\n")
        loss_csv.flush()

    # Training loop. Track the final epoch's loss for the end-of-run summary.
    first_epoch_completed = False
    final_loss = float("nan")
    for epoch in range(start_epoch_nb, nb_train_epoch):
        # Accumulate the loss on-device and sync once per epoch, rather than
        # calling .item() every iteration (each .item() forces a GPU->CPU
        # synchronization that serializes the training step).
        epoch_loss_sum = torch.zeros((), device=device)

        for batch in tqdm(dl, desc=f"Epoch {epoch+1}/{nb_train_epoch}"):
            # Per-sample: with 50% probability swap which noisy copy is input vs. target
            data1 = batch["split1_volume"].to(device, non_blocking=True)
            data2 = batch["split2_volume"].to(device, non_blocking=True)
            swap = (torch.rand(data1.shape[0], device=device) < 0.5).view(
                -1, 1, 1, 1, 1
            )
            input = torch.where(swap, data2, data1)
            target = torch.where(swap, data1, data2)

            # Proceed to a training step
            optimizer.zero_grad()

            # Forward + loss inside autocast; the ×1000 scaling composes with
            # GradScaler's dynamic scaling — both survive the backward pass.
            with _amp_autocast(use_amp):
                pred = model(input)
                # Use MSE loss. Loss is scaled by 1000 to counteract Adam's eps=1e-8
                # damping the update when gradients are small (z-score-normalized
                # inputs + MSE produce tiny gradient magnitudes). Equivalent to
                # using eps=1e-11; the scaling form is kept for numerical safety.
                total_loss = loss_func(pred, target)
                loss_val = total_loss * 1000

            scaler.scale(loss_val).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss_sum += loss_val.detach()

        # One device sync per epoch to read back the mean loss.
        epoch_loss = (epoch_loss_sum / len(dl)).item()
        logging.info(f"Mean loss value of the epoch : {epoch_loss:.4f}")
        loss_csv.write(f"{epoch},{epoch_loss:.6f}\n")
        loss_csv.flush()
        final_loss = epoch_loss

        # Show memory monitoring only after first epoch. We deliberately do
        # NOT reset the CUDA peak tracker -- letting it accumulate across all
        # epochs lets the end-of-run summary report the true overall peak.
        if not first_epoch_completed and torch.cuda.is_available():
            max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
            logging.info(f"GPU memory peak: {max_memory_allocated:.2f} GB")

        if not first_epoch_completed:
            process = psutil.Process()
            first_epoch_ram_gb = process.memory_info().rss / 1024**3
            logging.info(f"RAM memory peak: {first_epoch_ram_gb:.2f} GB")

        first_epoch_completed = True

        # Save checkpoint at each epoch
        logging.info("Saving checkpoint for epoch n°{}...".format(epoch))
        save_model(
            model, optimizer, epoch, checkpoint_dir / f"weights_epoch_{epoch:03d}.torch"
        )

        # If keep_only_last is set, delete the previous epoch's checkpoint
        # after the new one is safely on disk. Resume-from-checkpoint still
        # works because the loaded_checkpoint_path argument is just a file
        # path -- if the user re-runs and the file no longer exists, that's
        # the same as any other missing-file error.
        if keep_only_last:
            prev_ckpt = checkpoint_dir / f"weights_epoch_{epoch - 1:03d}.torch"
            if prev_ckpt.exists():
                prev_ckpt.unlink()

    loss_csv.close()

    # Return end-of-run stats for the caller's summary line. peak_vram_gb
    # is the true overall peak across all epochs (we removed the per-epoch
    # reset of the CUDA tracker above); peak_ram_gb is current RSS which
    # approximates the peak well for this pipeline (volumes are preloaded
    # at startup, no growing structures during training).
    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    peak_ram_gb = psutil.Process().memory_info().rss / 1024**3
    return {
        "final_loss": final_loss,
        "peak_vram_gb": peak_vram_gb,
        "peak_ram_gb": peak_ram_gb,
        "epochs_completed": max(0, nb_train_epoch - start_epoch_nb),
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main(params):
    """Main training function."""
    # Configure timestamped logging (mirrors inference.py).
    setup_logging()

    # Capture wall-clock start so the end-of-run summary can report total time.
    start_time = time.time()

    with open(params.input_json, "r") as f:
        dataset_info = json.load(f)

    # Get checkpoint directory from JSON and create it if it doesn't exist
    checkpoint_dir = Path(dataset_info["checkpoint_path"])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Resolve --cuda_device ('auto' picks the GPU with the most free memory).
    # After this, params.cuda_device is always a concrete int.
    params.cuda_device, cuda_info = _select_cuda_device(params.cuda_device)

    # Determine device + resolve every GPU-capability-gated decision (fp16,
    # torch.compile) in one block so adjacent log lines tell the user exactly
    # what's actually enabled on this hardware. Mirrors inference.py's layout.
    logging.info("Configuring device and precision...")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{params.cuda_device}")
        msg = f"    Using GPU device: cuda:{params.cuda_device}"
        if cuda_info:
            msg += f" ({cuda_info})"
        logging.info(msg)
        compute_major, compute_minor = torch.cuda.get_device_capability(device)
    else:
        device = torch.device("cpu")
        logging.info("    CUDA not available, using CPU")
        compute_major, compute_minor = 0, 0

    # fp16 mixed-precision training requires CUDA and a tensor-core-capable
    # GPU (compute capability >= 7.0: Volta/Turing/Ampere/Ada/Hopper). On
    # older cards (e.g. Pascal GTX 10-series) fp16 throughput is much lower
    # than fp32, so autocast would slow training down.
    use_amp = False
    if not params.no_half and device.type == "cuda" and compute_major >= 7:
        use_amp = True
        logging.info(
            f"    Mixed precision (fp16): enabled "
            f"(compute capability {compute_major}.{compute_minor})"
        )
    else:
        if params.no_half:
            reason = "--no_half"
        elif device.type != "cuda":
            reason = "CPU"
        else:
            reason = (
                f"compute capability {compute_major}.{compute_minor} lacks "
                f"tensor cores; fp16 would run slower than fp32"
            )
        logging.info(f"    Mixed precision (fp16): disabled ({reason})")

    # torch.compile is opt-in (--compile). It needs PyTorch 2.0+ + CUDA +
    # compute capability >= 7.0 (Triton, the inductor backend, refuses to
    # compile for compute < 7.0) and, on Windows, an MSVC toolchain + Windows
    # SDK on PATH to build the generated kernels -- which not every machine has.
    # Defaulting it off keeps the common path warning-free; users who know their
    # toolchain is set up pass --compile for the ~1.2-1.5x speedup. The decision
    # and its log line are resolved here; the wrap itself is applied later
    # (after params.json is written and the optimizer is created).
    if not params.compile:
        use_compile = False
        logging.info("    torch.compile: disabled (default; pass --compile to enable)")
    elif not hasattr(torch, "compile"):
        use_compile = False
        logging.info("    torch.compile: disabled (PyTorch < 2.0)")
    elif device.type != "cuda" or compute_major < 7:
        use_compile = False
        logging.info(
            f"    torch.compile: disabled "
            f"(compute capability {compute_major}.x lacks Triton backend support; requires >= 7.0)"
        )
    else:
        use_compile = True
        logging.info(
            "    torch.compile: enabled (--compile; will compile on first training step)"
        )

    # Initialize the model to be trained
    model = create_model(
        device=params.cuda_device if torch.cuda.is_available() else "cpu",
        norm_division_factor=getattr(params, "norm_division_factor", 1),
        num_res_units=getattr(params, "num_res_units", 0),
        unet_depth=getattr(params, "unet_depth", 4),
        unet_stride=getattr(params, "unet_stride", 2),
    )

    # Create the memory-efficient data loading pipeline
    logging.info("Setting up data loader...")
    train_dataset = N2IDataset(
        params.input_json, TRAIN_PATCH_SIZE, NB_PATCH_PER_EPOCH, normalization=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=params.num_workers,
        persistent_workers=params.num_workers > 0,
        worker_init_fn=_worker_init_fn if params.num_workers > 0 else None,
        # Page-locked host buffers let the .to(device, non_blocking=True) copy
        # in the training loop overlap with compute. Only meaningful with CUDA.
        pin_memory=torch.cuda.is_available(),
    )

    # Create loss function and optimizer. MSE is the N2N estimator for zero-mean
    # noise: it recovers the conditional mean of the clean signal and converges
    # to a spatially consistent solution. (L1 recovers the conditional median,
    # which is biased on asymmetric noise and produced patch-grid block
    # artifacts on this data, so it is not offered.)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=WEIGHT_DECAY, lr=LEARNING_RATE
    )

    # Save the training parameters including normalization statistics
    params_dict = dict(vars(params))
    params_dict.update(
        {
            # Data parameters:
            "normalization_mean": float(train_dataset.mean),
            "normalization_std": float(train_dataset.std),
            # Training parameters:
            "loaded_checkpoint_path": params.loaded_checkpoint_path,
            "loss_function": loss_func.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "train_patch_size": TRAIN_PATCH_SIZE,
            "nb_patch_per_epoch": NB_PATCH_PER_EPOCH,
            "nb_train_epoch": params.nb_train_epoch,
            "training_cuda_device": params.cuda_device,
            "training_batch_size": params.batch_size,
            "training_mixed_precision": use_amp,
            "training_crop": train_dataset.crop,
            "training_circle_mask": train_dataset.circle_mask,
            # Image-level residual learning: the trained model computes
            # output = input + unet(input). Recorded in params.json so any
            # downstream tooling that needs to reason about checkpoint type
            # can detect it without inspecting state_dict keys.
            "residual_learning": True,
            # UNet model architecture parameters (read from the inner unet,
            # which is wrapped by ResidualUNet on this branch).
            "unet_in_channels": model.unet.in_channels,
            "unet_out_channels": model.unet.out_channels,
            "unet_channels": model.unet.channels,
            "unet_strides": model.unet.strides,
            "unet_kernel_size": model.unet.kernel_size,
            "unet_up_kernel_size": model.unet.up_kernel_size,
            "unet_num_res_units": model.unet.num_res_units,
            "unet_act": model.unet.act,
            "unet_norm": model.unet.norm,
            "unet_dropout": model.unet.dropout,
        }
    )
    with open(checkpoint_dir / "params.json", "w") as par_file:
        json.dump(params_dict, par_file)
    logging.info(
        f"Saved training parameters with normalization statistics: mean={train_dataset.mean:.6f}, std={train_dataset.std:.6f}"
    )

    # Apply the already-resolved torch.compile decision (use_compile was set and
    # logged in the device/precision block above). The wrap is done here, after
    # params.json is written (which reads model.unet.*) and after the optimizer
    # is created (the compiled module shares the same parameter tensors, so the
    # optimizer keeps updating the right weights). Training uses a fixed 96^3
    # patch shape, so the model compiles once on the first step with no later
    # recompiles. Checkpoints stay compile-agnostic because save_model unwraps
    # _orig_mod. The wrap can still raise on edge cases; fall back to eager mode.
    if use_compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            logging.warning(
                f"    torch.compile failed at wrap time; using eager mode: {e}"
            )

    # Train model
    stats = train_model(
        train_loader,
        model,
        loss_func,
        optimizer,
        checkpoint_dir,
        params.loaded_checkpoint_path,
        params.nb_train_epoch,
        device,
        use_amp,
        params.keep_only_last,
    )

    # End-of-run summary: total wall-clock, epochs completed, last epoch's
    # mean loss, peak memory across the full run, and where to find the
    # checkpoints. Collapses what was previously scattered status into one
    # scannable record.
    elapsed = str(timedelta(seconds=int(time.time() - start_time)))
    logging.info(
        f"Training complete: {stats['epochs_completed']} epoch(s) in {elapsed}, "
        f"final mean loss {stats['final_loss']:.4f}, "
        f"peak VRAM {stats['peak_vram_gb']:.2f} GB, "
        f"peak RAM {stats['peak_ram_gb']:.2f} GB. "
        f"Checkpoints in {checkpoint_dir}"
    )


if __name__ == "__main__":

    parse = ArgumentParser(
        description="Train a model with Noise2Inverse, using 3d convolutions"
    )
    parse.add_argument(
        "input_json",
        help="Path to JSON file containing dataset information and processing paths",
    )
    parse.add_argument(
        "--loaded_checkpoint_path",
        default=None,
        help="If set, load the checkpoint located at the provided path",
    )
    parse.add_argument(
        "--nb_train_epoch", default=50, type=int, help="The number of training epochs"
    )
    parse.add_argument(
        "--batch_size", default=16, type=int, help="The number of patch per batch"
    )
    parse.add_argument(
        "--cuda_device",
        default="auto",
        type=_cuda_device_arg,
        help="CUDA device to use: a non-negative integer or 'auto' (picks the GPU with the most free memory via nvidia-smi). Default: auto.",
    )
    parse.add_argument(
        "--num_res_units",
        default=0,
        type=int,
        help="Number of residual conv units per level inside the MONAI U-Net (default: 0, a plain conv block per level). This is MONAI's intra-block residual learning, distinct from the image-level residual wrapper. Values of 1 or 2 add deeper per-level blocks with internal skip connections, which can improve denoising fidelity / edge sharpness at the cost of more compute, memory, and parameters. Recorded in params.json so inference reconstructs the matching architecture.",
    )
    parse.add_argument(
        "--unet_depth",
        default=4,
        type=int,
        help="Number of U-Net levels (default: 4). Channels start at 56 and double per level, so depth 4 = (56,112,224,448) and depth 3 = (56,112,224); there are unet_depth-1 downsampling stages. Fewer levels keep detail at a finer resolution (less of the smoothing caused by the coarse bottleneck) but shrink the receptive field; more levels widen context at the cost of more downsampling. Must be >= 2. Recorded in params.json so inference reconstructs the matching architecture.",
    )
    parse.add_argument(
        "--unet_stride",
        default=2,
        type=int,
        help="Downsampling factor applied uniformly at every U-Net stage (default: 2). Set to 1 for a no-downsampling, full-resolution network: the sharpest option since no spatial information is lost, but dramatically more memory- and compute-hungry. Must be >= 1. Recorded in params.json so inference reconstructs the matching architecture.",
    )
    parse.add_argument(
        "--norm_division_factor",
        default=56,
        type=int,
        help="Division factor for group normalization. 56 (default) = layer norm (num_groups=1), the best pairing with residual learning; 1 = instance norm (num_groups=56); intermediate divisors of 56 give true group norm. Valid values: 1, 2, 4, 7, 8, 14, 28, 56.",
    )
    parse.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of DataLoader worker processes (default: 4; use 0 on very low-RAM systems)",
    )
    parse.add_argument(
        "--no_half",
        action="store_true",
        help="Disable fp16 mixed precision training (default: enabled on tensor-core GPUs only, i.e. compute capability >= 7.0)",
    )
    parse.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (default: disabled). Gives a ~1.2-1.5x training speedup after a one-time compilation on the first step, but requires PyTorch 2.0+, a GPU with compute capability >= 7.0, and -- on Windows -- an MSVC toolchain + Windows SDK on PATH to build the kernels. Only enable it if your toolchain is set up; otherwise Triton prints repeated 'Failed to find MSVC' warnings and falls back to eager mode.",
    )
    parse.add_argument(
        "--keep_only_last",
        action="store_true",
        help="Keep only the most recent epoch's checkpoint on disk; delete previous ones after each save (default: keep every epoch)",
    )

    main(parse.parse_args())
