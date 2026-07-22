"""Model definition plus the environment setup shared by train.py and inference.py.

Everything below the model factory is infrastructure that both entry points
need identically: backend flags, logging format, z-score normalization, the
AMP / torch.load version shims, CUDA device selection, and the
capability-gated fp16 / torch.compile decisions. Keeping a single copy here
means the two scripts cannot drift apart -- previously each maintained its own
copy and the normalization epsilon had already diverged between them.
"""

from monai.networks.nets import UNet
import torch
import logging
import inspect as _inspect


# ============================================================================
# BACKEND CONFIGURATION
# ============================================================================

# Enable cuDNN autotuning and TF32. Training and inference both run fixed-shape
# convolutions (96^3 patches / fixed sliding-window roi), so cuDNN's benchmark
# mode pays its one-time autotune cost back immediately by selecting the fastest
# conv algorithm. TF32 (Ampere+; a no-op on older cards) speeds up the fp32
# fallback paths with precision that is irrelevant for denoising.
#
# Set at import time: both entry points import this module before doing any
# CUDA work, so the flags apply exactly as early as they did when each script
# set them itself.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# NORMALIZATION
# ============================================================================

# Guards against division by zero on a (near-)constant volume. Training and
# inference must use the identical expression or the network sees a different
# input distribution at inference than it was trained on.
NORM_EPS = 1e-7


def zscore_normalize(x, mean, std):
    """Z-score normalize an array or tensor. Shared by training and inference."""
    return (x - mean) / (std + NORM_EPS)


# ============================================================================
# LOGGING
# ============================================================================


def setup_logging() -> None:
    """Configure timestamped logging.

    Training and inference share one format so a back-to-back train/infer
    session produces output that can be cross-correlated by wall clock.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================================
# VERSION COMPATIBILITY SHIMS
# ============================================================================

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
_TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY = (
    "weights_only" in _inspect.signature(torch.load).parameters
)


def _safe_torch_load(path, map_location):
    if _TORCH_LOAD_SUPPORTS_WEIGHTS_ONLY:
        return torch.load(path, map_location=map_location, weights_only=True)
    return torch.load(path, map_location=map_location)


# ============================================================================
# DEVICE SELECTION
# ============================================================================


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
    summarizes the choice so the caller can fold it into a single log line.
    Warnings are logged directly here and the index falls back to 0.

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


def resolve_device(cuda_device_arg):
    """Resolve --cuda_device into a concrete device and log the choice.

    Returns (device, index, compute_major, compute_minor). ``device`` is the
    torch.device every tensor in the run must be placed on -- callers should
    use it rather than bare ``.cuda()``, which ignores the selected index and
    would strand tensors on cuda:0 whenever 'auto' picks a different GPU.
    """
    index, info = _select_cuda_device(cuda_device_arg)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{index}")
        msg = f"    Using GPU device: cuda:{index}"
        if info:
            msg += f" ({info})"
        logging.info(msg)
        compute_major, compute_minor = torch.cuda.get_device_capability(device)
    else:
        device = torch.device("cpu")
        index = 0
        logging.info("    CUDA not available, using CPU")
        compute_major, compute_minor = 0, 0

    return device, index, compute_major, compute_minor


# ============================================================================
# CAPABILITY-GATED ACCELERATION
# ============================================================================


def resolve_mixed_precision(want_half, device, compute_major, compute_minor):
    """Decide whether fp16 autocast is used, and log why.

    fp16 mixed precision requires CUDA and a tensor-core-capable GPU (compute
    capability >= 7.0: Volta/Turing/Ampere/Ada/Hopper). On older cards (e.g.
    Pascal GTX 10-series) fp16 throughput is much lower than fp32, so autocast
    would slow the run down rather than speed it up.
    """
    if want_half and device.type == "cuda" and compute_major >= 7:
        logging.info(
            f"    Mixed precision (fp16): enabled "
            f"(compute capability {compute_major}.{compute_minor})"
        )
        return True

    if not want_half:
        reason = "--no_half"
    elif device.type != "cuda":
        reason = "CPU"
    else:
        reason = (
            f"compute capability {compute_major}.{compute_minor} lacks "
            f"tensor cores; fp16 would run slower than fp32"
        )
    logging.info(f"    Mixed precision (fp16): disabled ({reason})")
    return False


def resolve_compile(want_compile, device, compute_major, first_call_desc):
    """Decide whether torch.compile is used, and log why.

    torch.compile is opt-in (--compile). It needs PyTorch 2.0+ + CUDA +
    compute capability >= 7.0 (Triton, the inductor backend, refuses to compile
    for compute < 7.0) and, on Windows, an MSVC toolchain + Windows SDK on PATH
    to build the generated kernels -- which not every machine has. Defaulting it
    off keeps the common path warning-free.

    ``first_call_desc`` describes when compilation happens (e.g. "first
    training step") so the log line is accurate for either entry point.
    """
    if not want_compile:
        logging.info("    torch.compile: disabled (default; pass --compile to enable)")
        return False
    if not hasattr(torch, "compile"):
        logging.info("    torch.compile: disabled (PyTorch < 2.0)")
        return False
    if device.type != "cuda" or compute_major < 7:
        logging.info(
            f"    torch.compile: disabled "
            f"(compute capability {compute_major}.x lacks Triton backend support; requires >= 7.0)"
        )
        return False
    logging.info(
        f"    torch.compile: enabled (--compile; will compile on {first_call_desc})"
    )
    return True


def maybe_compile(model, use_compile):
    """Wrap the model with torch.compile, falling back to eager mode on error.

    The wrap can still raise on edge cases unrelated to compute capability
    (missing toolchain, unsupported backend), which must not abort the run.
    """
    if not use_compile:
        return model
    try:
        return torch.compile(model)
    except Exception as e:
        logging.warning(f"    torch.compile failed at wrap time; using eager mode: {e}")
        return model


# ============================================================================
# MODEL
# ============================================================================


class ResidualUNet(torch.nn.Module):
    """MONAI UNet wrapped to operate in image-level residual learning mode.

    The forward pass computes ``output = input + unet(input)``: the network
    learns the per-voxel correction added to the input rather than
    reconstructing the entire signal from scratch. For voxels where the
    network predicts ~0 residual (well-conditioned regions, low local noise),
    the output equals the input exactly -- preserving its intensity, and so
    its mean, by construction.

    For Noise2Noise training this addresses the small systematic mean drift
    that can otherwise appear in the denoised output: any accumulated bias
    from activation asymmetries, GroupNorm parameters, or finite-capacity
    estimation is bounded by what the network actually predicts, not what
    propagates through the whole forward pass.

    Theoretically equivalent (in expectation) to the underlying N2N
    objective: the network learns the conditional mean of (target - input),
    which is the conditional mean of (clean - input) plus a zero-mean noise
    term that integrates out.

    The identity path cuts both ways, which is why direct prediction remains
    selectable (``create_model(residual=False)``): where the network predicts
    ~0 correction the input passes through *including its noise*, so residual
    output can look grainier in flat regions than a direct prediction that is
    free to reshape the whole intensity distribution. Residual preserves
    quantitative values; direct sometimes looks better qualitatively.
    """

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        return x + self.unet(x)


def create_model(
    device=0,
    norm_division_factor=56,
    num_res_units=0,
    unet_depth=4,
    unet_stride=2,
    residual=True,
):
    """
    Create a 3D U-Net model with configurable normalization, in either
    image-level residual mode (output = input + unet(input)) or direct
    prediction mode (output = unet(input)).

    Args:
        device (int): The device to use (default: 0 for GPU, or 'cpu' for CPU)
        residual (bool): Prediction mode (default: True, i.e. residual).

                True wraps the U-Net so it learns the per-voxel *correction*
                applied to the input; see ResidualUNet for why this bounds the
                systematic mean drift that direct prediction can exhibit.

                False returns the bare U-Net, which estimates the denoised
                volume directly. This is the original formulation and remains
                available for comparison.

                The two modes produce different state_dict key namespaces
                (``unet.model.0...`` vs ``model.0...``), which is deliberate: a
                checkpoint loaded in the wrong mode fails with a key mismatch
                instead of silently running a correction-predicting network as
                a whole-volume predictor. The mode is recorded in params.json
                as ``residual_learning`` so inference reconstructs it.
        unet_depth (int): Number of U-Net levels, i.e. the length of the channel
                schedule (default: 4). Channels start at 56 and double per level,
                so depth 4 = (56, 112, 224, 448) and depth 3 = (56, 112, 224).
                There are ``unet_depth - 1`` downsampling stages. Fewer levels
                means the network processes detail at a finer resolution (less of
                the smoothing that comes from the coarse bottleneck) but has a
                smaller receptive field; more levels widen context at the cost of
                more downsampling. Must be >= 2. Recorded in params.json so
                inference reconstructs the matching architecture.
        unet_stride (int): Downsampling factor applied at every stage (default: 2,
                used uniformly for all ``unet_depth - 1`` stages). Set to 1 for a
                no-downsampling, full-resolution network (MSD-like): the sharpest
                option since no spatial information is lost, but dramatically more
                memory- and compute-hungry because every level keeps full-size
                feature maps. Must be >= 1. Recorded in params.json so inference
                reconstructs the matching architecture.
        num_res_units (int): Number of residual conv units per level inside the
                MONAI U-Net (default: 0, a plain conv block per level). This is
                MONAI's *intra-block* residual learning and is distinct from the
                image-level residual wrapper (ResidualUNet) selected by the
                ``residual`` argument; the two are independent and can be
                combined. Values of 1 or 2 add
                deeper per-level blocks with internal skip connections, which can
                improve denoising fidelity / edge sharpness at the cost of more
                compute, memory, and parameters. The value is recorded in
                params.json so inference reconstructs the matching architecture.
        norm_division_factor (int): Division factor for group normalization. The
                number of groups is calculated as: first_channels // norm_division_factor.
                Must be a positive divisor of the first channel count (56), so
                valid values are 1, 2, 4, 7, 8, 14, 28, 56. Default is 56, which
                means layer normalization (num_groups=1) -- empirically the best
                pairing with the residual learning wrapper, because it preserves
                inter-channel structure that helps signal/noise discrimination.
                (That comparison was run in residual mode; it has not been
                re-measured for direct prediction.) Setting it to 1 selects
                instance normalization (num_groups=56); intermediate divisors
                give true group normalization.
    Returns:
        torch.nn.Module: in residual mode, a ResidualUNet wrapping the MONAI
        UNet, with the inner network accessible as ``model.unet``; in direct
        mode, the MONAI UNet itself. Callers that need the inner network in
        both cases should use ``getattr(model, "unet", model)``.
    """
    if not isinstance(unet_depth, int) or unet_depth < 2:
        raise ValueError(f"unet_depth must be an integer >= 2, got {unet_depth}")
    if not isinstance(unet_stride, int) or unet_stride < 1:
        raise ValueError(f"unet_stride must be an integer >= 1, got {unet_stride}")

    first_channels = 56
    channels = tuple(first_channels * (2**i) for i in range(unet_depth))
    strides = (unet_stride,) * (unet_depth - 1)

    if not isinstance(norm_division_factor, int) or norm_division_factor < 1:
        raise ValueError(
            f"norm_division_factor must be a positive divisor of {first_channels}, "
            f"got {norm_division_factor}"
        )
    if first_channels % norm_division_factor != 0:
        valid = [d for d in range(1, first_channels + 1) if first_channels % d == 0]
        raise ValueError(
            f"norm_division_factor={norm_division_factor} does not divide "
            f"first_channels={first_channels}. Valid divisors: {valid}"
        )
    num_groups = first_channels // norm_division_factor
    norm_arg = ("group", {"num_groups": num_groups})

    unet = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels,
        strides=strides,
        norm=norm_arg,
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=num_res_units,
        act="ELU",
        dropout=0.0,
    )

    # Residual mode wraps the network so it predicts the correction; direct
    # mode returns the bare U-Net, which predicts the denoised volume itself.
    model = ResidualUNet(unet) if residual else unet

    # Handle device setup
    if torch.cuda.is_available() and device != "cpu":
        model = model.to(f"cuda:{device}")

    return model
