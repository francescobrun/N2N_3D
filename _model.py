from monai.networks.nets import UNet
import torch


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
    """

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        return x + self.unet(x)


def create_model(
    device=0, norm_division_factor=56, num_res_units=0, unet_depth=4, unet_stride=2
):
    """
    Create a 3D U-Net model with configurable normalization, wrapped in
    image-level residual learning mode (output = input + unet(input)).

    Args:
        device (int): The device to use (default: 0 for GPU, or 'cpu' for CPU)
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
                image-level residual wrapper (ResidualUNet). Values of 1 or 2 add
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
                Setting it to 1 selects instance normalization (num_groups=56);
                intermediate divisors give true group normalization.
    Returns:
        torch.nn.Module: a ResidualUNet wrapping the underlying MONAI UNet.
        The inner network is accessible as ``model.unet``.
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

    model = ResidualUNet(unet)

    # Handle device setup
    if torch.cuda.is_available() and device != "cpu":
        model = model.to(f"cuda:{device}")

    return model
