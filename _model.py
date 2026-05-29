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


def create_model(device=0, norm_division_factor=1):
    """
    Create a 3D U-Net model with configurable normalization, wrapped in
    image-level residual learning mode (output = input + unet(input)).

    Args:
        device (int): The device to use (default: 0 for GPU, or 'cpu' for CPU)
        norm_division_factor (int): Division factor for group normalization. The
                number of groups is calculated as: first_channels // norm_division_factor.
                Must be a positive divisor of the first channel count (56), so
                valid values are 1, 2, 4, 7, 8, 14, 28, 56. Default is 1, which
                means instance normalization.
    Returns:
        torch.nn.Module: a ResidualUNet wrapping the underlying MONAI UNet.
        The inner network is accessible as ``model.unet``.
    """
    channels = (56, 112, 224, 448)
    first_channels = channels[0]

    if not isinstance(norm_division_factor, int) or norm_division_factor <= 0:
        raise ValueError(
            f"norm_division_factor must be a positive integer, got {norm_division_factor}"
        )
    if first_channels % norm_division_factor != 0:
        valid = [d for d in range(1, first_channels + 1) if first_channels % d == 0]
        raise ValueError(
            f"norm_division_factor={norm_division_factor} does not divide "
            f"first_channels={first_channels}. Valid divisors: {valid}"
        )
    num_groups = first_channels // norm_division_factor

    unet = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels,
        strides=(2, 2, 2),
        norm=("group", {"num_groups": num_groups}),
        # Everything beyond this is actually default:
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=0,
        act="ELU",
        dropout=0.0,
    )

    model = ResidualUNet(unet)

    # Handle device setup
    if torch.cuda.is_available() and device != 'cpu':
        model = model.to(f'cuda:{device}')

    return model
