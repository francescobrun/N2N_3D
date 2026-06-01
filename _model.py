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


def create_model(device=0, norm_division_factor=56):
    """
    Create a 3D U-Net model with configurable normalization, wrapped in
    image-level residual learning mode (output = input + unet(input)).

    Args:
        device (int): The device to use (default: 0 for GPU, or 'cpu' for CPU)
        norm_division_factor (int): Division factor for group normalization. The
                number of groups is calculated as: first_channels // norm_division_factor.
                Must be a positive divisor of the first channel count (56), so
                valid values are 1, 2, 4, 7, 8, 14, 28, 56. Default is 56, which
                means layer normalization (num_groups=1) -- empirically the best
                pairing with the residual learning wrapper, because it preserves
                inter-channel structure that helps signal/noise discrimination.
                Setting it to 1 selects instance normalization (num_groups=56);
                intermediate divisors give true group normalization.

                Special value: 0 disables normalization entirely (norm=None
                in MONAI). The U-Net then relies purely on its skip
                connections and the residual wrapper for training stability.
                This is experimental — modern normalization-free networks
                (e.g. NF-ResNets, NAFNet) work, but training may converge
                more slowly or fail to converge cleanly depending on the
                data; verify with a short run before committing to it.
    Returns:
        torch.nn.Module: a ResidualUNet wrapping the underlying MONAI UNet.
        The inner network is accessible as ``model.unet``.
    """
    channels = (56, 112, 224, 448)
    first_channels = channels[0]

    if not isinstance(norm_division_factor, int) or norm_division_factor < 0:
        raise ValueError(
            f"norm_division_factor must be a non-negative integer "
            f"(0 = no normalization, positive divisors of {first_channels} = group norm), "
            f"got {norm_division_factor}"
        )
    if norm_division_factor == 0:
        # No-normalization variant: rely on the U-Net's skip connections and
        # the residual wrapper for training stability.
        norm_arg = None
    else:
        if first_channels % norm_division_factor != 0:
            valid = [d for d in range(1, first_channels + 1) if first_channels % d == 0]
            raise ValueError(
                f"norm_division_factor={norm_division_factor} does not divide "
                f"first_channels={first_channels}. Valid divisors: {valid} "
                f"(or 0 to disable normalization)"
            )
        num_groups = first_channels // norm_division_factor
        norm_arg = ("group", {"num_groups": num_groups})

    unet = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels,
        strides=(2, 2, 2),
        norm=norm_arg,
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
