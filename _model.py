from monai.networks.nets import UNet
import torch


def create_model(device=0, norm_division_factor=1):
    """
    Create a 3D U-Net model with configurable normalization.

    Args:
        device (int): The device to use (default: 0 for GPU, or 'cpu' for CPU)
        norm_division_factor (int): Division factor for group normalization. The
                number of groups is calculated as: first_channels // norm_division_factor.
                Must be a positive divisor of the first channel count (56), so
                valid values are 1, 2, 4, 7, 8, 14, 28, 56. Default is 1, which
                means instance normalization.
    Returns:
        torch.nn.Module: The created U-Net model
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

    model = UNet(
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
        act="PRELU",
        dropout=0.0,
    )

    # Handle device setup
    if torch.cuda.is_available() and device != 'cpu':
        model = model.to(f'cuda:{device}')

    return model
