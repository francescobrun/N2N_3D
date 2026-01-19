from monai.networks.nets import UNet
import torch


def create_model(device=0, norm_division_factor=1):
    """
    Create a 3D U-Net model with configurable normalization.
    
    Args:
        device (int): The device to use (default: 0 for GPU, or 'cpu' for CPU)
        norm_division_factor (int): Division factor for group normalization. The 
                number of groups is calculated as: first_channels // norm_division_factor.
                Default is 1, which means instance normalization.    
    Returns:
        torch.nn.Module: The created U-Net model
    """
    # Calculate number of groups based on division factor
    first_channels = 56  # First channel in the channels list
    
    # Ensure integer division and validate input
    if not isinstance(norm_division_factor, int) or norm_division_factor <= 0:
        raise ValueError(f"norm_division_factor must be a positive integer, got {norm_division_factor}")
    
    num_groups = first_channels // norm_division_factor  # Integer division (floor)
    
    # Ensure num_groups is at least 1 and doesn't exceed first_channels
    num_groups = max(1, min(num_groups, first_channels))
        
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=[56, 112, 224, 448],
        strides=[2, 2, 2],
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
