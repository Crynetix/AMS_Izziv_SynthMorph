import torch
import matplotlib.pyplot as plt
import nibabel as nib
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np


def estimate_memory_usage(nn_mult, grid_sp, disp_hw, H, W, D):
    """
    Estimate memory usage for a given configuration.
    Returns memory in bytes.
    """
    # Downsampled dimensions
    H_d = H // grid_sp
    W_d = W // grid_sp
    D_d = D // grid_sp

    # Estimate sizes of critical tensors
    size_mind_fix = nn_mult * H_d * W_d * D_d * 8  # 4 bytes per float32
    size_mind_mov = (
        nn_mult * (H_d + 2 * disp_hw) * (W_d + 2 * disp_hw) * (D_d + 2 * disp_hw) * 8
    )
    size_ssd = (2 * disp_hw + 1) ** 3 * H_d * W_d * D_d * 8

    # Total estimated memory
    total_memory = size_mind_fix + size_mind_mov + size_ssd
    return total_memory


def estimate_memory_usage_MIND(H, W, D, n_ch, disp_hw, grid_sp):
    """
    Estimate GPU memory usage for MIND descriptor correlation.

    Parameters:
    - H, W, D: Dimensions of the input image.
    - n_ch: Number of feature channels in MIND descriptors.
    - disp_hw: Displacement half-width for correlation.
    - grid_sp: Grid spacing (downsampling factor).

    Returns:
    - Estimated memory usage in bytes.
    """
    # Downsampled spatial dimensions
    H_d = H // grid_sp
    W_d = W // grid_sp
    D_d = D // grid_sp

    # Displacement search space size
    disp_size = (disp_hw * 2 + 1) ** 3

    # Memory for SSD tensor (float32: 4 bytes per value)
    ssd_memory = disp_size * H_d * W_d * D_d * 4

    # Memory for padded moving feature map (n_ch channels)
    padded_mov_memory = (
        n_ch * (H + 2 * disp_hw) * (W + 2 * disp_hw) * (D + 2 * disp_hw) * 4
    )

    # Memory for intermediate tensors during SSD computation
    intermediate_memory = n_ch * H_d * W_d * D_d * 4

    # Total memory
    total_memory = ssd_memory + padded_mov_memory + intermediate_memory
    return total_memory


print(estimate_memory_usage(12, 2, 5, 192, 192, 208) / (1024**3))


def get_common_bounding_box(data1, data2, threshold1=-900, threshold2=10):
    """
    Compute a common bounding box that includes non-background regions of both images.
    :param data1: 3D tensor (e.g., CT image)
    :param data2: 3D tensor (e.g., MR image)
    :param threshold1: threshold for identifying background in data1
    :param threshold2: threshold for identifying background in data2
    :return: cropped data1 and data2 using the common bounding box
    """

    # Conver to long
    data1 = data1.long()
    data2 = data2.long()

    # Create masks for non-background regions
    mask1 = data1 > threshold1
    mask2 = data2 > threshold2

    # Combine masks to find the union of non-background regions
    combined_mask = mask1 | mask2

    # Get non-zero coordinates in each dimension for the combined mask
    non_zero_indices = torch.nonzero(combined_mask, as_tuple=True)
    min_x, max_x = non_zero_indices[0].min(), non_zero_indices[0].max()
    min_y, max_y = non_zero_indices[1].min(), non_zero_indices[1].max()
    min_z, max_z = non_zero_indices[2].min(), non_zero_indices[2].max()

    # Crop both data1 and data2 using the common bounding box
    cropped_data1 = data1[min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1]
    cropped_data2 = data2[min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1]

    return cropped_data1, cropped_data2


def adjust_values(image_ct, image_mr):
    """
    Adjust the intensity of images from 0 to 255.
    :param image_ct: 3D tensor (CT image)
    :param image_mr: 3D tensor (MR image)
    :return: adjusted CT image
    """

    # Slope calculation
    # slope = (image_mr.max() - image_mr.min()) / (image_ct.max() - image_ct.min())
    slope_ct = 255 / (image_ct.max() - image_ct.min())
    slope_mr = 255 / (image_mr.max() - image_mr.min())

    # Adjust and round the intensities
    adjusted_image_ct = torch.round(slope_ct * (image_ct - image_ct.min()))
    adjusted_image_mr = torch.round(slope_mr * (image_mr - image_mr.min()))

    # Clip values to ensure they are within [0, 255]
    adjusted_image_ct = torch.clamp(adjusted_image_ct, 0, 255)
    adjusted_image_mr = torch.clamp(adjusted_image_mr, 0, 255)

    return adjusted_image_ct, adjusted_image_mr


def downsample_image(image, scale_factor=None, target_size=None, mode="nearest"):
    """
    Downsample a 3D image tensor.

    :param image: 3D tensor of shape (H, W, D)
    :param scale_factor: Factor to scale the image (e.g., 0.5 for half the size)
    :param target_size: Tuple of target dimensions (H, W, D)
    :param mode: Interpolation mode, 'trilinear' or 'nearest' for binary masks
    :return: Downsampled 3D tensor
    """
    # Add batch and channel dimensions to use with F.interpolate
    image = image.float().unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, H, W, D)

    # Downsample using scale_factor or target_size
    if scale_factor is not None:
        # downsampled_image = F.interpolate(image, scale_factor=scale_factor, mode=mode, align_corners=False)
        downsampled_image = F.interpolate(image, scale_factor=scale_factor, mode=mode)
    elif target_size is not None:
        # downsampled_image = F.interpolate(image, size=target_size, mode=mode, align_corners=False)
        downsampled_image = F.interpolate(image, size=target_size, mode=mode)
    else:
        raise ValueError("Either scale_factor or target_size must be provided.")

    # Remove batch and channel dimensions
    downsampled_image = downsampled_image.squeeze(0).squeeze(0)
    return downsampled_image

def apply_displacement_field(img1, img2, disp, result):
    """
    Apply displacement field to an image.

    :param img1: Path to the fixed image
    :param img2: Path to the moving image
    :param disp: Path to the displacement field
    :param result: Path to save the result
    """
    img1 = sitk.ReadImage(img1)
    img2 = sitk.ReadImage(img2)
    disp = sitk.ReadImage(disp)

    img1_array = sitk.GetArrayFromImage(img1)
    img2_array = sitk.GetArrayFromImage(img2)
    disp_array = sitk.GetArrayFromImage(disp)

    disp_array = np.transpose(disp_array, (1, 2, 3, 0))

    disp_vector = sitk.GetImageFromArray(disp_array, isVector=True)

    disp_vector.CopyInformation(img2)

    disp_vector = sitk.Cast(disp_vector, sitk.sitkVectorFloat64)

    disp_transform = sitk.DisplacementFieldTransform(disp_vector)

    warped_img = sitk.Resample(img2, img1, disp_transform, sitk.sitkBSpline, 0.0, img2.GetPixelID())

    warped_img_array = sitk.GetArrayFromImage(warped_img)

    diff = img1_array - warped_img_array

    sitk.WriteImage(warped_img, result + "/warped_img.nii.gz")

    slice_index = warped_img_array.shape[0] // 2

    img1_slice = img1_array[:, :, slice_index]
    img2_slice = img2_array[:, :, slice_index]
    warped_img_slice = warped_img_array[:, :, slice_index]

    plt.imshow(img1_slice, cmap="gray")
    plt.imshow(img2_slice, cmap="jet", alpha=0.5)
    plt.show()

    plt.imshow(img1_slice, cmap="gray")
    plt.imshow(warped_img_slice, cmap="jet", alpha=0.5)
    plt.show()


