import numpy as np
import nibabel as nib
from skimage import filters, morphology, measure
from scipy import ndimage as ndi
import os

"""
    Usage: mask = get_img_mask(image_filename_with_full_path)
    ----------
    fname : filename of 4D skull image
    mask  : 3D image matrix of the brain mask
"""

def get_vol_mask(vol):
    # Threshold the image by the mean of the image
    threshold_value = np.mean(vol)/0.9
    binary_mask = (vol > threshold_value).astype(np.uint8)

    # Remove all small chunks and find the largest one
    label_image, num_features = measure.label(binary_mask, connectivity=2, return_num=True)
    region_props = measure.regionprops(label_image)
    largest_component = max(region_props, key=lambda prop: prop.area)

    # Create a binary mask for the largest component and fill the holes
    largest_component_mask = label_image == largest_component.label
    vol_mask = ndi.binary_fill_holes(largest_component_mask)

    return vol_mask

def get_img_mask(fname):
    # Load the NIfTI image
    img = nib.load(fname)
    data = img.get_fdata()
    mask_4D = np.zeros(data.shape)

    for i in range(data.shape[-1]):
        vol = data[..., i]
        mask_4D[..., i] = get_vol_mask(vol)

    mask_average = np.mean(mask_4D, axis=-1)
    mask_3D = (mask_average > 0.5).astype(np.uint8)


    #To check masking result, save the resulting brain mask as a new NIfTI image in the same directory of fname
    output_dir = os.path.dirname(fname)
    base_filename = os.path.basename(fname)
    base_name = base_filename.split(".")[0]
    output_filename = os.path.join(output_dir, base_name + "_mask.nii.gz")
    mask_nifti = nib.Nifti1Image(mask_3D, affine=img.affine)
    nib.save(mask_nifti, output_filename)

    return mask_3D


#Example:
#get_img_mask('/Users/yingjun/Desktop/nipraxis/diagnostics-radcmri/data/group-01/sub-07/func/sub-07_task-taskzero_run-02_bold.nii.gz')


