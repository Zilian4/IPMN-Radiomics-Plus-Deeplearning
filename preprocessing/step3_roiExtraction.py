import nibabel as nib
import numpy as np
import argparse
import os
def extract_roi(ct_image_path, segmentation_mask_path, output_path):
    """
    Extract ROI based on the non-zero region of a segmentation mask and save it.
    
    Args:
        ct_image_path (str): Path to the CT image (.nii.gz file).
        segmentation_mask_path (str): Path to the segmentation mask (.nii.gz file).
                                      The mask should contain 1 for the object and 0 for the background.
        output_path (str): Path to save the ROI image (.nii.gz file).
    """
    # Load the CT image
    ct_img = nib.load(ct_image_path)
    ct_data = ct_img.get_fdata()  # 3D array of CT image

    # Load the segmentation mask
    mask_img = nib.load(segmentation_mask_path)
    mask_data = mask_img.get_fdata()  # 3D array of the mask

    # Ensure the dimensions match
    if ct_data.shape != mask_data.shape:
        raise ValueError("CT image and segmentation mask must have the same dimensions.")

    # Find the bounding box of the mask
    non_zero_coords = np.array(np.nonzero(mask_data))  # Shape: (3, N) where N is the number of non-zero voxels
    min_coords = non_zero_coords.min(axis=1)  # (x_min, y_min, z_min)
    max_coords = non_zero_coords.max(axis=1) + 1  # (x_max, y_max, z_max), +1 for inclusive slicing

    # Extract the ROI from the CT image using the bounding box
    x_min, y_min, z_min = min_coords
    x_max, y_max, z_max = max_coords
    roi_data = ct_data[x_min:x_max, y_min:y_max, z_min:z_max]

    # Save the ROI as a new NIfTI image
    roi_affine = ct_img.affine
    roi_header = ct_img.header.copy()  # Copy the header to preserve metadata
    roi_img = nib.Nifti1Image(roi_data, roi_affine, roi_header)

    nib.save(roi_img, output_path)
    print(f"ROI image saved to: {output_path}")
    print(f"ROI bounds: x[{x_min}:{x_max}], y[{y_min}:{y_max}], z[{z_min}:{z_max}]")


if __name__ == "__main__":
    # Required info
    parser = argparse.ArgumentParser(description="Classification Training.")
    parser.add_argument('-i',"--input-dir", default=None, required=True,type=str, help="images path")
    parser.add_argument('-m',"--mask-dir", default=None, required=True,type=str, help="images path")
    parser.add_argument('-o',"--output-dir", default="", type=str, help="path to save outputs")
    args = parser.parse_args()
    file_folder = args.input_dir
    mask_folder = args.mask_dir
    output_folder = args = args.output_dir
    
    for file in os.listdir(file_folder):
        file_path = os.path.join(file_folder,file)
        mask_path = os.path.join(mask_folder,file)
        save_path = os.path.join(output_folder,file)
        if file_path.endswith('.nii.gz'):
            extract_roi(file_path,mask_path,save_path)
    