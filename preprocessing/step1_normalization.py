import os
import nibabel as nib
import numpy as np
import argparse
def normalize_and_save_nii(file_path, output_path):
    try:
        # Load the .nii.gz file
        nii = nib.load(file_path)
        data = nii.get_fdata()

        # Normalize the data to the range [0, 1]
        data_min = data.min()
        data_max = data.max()
        normalized_data = (data - data_min) / (data_max - data_min)

        # Create a new NIfTI image with the same metadata
        normalized_nii = nib.Nifti1Image(normalized_data, nii.affine, nii.header)

        # Save the normalized image
        nib.save(normalized_nii, output_path)
        print(f"Normalized file saved: {output_path}")

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

# Normalize a specific file

    # normalize_and_save_nii(file_path, output_path)
if __name__ == "__main__":
    # Required info
    parser = argparse.ArgumentParser(description="Classification Training.")
    parser.add_argument('-i',"--input-dir", default=None, required=True,type=str, help="images path")
    parser.add_argument('-o',"--output-dir", default="", type=str, help="path to save outputs")
    args = parser.parse_args()
    file_folder = args.input_dir
    output_folder = args = args.output_dir
    
    for file in os.listdir(file_folder):
        if file.endswith('nii.gz'):
            file_path = os.path.join(file_folder,file)
            save_path = os.path.join(output_folder,file)
            normalize_and_save_nii(file_path,save_path)
    

        