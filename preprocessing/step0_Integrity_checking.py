# Checking the dimension problem
import os
import nibabel as nib
import numpy as np
import argparse
def integrity_checking(file_path, mask_path):
    try:
        # Load the .nii.gz file
        data_nii = nib.load(file_path)
        data = data_nii.get_fdata()

        mask_nii = nib.load(mask_path)
        mask = mask_nii.get_fdata()
        
        if data.shape!=mask.shape:
            print('Shape error,please check file')
            print(f'image:{file_path}\t {data.shape}')
            print(f'mask:{mask_path}\t {mask.shape}')
            
            return file_path 
        else:
            print(f'{file_path} finished')
            return
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
          

if __name__ == "__main__":
    # Required info
    parser = argparse.ArgumentParser(description="Classification Training.")
    parser.add_argument('-i',"--input-dir", default=None, required=True,type=str, help="images path")
    parser.add_argument('-m',"--mask-dir", default="", type=str, help="mask path")
    args = parser.parse_args()
    file_folder = args.input_dir
    mask_folder = args.mask_dir
    error_list = []
    for file in os.listdir(file_folder):
        file_path = os.path.join(file_folder,file)
        mask_path = os.path.join(mask_folder,file)
        if file_path.endswith('.nii.gz'):
            integrity_checking(file_path,mask_path)