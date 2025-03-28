import nibabel as nib
import os
import argparse

def clean(file_path, seg_path,output_path):
    try:
        ct_image_path = file_path
        ct_img = nib.load(ct_image_path)
        ct_data = ct_img.get_fdata()  
        segmentation_path = seg_path
        seg_img = nib.load(segmentation_path)
        seg_data = seg_img.get_fdata()
        
        result_data = ct_data * seg_data 
        result_img = nib.Nifti1Image(result_data, ct_img.affine)
        nib.save(result_img, output_path)
        print(f"Cleaned file saved: {output_path}")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")


if __name__ == "__main__":
    # Required info
    parser = argparse.ArgumentParser(description="Classification Training.")
    parser.add_argument('-i',"--input-dir", default=None, required=True,type=str, help="images path")
    parser.add_argument('-m',"--mask-dir", default=None, required=True,type=str, help="mask path")
    parser.add_argument('-o',"--output-dir", default="", type=str, help="path to save outputs")
    args = parser.parse_args()
    file_folder = args.input_dir
    mask_folder = args.mask_dir
    output_folder = args = args.output_dir
    
    for file in os.listdir(file_folder):
        file_path = os.path.join(file_folder,file)
        mask_path = os.path.join(mask_folder,file)
        save_path = os.path.join(output_folder,file)
        clean(file_path,mask_path,save_path)