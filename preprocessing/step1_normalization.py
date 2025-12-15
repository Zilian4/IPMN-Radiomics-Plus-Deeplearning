import os
import nibabel as nib
import numpy as np
import argparse

def nyul_normalize(data, test_landmarks, train_landmarks):
    data = data.astype(np.float32)
    out = data.copy()

    mask = data != 0
    vals = data[mask]

    test = np.array(test_landmarks, dtype=np.float32)
    train = np.array(train_landmarks, dtype=np.float32)

    mapped = np.zeros_like(vals)

    for i in range(len(test) - 1):
        t0, t1 = test[i], test[i+1]
        r0, r1 = train[i], train[i+1]

        slope = (r1 - r0) / (t1 - t0 + 1e-6)
        intercept = r0 - slope * t0

        idx = (vals >= t0) & (vals < t1)
        mapped[idx] = slope * vals[idx] + intercept

    slope = (train[-1] - train[-2]) / (test[-1] - test[-2] + 1e-6)
    intercept = train[-1] - slope * test[-1]
    idx = vals >= test[-1]
    mapped[idx] = slope * vals[idx] + intercept

    out[mask] = mapped
    return out

def normalize_and_save_nii(file_path, output_path):
    try:
        # Load the .nii.gz file
        nii = nib.load(file_path)
        data = nii.get_fdata()

        train_landmarks = [6.493960, 12.197183, 15.635097, 19.892419, 23.261176, 
                            26.619607, 30.039585, 33.978876, 38.852177, 46.077461, 
                            62.049146, 96.856046, 136.691056, 265.186064, 339.037614] # t2
        test_landmarks  = [42.000000, 116.000000, 152.000000, 205.000000, 267.000000, 
                            345.000000, 446.000000, 549.000000, 653.000000, 775.000000, 
                            996.000000, 1309.000000, 1548.000000, 2109.000000, 2589.000000] # t2
        # train_landmarks = [5.659203, 19.980912, 30.529957, 44.728278, 55.261342, 
        #                    63.839996, 71.569601, 77.959418, 83.664361, 91.304222, 
        #                    101.305804, 109.691463, 115.697906, 129.491611, 138.693975] # t1
        # test_landmarks  = [34.000000, 62.000000, 95.000000, 171.000000, 223.000000, 
        #                    278.000000, 345.000000, 424.000000, 566.000000, 891.000000, 
        #                    1431.000000, 1813.000000, 2022.000000, 2407.000000, 2527.000000] # t1

        normalized_data = nyul_normalize(data, test_landmarks, train_landmarks)

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
        if file.endswith('nii.gz') or file.endswith('nii'):
            file_path = os.path.join(file_folder,file)
            save_path = os.path.join(output_folder,file)
            normalize_and_save_nii(file_path,save_path)
