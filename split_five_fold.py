import os
import json
import random
import argparse
from typing import List, Dict, Any
from sklearn.model_selection import KFold, train_test_split

def generate_cross_validation_json(folder_path: str, output_json_path: str = 'cross_validation.json') -> None:
    """
    Generates a JSON file with a 10% test split and five-fold cross-validation splits
    for the remaining 90% of files in a given folder.

    Parameters:
    - folder_path (str): Path to the folder containing files for cross-validation.
    - output_json_path (str): Path where the generated JSON file will be saved.

    JSON Output Structure:
    {
        "test_split": [<test_file1>, <test_file2>, ...],
        "cross_validation": [
            {
                "fold": <fold_number>,
                "train_files": [<train_file1>, <train_file2>, ...],
                "validation_files": [<val_file1>, <val_file2>, ...]
            },
            ...
        ]
    }
    """
    
    # Step 1: Retrieve all file names in the specified folder
    files: List[str] = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Step 2: Shuffle files to ensure randomness
    random.shuffle(files)
    
    # Step 3: Split files into 10% for testing and 90% for cross-validation
    train_files, test_files = train_test_split(files, test_size=0.1, random_state=42)
    
    # Step 4: Initialize KFold for five splits on the 90% training data
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_validation: List[Dict[str, Any]] = []
    
    # Step 5: Generate train/validation indices for each fold and add to `cross_validation` list
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_files), start=1):
        fold_train_files = [train_files[i] for i in train_idx]
        fold_val_files = [train_files[i] for i in val_idx]
        
        # Append fold details to cross-validation list
        cross_validation.append({
            'fold': fold_idx,
            'train_files': fold_train_files,
            'validation_files': fold_val_files
        })
    
    # Step 6: Create the final data structure with test and cross-validation splits
    data = {
        'test_split': test_files,
        'cross_validation': cross_validation
    }
    
    # Step 7: Write the data to the JSON file
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Cross-validation JSON file created at: {output_json_path}")

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for input and output paths.
    Returns:
        argparse.Namespace: Parsed arguments including input and output paths.
    """
    parser = argparse.ArgumentParser(description="Generate a JSON file with 10% test split and five-fold cross-validation.")
    parser.add_argument('-i', '--input', required=True, help="Path to the folder containing files for cross-validation.")
    parser.add_argument('-o', '--output', default='cross_validation.json', help="Output path for the JSON file (default: cross_validation.json).")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    generate_cross_validation_json(folder_path=args.input, output_json_path=args.output)
