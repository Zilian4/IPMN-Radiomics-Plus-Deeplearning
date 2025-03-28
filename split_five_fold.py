import os
import json
import random
import argparse
from typing import List, Dict, Any
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from sklearn.model_selection import KFold
import json

def generate_cross_validation_json(folder_path: str, test_path, output_json_path: str = 'cross_validation.json') -> None:

    df = pd.read_csv(folder_path)

    # 创建 KFold 对象，5个fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 构建cross-validation字典
    cross_validation = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_files = df.loc[train_idx, 'name'].tolist()
        validation_files = df.loc[val_idx, 'name'].tolist()
        
        cross_validation.append({
            "fold": fold,
            "train_files": train_files,
            "validation_files": validation_files
        })

    # 保存到 JSON 文件
    output = {"cross_validation": cross_validation}

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"Cross validation split saved as JSON:{output_json_path}")

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for input and output paths.
    Returns:
        argparse.Namespace: Parsed arguments including input and output paths.
    """
    parser = argparse.ArgumentParser(description="Generate a JSON file with 10% test split and five-fold cross-validation.")
    parser.add_argument('-i', '--input', required=True, help="Path to the folder containing files for cross-validation.")
    parser.add_argument('-o', '--output', default='cross_validation.json', help="Output path for the JSON file (default: cross_validation.json).")
    parser.add_argument('-t', '--testset',default=None,help="Path to the folder containing files for cross-validation.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    generate_cross_validation_json(folder_path=args.input, output_json_path=args.output,test_path=args.testset)
