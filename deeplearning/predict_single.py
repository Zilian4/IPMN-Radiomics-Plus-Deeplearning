import numpy as np
import torch
from monai.networks.nets import DenseNet121
from monai.data import DataLoader, ImageDataset
from monai.transforms import Compose, ScaleIntensity, EnsureChannelFirst, Resize
import os
import argparse
import json

def load_model(model_path, device='cuda'):
    """
    Load a trained DenseNet121 model
    
    Args:
        model_path: Path to the saved model weights (.pth file)
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        Loaded model ready for inference
    """
    densenet = DenseNet121(
        spatial_dims=3,  # 3D input
        in_channels=1,   # Grayscale (e.g., MRI/CT scans)
        out_channels=2   # Binary classification
    )
    
    densenet.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    densenet.to(device)
    densenet.eval()  # Set to evaluation mode
    
    return densenet


def predict_single_file(image_path, model, device='cuda'):
    """
    Predict on a single medical image file
    
    Args:
        image_path: Path to the .nii.gz file
        model: Trained DenseNet121 model
        device: Device to run inference on
    
    Returns:
        dict with prediction class, probabilities for each class
    """
    # Define transforms (same as in cv_test.py)
    transforms = Compose([
        ScaleIntensity(), 
        EnsureChannelFirst(), 
        Resize((96, 96, 96))
    ])
    
    # Create dataset with single image (label is dummy, not used)
    dataset = ImageDataset(image_files=[image_path], labels=[0], transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Perform inference
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            pred = model(X)
            probabilities = torch.nn.functional.softmax(pred, dim=-1).cpu().detach().numpy().reshape(2)
            predicted_class = np.argmax(probabilities)
            
            return {
                'predicted_class': int(predicted_class),
                'probabilities': {
                    'class_0': float(probabilities[0]),
                    'class_1': float(probabilities[1])
                },
                'confidence': float(probabilities[predicted_class])
            }


def predict(image_path, model_dir, train_test_info='Train_Test_4', device='cuda'):
    """
    Predict using ensemble of all 5 fold models
    
    Args:
        image_path: Path to the .nii.gz file
        model_dir: Directory containing the model weights
        train_test_info: Name of the train/test split (e.g., 'Train_Test_4')
        device: Device to run inference on
    
    Returns:
        dict with ensemble prediction and individual fold predictions
    """
    all_probabilities = []
    fold_predictions = []
    image_name = os.path.basename(image_path)

    with open(os.path.join(model_dir, f'{train_test_info}.json'), 'r') as f:
        train_test_info_data = json.load(f)
    test_list = train_test_info_data['test_files']
    assert image_name in test_list, f"Image {image_name} not found in the test set in the {train_test_info} split"
    print(f"Predicting on: {image_name} from the test set in the {train_test_info} split")
    # Load and predict with each fold model
    for fold in range(5):
        model_path = os.path.join(model_dir, f'model_loss_{train_test_info}_fold{fold}.pth')
        
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            continue
        
        model = load_model(model_path, device)
        result = predict_single_file(image_path, model, device)
        
        all_probabilities.append([result['probabilities']['class_0'], 
                                 result['probabilities']['class_1']])
        fold_predictions.append(result)
        
        print(f"Using model from Fold {fold}: Class {result['predicted_class']} "
              f"(Probability: {result['confidence']:.4f})")
    



def main():
    parser = argparse.ArgumentParser(description='Predict on a single medical image file')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to the input .nii.gz file')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to a single model .pth file')
    parser.add_argument('--model_dir', type=str, 
                       default='/data/Ziliang/IPMN_cysts_20240909/DenseNet121_weights',
                       help='Directory containing model weights (for ensemble prediction)')
    parser.add_argument('--train_test_info', type=str, default='Train_Test_4',
                       help='Train/test split name')
    parser.add_argument('--fold', type=int, default=None,
                       help='Specific fold to use (0-4). If not specified, uses ensemble of all folds')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble of all 5 fold models')
    
    args = parser.parse_args()

    assert os.path.dirname(args.image_path) == '/data/Ziliang/IPMN_cysts_20240909/deeplearning/ROI','**Image path must from /data/Ziliang/IPMN_cysts_20240909/deeplearning/ROI !!!**'
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        return
    
    print(f"Predicting on: {args.image_path}")
    print("-" * 60)
    
    # Ensemble prediction (use all 5 folds)
    if args.ensemble or (args.model_path is None and args.fold is None):
        print("Using ensemble prediction (all 5 folds):")
        print("-" * 60)
        results = predict(args.image_path, args.model_dir, 
                                       args.train_test_info, args.device)
        print("-" * 60)

    
    # Single model prediction
    else:
        if args.model_path:
            model_path = args.model_path
        else:
            # Use specific fold
            fold = args.fold if args.fold is not None else 0
            model_path = os.path.join(args.model_dir, 
                                     f'model_loss_{args.train_test_info}_fold{fold}.pth')
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return
        
        print(f"Loading model from: {model_path}")
        model = load_model(model_path, args.device)
        
        result = predict_single_file(args.image_path, model, args.device)
        
        print("-" * 60)
        print("Prediction Result:")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Class 0 Probability: {result['probabilities']['class_0']:.4f}")
        print(f"  Class 1 Probability: {result['probabilities']['class_1']:.4f}")
        print(f"  Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()

