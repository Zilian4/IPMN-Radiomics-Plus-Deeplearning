from monai.networks.nets import DenseNet121

def get_model():
    return DenseNet121(
        spatial_dims=3,  # 3D input
        in_channels=1,   # Typically for grayscale (e.g., MRI/CT scans), change to 3 for RGB
        out_channels=3,   # Adjust for binary or multi-class segmentation/classification
        dropout_prob=0.3, 
        # dropout_dim = 3
    )

