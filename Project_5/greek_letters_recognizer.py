# Yunxuan 'Kaya' Rao
# Nov 25, 2024
# Transfer Learning on Greek Letters

# Import packages
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import argparse

### Import the network from task 1
from model_train import MyNetwork

### greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        # transform the RGB images to grayscale
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        # scale and crop them to the correct size
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        # invert the intensities to match the MNIST digits
        return torchvision.transforms.functional.invert( x )

### Read the existing model from a file and load the pre-trained weight
def read_model(file_path):
    # Load the trained model
    try:
        model = MyNetwork()
        model.load_state_dict(torch.load(model_path))
        print("\nModel successfully loaded!")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

### Freeze the network weight





### Replace the last layer with a new Linear layer with three nodes




if __name__ == "__main__":
    # Get model_path and image_dir from command line
    parser = argparse.ArgumentParser(description='Transfer Learning on Greek Letters')
    parser.add_argument('model_path', type=str, help='Path to the trained model file')
    parser.add_argument('image_dir', type=str, help='Path to the directory containing greek letters\' images')
    args = parser.parse_args()

    