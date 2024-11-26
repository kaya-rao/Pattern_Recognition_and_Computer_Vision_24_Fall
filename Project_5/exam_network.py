# Yunxuan 'Kaya' Rao
# Nov 25, 2024
# Exam the network and analyze how it processes the data


import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
import argparse
import matplotlib.pyplot as plt
from model_train import MyNetwork


def analyze_first_layer(model):
    # The weights of the first convolutional layer (conv10)
    conv1_weights = model.conv10.weight.data  # Shape: [10, 1, 5, 5]
    print("\nWeights of the first layer (conv10):")
    print(conv1_weights)
    print("\nShape of the weights tensor:", conv1_weights.shape)

    # Visualize the filters in a 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        # There are less than 12 filters
        if i >= conv1_weights.shape[0]:  
            ax.axis('off')
            continue
        filter_weights = conv1_weights[i, 0].numpy()  # Extract the ith 5x5 filter
        ax.imshow(filter_weights, cmap='viridis')  # Use a colormap like 'viridis'
        ax.set_title(f"Filter {i + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()



def show_effect_on_filter(model):
    # Access the weights of the first convolutional layer
    with torch.no_grad():
        conv1_weights = model.conv10.weight.data  # Shape: [10, 1, 5, 5]
    
    # Load the MNIST training dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./Project_5/data', train=False, download=True, transform=transform),
        batch_size=1, shuffle=False
    )

    # Get the first training image
    data_iter = iter(test_loader)
    image, label = next(data_iter)  
    image_np = image[0, 0].numpy()  # Convert to numpy

    # Apply the filters to the image using OpenCV's filter2D
    filtered_images = []
    for i in range(conv1_weights.shape[0]):
        # Extract the ith filter and convert it to a NumPy array
        kernel = conv1_weights[i, 0].numpy()
        # Normalize the kernel 
        kernel = kernel / np.sum(np.abs(kernel))
        # Apply the filter
        filtered_image = cv2.filter2D(image_np, -1, kernel)
        filtered_images.append((kernel, filtered_image))

    # Plot the filter and filtered images
    fig, axes = plt.subplots(5, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        if i % 2 == 0:
            # Plot the filters
            ax.imshow(filtered_images[i // 2][0], cmap='viridis')
        else:
            # Plot the filtered images
            ax.imshow(filtered_images[i // 2][1], cmap='gray')

        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()



def analyze_model(model_path):
    # Load the trained model
    try:
        model = MyNetwork()
        model.load_state_dict(torch.load(model_path))
        print("\nModel successfully loaded!")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    # Print the model structure
    print("\nModel Structure:")
    print(model)

    # Print layer names and their corresponding modules
    print("\nLayers and Names:")
    for name, module in model.named_children():
        print(f"Layer Name: {name}, Layer Type: {type(module).__name__}")

    # Analyze the first layer
    analyze_first_layer(model)
    
    # Show the effects on filter
    show_effect_on_filter(model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print the structure of a trained model.')
    parser.add_argument('model_path', type=str, help='Path to the trained model file')
    args = parser.parse_args()

    analyze_model(args.model_path)