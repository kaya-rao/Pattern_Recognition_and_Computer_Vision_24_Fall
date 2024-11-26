# Yunxuan 'Kaya' Rao
# Nov 25, 2024
# Read handwritten digit images, preprocess them, and evaluate using the pre-trained network.

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import argparse
from model_train import MyNetwork

def preprocess_image(image_path):
    """
    Preprocess a single image: convert to grayscale, resize to 28x28,
    and normalize intensity to match MNIST dataset.
    """
    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize
    ])

    try:
        image = Image.open(image_path)
        processed_image = transform_pipeline(image)
        return processed_image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def evaluate_custom_images(model_path, image_dir):
    # Load the saved model
    try:
        model = MyNetwork()
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    # Collect all image paths from the specified directory
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_paths:
        print(f"No images found in directory: {image_dir}")
        return

    # Preprocess images and run them through the network
    predictions = []
    processed_images = []

    for image_path in image_paths:
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            continue
        processed_images.append(processed_image)
        
        # Add batch dimension (1, 1, 28, 28)
        input_tensor = processed_image.unsqueeze(0)
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
        predictions.append((image_path, predicted_label))

    # Print predictions
    for image_path, predicted_label in predictions:
        print(f"Image: {image_path}, Predicted Label: {predicted_label}")

    # Plot the first 10 images in a 2x5 grid with predictions
    fig, axes = plt.subplots(2, 5, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        if i >= len(processed_images):
            break
        ax.imshow(processed_images[i].squeeze(0), cmap='gray')
        ax.set_title(f"Pred: {predictions[i][1]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate custom handwritten digit images.')
    parser.add_argument('model_path', type=str, help='Path to the trained model file')
    parser.add_argument('image_dir', type=str, help='Path to the directory containing handwritten digit images')
    args = parser.parse_args()

    evaluate_custom_images(args.model_path, args.image_dir)
