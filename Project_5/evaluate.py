# Yunxuan 'Kaya' Rao
# Nov 25, 2024
# Evaluate the trained network on the first 10 examples in the test set.


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import argparse

# Use the same network so I'll just import it
from model_train import MyNetwork


def evaluate(file_path):
    # Load the pre-saved model
    try:
        model = MyNetwork()
        model.load_state_dict(torch.load(file_path))
        model.eval()  # Set the model to evaluation mode
    except FileNotFoundError:
        print(f"Error: Model file not found at {file_path}")
        return
    

    # Load the MNIST test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./Project_5/data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=10, shuffle=False)
    


    # Extract the first 10 images and labels
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # Run the model on these images
    outputs = model(images)

    # Print the results for each image
    for i in range(10):
        output = outputs[i]
        # Convert log_softmax to probabilities
        probabilities = torch.exp(output)  
        probs_rounded = [f"{p:.2f}" for p in probabilities.tolist()]
        predicted_label = torch.argmax(probabilities).item()
        true_label = labels[i].item()

        print(f"Image {i+1}:")
        print(f"  Network Output: {probs_rounded}")
        print(f"  Predicted Label: {predicted_label}")
        print(f"  True Label: {true_label}")
        print("--------------------------------------")
    
    # Plot the first 9 images in a 3x3 grid with predictions
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        if i >= 9:
            break
        ax.imshow(images[i][0].numpy(), cmap='gray')
        ax.set_title(f"Pred: {torch.argmax(outputs[i]).item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()



# Main funtion, take argument from command line as file path
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained MNIST model.')
    parser.add_argument('model_path', type=str, help='Path to the trained model file')
    args = parser.parse_args()
    evaluate(args.model_path)