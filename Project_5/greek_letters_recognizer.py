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
        model.load_state_dict(torch.load(file_path))
        print("\nModel successfully loaded!")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {file_path}")
        return

# DataLoader for the Greek dataset
def load_greek_letters(training_set_path):
    greek_train = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            training_set_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                GreekTransform(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=5,
        shuffle=True
    )
    return greek_train

# Fine-Tune the MNIST Network for Greek Letters
def train_greek_network(file_path, image_dir):
    # Load the model
    model = read_model(file_path)

    ### Freeze the network weight
    for param in model.parameters():
        param.requires_grad = False

    ### Replace the last layer with a new Linear layer with three nodes
    model.fc2 = nn.Linear(50, 3)  # Output features -> 3 nodes
    print("Replaced the last layer for Greek letter classification.")

    # Optimizer and loss function
    optimizer = optim.SGD(model.fc2.parameters(), lr=0.01, momentum=0.9)  
    # Only train the new layer
    criterion = nn.CrossEntropyLoss()

    # load training set
    greek_train = load_greek_letters(image_dir)

    
    # Training Loop
    train_counter = []
    train_losses = []
    
    n_epochs = 20

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(greek_train):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            
            # Record training loss and counter for ploting
            train_losses.append(loss.item())
            train_counter.append(batch_idx * len(data) + (epoch - 1) * len(greek_train.dataset))
            

        accuracy = 100. * correct / total
        print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Stop training early if near-perfect accuracy is achieved
        if accuracy >= 99.0:
            print("Achieved near-perfect accuracy. Stopping training.")
            break
    
    
    # Save the fine-tuned model
    torch.save(model.state_dict(), './Project_5/results/greek_model.pth')
    print("Saved the fine-tuned Greek letter model.")

    print("Modified Network Structure (Layer by Layer):")
    for name, layer in model.named_children():
        print(f"Layer Name: {name}, Layer Type: {type(layer).__name__}")
    
    
    # Plot training errors
    def plot_training_error(train_counter, train_losses):
        fig = plt.figure(figsize=(8, 6))
        plt.plot(train_counter, train_losses, marker='o', linestyle='-', color='blue', label='Train Loss')
        plt.title('Training Error Over Epochs')
        plt.xlabel('Number of Training Examples Seen')
        plt.ylabel('Negative Log Likelihood Loss')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
    
    plot_training_error(train_counter, train_losses)
    
# Evaluate the Network on Test Images from a Folder

def evaluate_greek_model_from_folder(model_path, test_folder_path):
    # Load the fine-tuned model
    model = MyNetwork()
    model.fc2 = nn.Linear(50, 3)  # Adjust the last layer
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # DataLoader for the test folder
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            test_folder_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                GreekTransform(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=1,  # One image at a time for evaluation
        shuffle=False
    )

    # Map class indices to Greek letter names
    class_names = test_loader.dataset.classes
    print(f"Class mapping: {class_names}")
    
    # Evaluate each test image
    correct = 0
    total = 0
    for data, target in test_loader:
        output = model(data)
        predicted_class = output.argmax(dim=1).item()
        true_class = target.item()

        print(f"True Label: {class_names[true_class]}, Predicted Label: {class_names[predicted_class]}")
        if predicted_class == true_class:
            correct += 1
        total += 1

    # Print overall accuracy
    accuracy = 100. * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    
    # Evaluate and display each test image
    fig = plt.figure(figsize=(10, 10))
    num_images = len(test_loader.dataset)
    rows = (num_images + 2) // 5  # Calculate number of rows for the plot
    cols = 5

    for idx, (data, target) in enumerate(test_loader):
        output = model(data)
        predicted_class = output.argmax(dim=1).item()
        true_class = target.item()

        # Denormalize the image for display
        image = data[0].squeeze().numpy()
        image = (image * 0.3081 + 0.1307)  # Reverse normalization

        # Display the image
        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.imshow(image, cmap='gray')
        ax.set_title(f"True: {class_names[true_class]}\nPred: {class_names[predicted_class]}")
        ax.axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Get model_path, train_dir and test_dir from command line
    parser = argparse.ArgumentParser(description='Transfer Learning on Greek Letters')
    parser.add_argument('model_path', type=str, help='Path to the trained model file')
    parser.add_argument('train_dir', type=str, help='Path to the directory containing greek letters\' images')
    parser.add_argument('test_dir', type=str, help='Path to the test  files')
    args = parser.parse_args()

    # Train the greek network
    train_greek_network(args.model_path, args.train_dir)

    # Evaluate the network on test images
    evaluate_greek_model_from_folder('./Project_5/results/greek_model.pth', args.test_dir)

    