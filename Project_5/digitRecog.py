# Yunxuan 'Kaya' Rao
# Nov 16, 2024
# Build and train a network to do digit recognition using the MNIST data base
# Save the network to a file

# import statements
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# class definitions

class MyNetwork(nn.Module):
    def __init__(self):
        pass

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        return x

# useful functions with a comment for each function
def train_network( arguments ):
    pass

# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv

    # main function code

    # Preparing the dataset
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    
    # Load the MNIST dataset
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=False)

    # Visualize the first six images in the test set
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    # Plot the first six test images
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f"Ground Truth: {example_targets[i].item()}")
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main(sys.argv)