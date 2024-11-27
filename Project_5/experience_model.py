# Yunxuan 'Kaya' Rao
# Nov 26, 2024
# Run experience on the MNIST data base

# import statements
import sys
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import itertools
from model_train import MyNetwork


# useful functions with a comment for each function
def experiment_network(model, train_loader, test_loader, n_epochs, learning_rate, momentum, log_interval):
    # Train the model
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    def train(epoch):
        # Training phase
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()


    def test():
        # Evaluation phase
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
        return accuracy


    for epoch in range(n_epochs):
        train(epoch)

    return test()
        

# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv
    # main function code
    # Preparing the dataset
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    
    # -------- Load the MNIST dataset ----------- #
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./Project_5/data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size_test, shuffle=False)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./Project_5/data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size_train, shuffle=True)


    # ------------ Exam the dropout rates -------------- #
    # Exam dropout rates
    def exam_dropout_rates():
        dropout_rates = np.concatenate((
            np.linspace(0.1, 0.3, 5),  # Wider sampling for low dropout
            np.linspace(0.3, 0.5, 10), # Denser sampling for optimal range
            np.linspace(0.5, 0.7, 5)   # Wider sampling for high dropout
        ))

        test_accuracies = []
        
        for rate in dropout_rates:
            print(f"Testing dropout rate: {rate}")
            model = MyNetwork(dropout_rate=rate)
            accuracy = experiment_network(model, train_loader, test_loader, n_epochs, learning_rate, momentum, log_interval)
            test_accuracies.append(accuracy)
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(dropout_rates, test_accuracies, marker='o', linestyle='-', color='b')
        plt.title('Test Accuracy vs Dropout Rate')
        plt.xlabel('Dropout Rate')
        plt.ylabel('Test Accuracy (%)')
        plt.grid(True)
        plt.show()
    # exam_dropout_rates()

    # ------------ Exam the number of epochs -------------- #
    def exam_num_epochs():
        model = MyNetwork(dropout_rate=0.3)
        n_epochs_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        test_accuracies = []
        for n_epochs in n_epochs_set:
            print(f"Testing num_epochs: {n_epochs}")
            # Training on the same model, so add one epoch in each training
            accuracy = experiment_network(model, train_loader, test_loader, 1, learning_rate, momentum, log_interval)
            test_accuracies.append(accuracy)
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(n_epochs_set, test_accuracies, marker='o', linestyle='-', color='b')
        plt.title('Test Accuracy vs num_epochs')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Test Accuracy (%)')
        plt.grid(True)
        plt.show() 
    # exam_num_epochs()

    # Exam dropout rates
    def exam_pooling_sizes():
        pooling_sizes = [i for i in range(1, 6)]
        test_accuracies = []
        
        for filter_size in pooling_sizes:
            print(f"Testing pooling filter size: {filter_size}")
            model = MyNetwork(dropout_rate=0.3, pooling_size=filter_size)
            accuracy = experiment_network(model, train_loader, test_loader, n_epochs, learning_rate, momentum, log_interval)
            test_accuracies.append(accuracy)
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(pooling_sizes, test_accuracies, marker='o', linestyle='-', color='b')
        plt.title('Test Accuracy vs Pooling Filter Size')
        plt.xlabel('Pooling Filter Size')
        plt.ylabel('Test Accuracy (%)')
        plt.grid(True)
        plt.show()
    # exam_pooling_sizes()

    # With pooling_size = 3, exam num_epochs and dropout_rate
    def exam_dropout_rates_num_epochs():
        dropout_rates = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        num_epochs = [4, 6, 8, 10, 12] 
        results = []

        for dropout_rate, n_epochs in itertools.product(dropout_rates, num_epochs):
            print(f"Testing dropout_rate: {dropout_rate}, num_epochs: {n_epochs}")
            model = MyNetwork(dropout_rate, pooling_size=3)
            accuracy = experiment_network(model, train_loader, test_loader, n_epochs, learning_rate, momentum, log_interval)
            results.append((dropout_rate, n_epochs, accuracy))
            print(f"Accuracy: {accuracy:.2f}%")
        
        # Find the best combination
        best_combination = max(results, key=lambda x: x[2])  # Maximize accuracy
        print(f"Best combination: Dropout Rate = {best_combination[0]}, Epochs = {best_combination[1]}, Accuracy = {best_combination[2]:.2f}%")

        # Plot results
        dropout_rates_plot, epochs_plot, accuracies = zip(*results)
        plt.figure(figsize=(12, 6))
        scatter = plt.scatter(dropout_rates_plot, epochs_plot, c=accuracies, cmap='viridis', s=100)
        plt.colorbar(scatter, label='Accuracy (%)')
        plt.title('Dropout Rate vs. Number of Epochs vs. Accuracy')
        plt.xlabel('Dropout Rate')
        plt.ylabel('Number of Epochs')
        plt.grid(True)
        plt.show()
    
    exam_dropout_rates_num_epochs()


if __name__ == "__main__":
    main(sys.argv)