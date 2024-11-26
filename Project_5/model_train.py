# Yunxuan 'Kaya' Rao
# Nov 16, 2024
# Build and train a network to do digit recognition using the MNIST data base
# Save the network to a file

# import statements
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# class definitions
# ----------- Build a network model ----------- #
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # A convolution layer with 10 5x5 filters
        self.conv10 = nn.Conv2d(1, 10, kernel_size=5)
        # A max pooling layer with a 2x2 window and a ReLU function applied.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # A second convolution layer with 20 5x5 filters
        self.conv20 = nn.Conv2d(10, 20, kernel_size=5)
        # A dropout layer with a 0.5 dropout rate (50%)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        
        # Fully connected Linear layer with 50 nodes
        self.fc1 = nn.Linear(320, 50)
        # Final fully connected Linear layer with 10 nodes
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        # First convolution layer, apply ReLU, max pooling
        x = self.pool(torch.relu(self.conv10(x)))
        # 2nd convolution layer, apply ReLU, max pooling, dropout
        x = self.pool(torch.relu(self.conv2_drop(self.conv20(x))))

        # A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the output
        x = x.view(-1, 20 * 4 * 4)
        x = torch.relu(self.fc1(x))

        # A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output
        x = torch.log_softmax(self.fc2(x), dim=1)

        return x





# useful functions with a comment for each function
def train_network(model, train_loader, test_loader, n_epochs, learning_rate, momentum, log_interval):
    # Train the model
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    def train(epoch):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # print(batch_idx)
            if batch_idx % log_interval == 0:
                
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(batch_idx * len(data) + (epoch - 1) * len(train_loader.dataset))

                # Save the model
                torch.save(model.state_dict(), './Project_5/results/model.pth')
                torch.save(optimizer.state_dict(), './Project_5/results/optimizer.pth')

        # avg_train_loss = train_loss / len(train_loader.dataset)
        # train_losses.append(avg_train_loss)


    def test():
        # Evaluation phase
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum()

        avg_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        # test_counter.append(len(train_loader.dataset) * epoch)

        test_accuracy = 100. * correct / len(test_loader.dataset)
        print(f"\nTest set: Average loss: {avg_test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
                f"({test_accuracy:.0f}%)\n")

    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()
        

    # Plot training and testing errors
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

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

    model = MyNetwork()
    train_network(model, train_loader, test_loader, n_epochs, learning_rate, momentum, log_interval)
    return

if __name__ == "__main__":
    main(sys.argv)