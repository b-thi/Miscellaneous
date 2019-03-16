# Image Classification using a Logistic Regression from PyTorch
# Barinder Thind

# Loading libraries
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as dsets
from torch.nn import functional as F

# Loading data set
train_dataset = dsets.MNIST(root= './data', 
                            train=True, 
                            transform = torchvision.transforms.ToTensor(), 
                            download=True)

# Looking at a single observation
train_dataset[0]

# Visualizing an image
show_img = train_dataset[0][0].numpy().reshape(28, 28)
plt.imshow(show_img, cmap='gray')

# Looking at label
train_dataset[0][1]

# Iterations
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size = 10,
                                           shuffle=True)

# Model
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        out = F.softmax(self.linear(x))
        return out

input_dim = 28*28 
output_dim = 10
model = LogisticRegressionModel(input_dim, output_dim)

# Defining some parameters
criterion = torch.nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training model
iter = 0
num_epochs = 100
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        # Load images as Variable
        images = torch.autograd.Variable(images.view(-1, 28*28))
        labels = torch.autograd.Variable(labels)
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        
        # Forward pass to get output/logits
        outputs = model(images)
        
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        optimizer.step()
        
        iter += 1
        
        if iter % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Load images to a Torch Variable
                images = Variable(images.view(-1, 28*28))
                
                # Forward pass only to get logits/output
                outputs = model(images)
                
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                
                # Total number of labels
                total += labels.size(0)
                
                # Total correct predictions
                correct += (predicted == labels).sum()
                
            accuracy = 100 * correct / total
            
            # Print Loss
            print("Iteration: {}. Loss: {}. Accuracy: {}".format(iter, loss.item(), accuracy))
            
            
# Printing outputs
iter_test = 0
for images, labels in test_loader:
    iter_test += 1
    images = images.view(-1, 28*28).requires_grad_()
    outputs = model(images)
    if iter_test == 1:
        print('OUTPUTS')
        print(outputs)
    _, predicted = torch.max(outputs.data, 1)