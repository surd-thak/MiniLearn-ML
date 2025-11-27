import numpy as np
import torch
from torchvision import datasets, transforms

from neural_network.nn_two_layer import TwoLayerNN

# Load the EMNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.EMNIST(root='./datasets', split='letters', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./datasets', split='letters', train=False, download=True, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Extract data from the data loaders
X_train = train_loader.dataset.data.numpy().reshape(-1, 28*28) / 255.0
y_train = train_loader.dataset.targets.numpy() -1

X_test = test_loader.dataset.data.numpy().reshape(-1, 28*28) / 255.0
y_test = test_loader.dataset.targets.numpy() -1

# Initialize and train the neural network
input_size = X_train.shape[1]
hidden_size = 64
output_size = len(np.unique(y_train))

nn = TwoLayerNN(input_size, hidden_size, output_size, lr=0.1, n_iters=1000, activation='relu')
nn.fit(X_train, y_train)

# Make predictions
predictions = nn.predict(X_test)

# Calculate accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {accuracy}")