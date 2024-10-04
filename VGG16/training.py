import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from model import ImageClassifierCNN  # Import the model class
from torchvision import datasets, transforms

# Hyperparameters
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# Transformation data
transform = transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder(root='/Users/vedanshkumar/Documents/Projects_sem5/IntelGenAI/dataset/Vandalism/Spit/', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = ImageClassifierCNN()
criterion = nn.BCELoss()  # Since you are using sigmoid, Binary Cross Entropy Loss is appropriate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to('mps'), labels.float().to('mps')  # Transfer data to GPU if available
        model = model.to('mps')
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

print('Training complete.')
