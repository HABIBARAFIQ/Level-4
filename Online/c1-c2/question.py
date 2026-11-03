import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For GPU (if used)
    torch.cuda.manual_seed_all(seed)  # For all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False


set_seed(42)

# Image transformations
transform = transforms.Compose([
    # TODO: Resize to 224x224
    transforms.Resize((224, 224)),
    # TODO: Convert to tensor
    transforms.ToTensor(),
    # TODO: Normalize with mean and std
    transforms.Normalize(mean=[0.485,
    0.456, 0.406],std=[0.229, 0.224, 0.225])

])

# Load dataset
dataset = datasets.ImageFolder(
    root="Online\\c1-c2\\images",
    transform=transform
)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=30,
    shuffle=True,
)

images, labels = next(iter(dataloader))

print("Batch image tensor shape:", images.shape)
print("Batch labels tensor shape:", labels.shape)

# Number of classes (auto-detected)
num_classes = len(dataset.classes)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # TODO: Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=5,padding=2,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,padding=0,stride=2)
        )

        # TODO: Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=19,padding=3,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,padding=0,stride=2)
        )

        # TODO: Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=4,padding=8,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,padding=0,stride=2)
        )

        # TODO: Block 4
        self.linear1 = nn.Sequential(
            nn.Linear(64*16*16,128),
            nn.ReLU()
        )

        # # TODO: Block 5
        self.linear2 = nn.Linear(128,2)
        self.dummy()
    def dummy(self):
        x = torch.randn(1,3,224,224)
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)    

    def forward(self, x):
        # TODO: Block 1
        x = self.conv1(x)

        # TODO: Block 2
        x = self.conv2(x)
        # TODO: Block 3
        x = self.conv3(x)
        # TODO: Block 4
        x = self.linear1(torch.flatten(x,1))

        # TODO: Block 5
        x = self.linear2(x)

        return x


device = torch.device("cpu")

model = SimpleCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
