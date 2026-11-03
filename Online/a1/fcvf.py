import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

h = 224
w = 224

# # MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                            train=True, 
#                                            transform=transforms.ToTensor(),
#                                            download=True)

# test_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                           train=False, 
#                                           transform=transforms.ToTensor())

# MNIST dataset
train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


# test_block = True
# test_full = True 


test_block = False
test_full = False 


class NiNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=3):
        super(NiNBlock, self).__init__()

        p = (k-1)//2

        x = torch.rand(1,in_channels,h,w)

        global test_block

        if test_block:
            print("\nTesting Single Block\n")

        if test_block:
            print(x.shape)

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, stride=1),
            nn.ReLU()
        )

        x = self.layer_1(x)
        
        if test_block:
            print(x.shape)

        self.layer_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, stride=1),
            nn.ReLU()
        )

        x = self.layer_2(x)
        
        if test_block:
            print(x.shape)

        self.layer_3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, stride=1),
            nn.ReLU()
        )

        x = self.layer_3(x)
        
        if test_block:
            print(x.shape)

    
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        return x
    

class NiN3(nn.Module):
    def __init__(self, in_channels=3, k=3, num_classes=10):
        super(NiN3, self).__init__()

        global test_full

        if test_full:
            print("\nTesting Full Net\n")

        x = torch.rand(1,in_channels,h,w)

        if test_full:
            print(x.shape)

        self.nin1 = nn.Sequential(
            NiNBlock(in_channels, 32, k)
        )

        x = self.nin1(x)

        if test_full:
            print(x.shape)

        self.p1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )

        x = self.p1(x)

        if test_full:
            print(x.shape)

        self.nin2 = nn.Sequential(
            NiNBlock(32,64,k)
        )

        x = self.nin2(x)

        if test_full:
            print(x.shape)

        self.p2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )

        x = self.p2(x)

        if test_full:
            print(x.shape)

        self.nin3 = nn.Sequential(
            NiNBlock(64,64,k)
        )

        x = self.nin3(x)

        if test_full:
            print(x.shape)

        self.conv = nn.Conv2d(64,10,kernel_size=1,padding=0,stride=1)

        x = self.conv(x)

        if test_full:
            print(x.shape)

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1)
        )

        x = self.gap(x)

        if test_full:
            print(x.shape)

        self.fc = nn.Sequential(
            nn.Linear(x.shape[1],num_classes)
        )

        if test_full:
            print(x.shape)

    
    def forward(self, x):
        x = self.nin1(x)
        x = self.p1(x)

        x = self.nin2(x)
        x = self.p2(x)

        x = self.nin3(x)

        x = self.conv(x)
        x = self.gap(x)

        x = self.fc(x)

        return x





# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# model = ConvNet(num_classes).to(device)

model = NiN3(in_channels=3, k=3, num_classes=10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        # print(images[0].shape)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')