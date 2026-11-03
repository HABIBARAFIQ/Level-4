import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import DataLoader


class TinyUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(TinyUNet, self).__init__()
        # Implement the Unet architecture
        # Input Resolution: 1 ×32 ×32 (Grayscale)
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # Pool1: kernel=2, stride=2 -> 32x32 -> 16x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Bottle: input from pooled enc1 (8 channels, 16x16) -> 16 channels
        # kernel=3, padding=1 to keep 16x16
        self.bottle = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Up conv: ConvTranspose2d to upsample 16x16 -> 32x32
        # kernel=2, stride=2, padding=0
        self.upConv = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0)

        # Dec1: after concat we have 8 + 8 = 16 channels at 32x32 -> reduce to 8
        # kernel=3, padding=1 to preserve spatial dimensions
        self.dec1 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final 1x1 conv to get back to single output channel
        self.finalConv = nn.Sequential(
            nn.Conv2d(8, out_channels, kernel_size=1, stride=1, padding=0)
        )

        self.dummy()
    def dummy(self):
        # This is a dummy function to help you understand the architecture of the Inception block. 
        # You can remove this function and implement the actual Inception block in the __init__ function.
        x = torch.randn(1,1,32,32)
        print("Dummy input shape:", x.shape)
        x = self.enc1_conv(x)
        print("After enc1 shape:", x.shape)
        x = self.pool1(x)
        print("After pool1 shape:", x.shape)
        x = self.bottle(x)
        print("After bottle shape:", x.shape)
        x = self.upConv(x)
        print("After upConv shape:", x.shape)
        x = torch.cat((x, x), dim=1)  # Concatenate with itself to simulate the skip connection
        print("After concatenation shape:", x.shape)
        x = self.dec1(x)
        print("After dec1 shape:", x.shape)
        x = self.finalConv(x)
        print("After finalConv shape:", x.shape)


    def forward(self, x):
    # Encoder
        enc1 = self.enc1_conv(x)      # (B,8,32,32)
        enc1_pooled = self.pool1(enc1)  # (B,8,16,16)

    # Bottleneck
        bottle = self.bottle(enc1_pooled)

    # Decoder
        up = self.upConv(bottle)      # (B,8,32,32)

    # Concatenate with pre-pooled feature map
        cat = torch.cat((up, enc1), dim=1)

        dec1 = self.dec1(cat)
        out = self.finalConv(dec1)

        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Pad(2),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
from torch.utils.data import Subset

small_trainset = Subset(trainset, range(10000))
trainloader = DataLoader(small_trainset, batch_size=512, shuffle=True)

model = TinyUNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting Training...")
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, _ = data
        inputs = inputs.to(device)
        # Implement the Training Step
        # 1. Clear the gradients from the previous iteration
        optimizer.zero_grad()
        # 2. Forward pass: Compute the predicted reconstruction
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        # 3. Compute loss, perform backpropagation, and update weights
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print("Finished Training.")