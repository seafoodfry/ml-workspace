import torch.nn as nn


class DeeperCNN(nn.Module):
    def __init__(self):
        """
        See svhn-workbench.ipynb for more details about the architecture.

        Remember that SVHN images are 32x32x8.

        Initial: 32x32x3 (SVHN input)
        After block 1: 16x16x16
        After block 2: 8x8x32
        After block 3: 4x4x64
        After block 4: 2x2x128

        Many CNN architectures transition to fully connected layers when the spatial dimensions become small
        (4x4 or 2x2).
        """
        super().__init__()

        self.features = nn.Sequential(
            # First block conv: 3 (RGB) -> 32 channels.
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # First block pooling: 32x32 (height,width) -> 16x16x.
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block conv: 32 channels -> 64 channels.
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Second block pooling: 16x16 (height,width) -> 8x8.
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block conv: 64 channels -> 128 channels.
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Third block pooling: 8x8 (height,width) -> 4x4.
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block: 128 -> 256 channels AND 4x4 -> 2x2.
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Flatten and go into fully connected layers.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),  # 256 channels of 2x2 feature maps -> 512 features.
            nn.ReLU(),
            nn.Dropout(0.3),              # Adding dropout to prevent overfitting.
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)            # Final classification layer: 256 -> 10 classes.
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
