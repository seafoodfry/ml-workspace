"""

"""
import torch.nn as nn


class DigitDetector(nn.Module):
    def __init__(self):
        """
        See svhn-workbench.ipynb for more details about the architecture.

        See svhn_script_even_deepercnn.py for mroe details about the architecture.
        """
        super().__init__()

        kernel_size = 3
        padding = (kernel_size - 1) // 2
        print(f'CNN: using {kernel_size=} and {padding=}')

        self.features = nn.Sequential(
            # First block conv: 3 (RGB) -> 32 channels.
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # First block pooling: 32x32 (height,width) -> 16x16x.
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            # Second block conv: 32 channels -> 64 channels.
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Second block pooling: 16x16 (height,width) -> 8x8.
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            # Third block conv: 64 channels -> 128 channels.
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Third block pooling: 8x8 (height,width) -> 4x4.
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            # Fourth block: 128 -> 256 channels AND 4x4 -> 2x2.
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
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
            nn.Linear(256, 2)            # Final classification layer: 256 -> 10 classes.
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
