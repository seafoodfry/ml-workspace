"""
As per https://paperswithcode.com/paper/multi-digit-number-recognition-from-street#code

Our best architecture consists of eight convolutional hidden layers, one locally connected hidden
layer, and two densely connected hidden layers.

All connections are feedforward and go from one layer to the next (no skip connections).

The first hidden layer contains maxout units (Goodfellow et al., 2013) (with three filters per unit)
while the others contain rectifier units (Jarrett et al., 2009; Glorot et al., 2011).

The number of units at each spatial location in each layer is [48, 64, 128, 160]
for the first four layers and 192 for all other locally connected layers.

The fully connected layers contain 3,072 units each.

Each convolutional layer includes max pooling and subtractive normalization.
The max pooling window size is 2 x 2.
The stride alternates between 2 and 1 at each layer, so that half of the layers don't reduce the
spatial size of the representation.

All convolutions use zero padding on the input to preserve representation size.

The subtractive normalization operates on 3x3 windows and preserves representation size.
All convolution kernels were of size 5 x 5.
We trained with dropout applied to all hidden layers but not the input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubtractiveNorm(nn.Module):
    """
    Subtractive normalization layer that operates on 3x3 windows
    and preserves representation size as specified in the paper.
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size-1) // 2
        
    def forward(self, x):
        # Calculate local mean using average pooling.
        local_mean = F.avg_pool2d(
            x, kernel_size=self.kernel_size, 
            stride=1, padding=self.padding
        )
        # Subtract local mean from the input.
        return x - local_mean

class MaxoutLayer(nn.Module):
    """
    Implementation of Maxout units (Goodfellow et al., 2013)
    with three filters per unit as specified in the paper
    """
    def __init__(self, in_features, out_features, num_pieces=3):
        super().__init__()  # Modern Python super() usage without arguments
        self.in_features = in_features
        self.out_features = out_features
        self.num_pieces = num_pieces
        # Create num_pieces linear transformations for each output
        self.linear = nn.Linear(in_features, out_features * num_pieces)
        
    def forward(self, x):
        # Shape: [batch, out_features * num_pieces]
        x = self.linear(x)
        # Reshape to [batch, out_features, num_pieces]
        x = x.view(-1, self.out_features, self.num_pieces)
        # Take maximum value among the pieces
        x = torch.max(x, dim=2).values  # Modern way to get max values
        return x

class MaxoutConvLayer(nn.Module):
    """First layer with maxout units (3 filters per unit)"""
    def __init__(self, in_channels, out_channels, num_pieces=3):
        super().__init__()
        kernel_size = 5
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(in_channels, out_channels * num_pieces, 
                             kernel_size=kernel_size, padding=padding)
        self.num_pieces = num_pieces
        self.out_channels = out_channels
        
    def forward(self, x):
        x = self.conv(x)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, self.out_channels, self.num_pieces, height, width)
        x = torch.max(x, dim=2).values
        return x


class SVHNNet(nn.Module):
    def __init__(self, num_classes=10):
        """
        Fully Connected Layer: Every output connects to EVERY input with UNIQUE weights
            (Maximum parameters, no spatial structure preserved)
                       
        Convolutional Layer: Every output connects to LOCAL inputs with SHARED weights
            (Minimum parameters, spatial structure preserved)
                            
        Locally Connected: Every output connects to LOCAL inputs with UNIQUE weights
            (Many parameters, spatial structure preserved)

        Locally Connected Layer: A locally connected layer is similar to a convolutional layer
          but without weight sharing. In a convolutional layer, the same filter is applied across
          the entire input (weights are shared). In a locally connected layer, different weights
          are used at each spatial location, but it still operates on local regions like a convolution.
          This increases the number of parameters but allows the layer to learn position-specific
          features.

        Densely Connected Hidden Layers: These are standard fully connected (or "dense") layers where
          every neuron is connected to every neuron in the previous layer. In the paper,
          they mention having two of these layers with 3,072 units each.
        """
        super().__init__()
        
        # First layer: Conv + Maxout
        self.first_layer = nn.Sequential(
            MaxoutConvLayer(3, 48, num_pieces=3),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reducing dimension
            SubtractiveNorm(kernel_size=3),
            nn.Dropout(0.5)
        )
        
        # Main convolutional blocks with properly alternating spatial reduction
        self.features = nn.Sequential(
            # Layer 2 - Maintains spatial dimensions
            nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),  # No reduction
            SubtractiveNorm(kernel_size=3),
            nn.Dropout(0.3),
            
            # Layer 3 - Reduces spatial dimensions
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduction
            SubtractiveNorm(kernel_size=3),
            nn.Dropout(0.3),
            
            # Layer 4 - Maintains spatial dimensions
            nn.Conv2d(128, 160, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),  # No reduction
            SubtractiveNorm(kernel_size=3),
            nn.Dropout(0.3),
            
            # Layer 5 - Reduces spatial dimensions
            nn.Conv2d(160, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduction
            SubtractiveNorm(kernel_size=3),
            nn.Dropout(0.3),
            
            # Layer 6 - Maintains spatial dimensions
            nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),  # No reduction
            SubtractiveNorm(kernel_size=3),
            nn.Dropout(0.3),
            
            # Layer 7 - Reduces spatial dimensions
            nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduction
            SubtractiveNorm(kernel_size=3),
            nn.Dropout(0.3),
            
            # Layer 8 - Maintains spatial dimensions
            nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),  # No reduction
            SubtractiveNorm(kernel_size=3),
            nn.Dropout(0.3)
        )
        
        # Calculate the final spatial dimensions (2x2 for SVHN with 32x32 input)
        spatial_dim = 2
        
        # Classification layers.
        # 1 locally connected layer (which is essentially the flattening of the final convolutional
        # features).
        # 2 fully connected layers with 3,072 units each before the output layer.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * spatial_dim * spatial_dim, 3072),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(3072, 3072),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(3072, num_classes)
        )
    
    def forward(self, x):
        x = self.first_layer(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
    

# Attempt to literal locally connected layer.
class LocallyConnectedLayer(nn.Module):
    """
    See:
    1. https://discuss.pytorch.org/t/locally-connected-layers/26979/20
    2. https://github.com/ptrblck/pytorch_misc/blob/master/LocallyConnected2d.py
    3. https://stackoverflow.com/questions/59455386/local-fully-connected-layer-pytorch
    4. https://www.cs.toronto.edu/~lczhang/aps360_20191/lec/w03/convnet.html
    
    """
    def __init__(self, in_channels, out_channels, input_height, input_width, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_height = input_height
        self.input_width = input_width
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Calculate output dimensions
        self.output_height = input_height
        self.output_width = input_width
        
        # Create a weight tensor for each location
        # This is where the "local" part comes in - different weights for each position
        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels * kernel_size * kernel_size,
                input_height * input_width
            ) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, input_height * input_width))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract local patches using unfold
        x_unfolded = F.unfold(
            x, kernel_size=self.kernel_size, 
            padding=self.padding
        )
        
        # Reshape for locally connected operation
        x_unfolded = x_unfolded.view(
            batch_size, 
            self.in_channels * self.kernel_size * self.kernel_size, 
            self.input_height * self.input_width
        )
        
        # Apply different weights to each location
        out = torch.bmm(
            self.weight.expand(batch_size, -1, -1, -1),
            x_unfolded
        )
        
        # Add bias
        out = out + self.bias
        
        # Reshape to proper output shape
        out = out.view(
            batch_size, 
            self.out_channels, 
            self.output_height, 
            self.output_width
        )
        
        return out