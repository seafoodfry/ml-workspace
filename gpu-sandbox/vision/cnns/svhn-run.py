import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from model_eval import evaluate_model, save_confusion_matrix
from svhn_script_deepercn import DeeperCNN


if torch.cuda.is_available():
    _device = 'cuda'
elif torch.backends.mps.is_available():
    _device = 'mps'
else:
    _device = 'cpu'
device = torch.device(_device)


def prep_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4377, 0.4438, 0.4728),
            (0.1980, 0.2010, 0.1970),
        )
    ])

    train_dataset = torchvision.datasets.SVHN(
        root='./data', 
        split='train', 
        download=True, 
        transform=transform
    )

    test_dataset = torchvision.datasets.SVHN(
        root='./data', 
        split='test', 
        download=True, 
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #, num_workers=os.cpu_count())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader):
    # Move model to device.
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop.
    model.train()
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            # Zero the gradients.
            optimizer.zero_grad()
            
            logits = model(features)
            loss = criterion(logits, labels)

            # Backward pass and optimize.
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print statistics.
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.5f}')
        

    model.eval()
    return model

if '__name__' == '__name__':
    """
    uv run python cnns/svhn-run.py
    """
    print('Creating model...')
    model = DeeperCNN()
    print('Created model')

    print('Gathering data...')
    train_loader, test_loader = prep_data()
    print('Gathered data')

    print('Training model...')
    model = train_model(model, train_loader)
    print('Trained model')

    print('Evaluating model...')
    predictions, true_labels, accuracy = evaluate_model(model, test_loader, device)
    save_confusion_matrix(true_labels, predictions, accuracy, 'deepercn_cm.png')
