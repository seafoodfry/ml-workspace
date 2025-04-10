"""
uv run python cnns/digit-run.py
"""
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import scipy.io as sio
import numpy as np
import os
import time

from model_eval import evaluate_model, save_confusion_matrix
from svhn_script_digit_detector import DigitDetector


if torch.cuda.is_available():
    _device = 'cuda'
elif torch.backends.mps.is_available():
    _device = 'mps'
else:
    _device = 'cpu'
device = torch.device(_device)
print(f'{device=}')


class NegativeSamplesDataset(Dataset):
    def __init__(self, mat_file, transform=None):
        """
        Dataset for loading negative samples from a .mat file
        
        Args:
            mat_file (str): Path to the .mat file containing negative samples
            transform: Optional transforms to apply to the samples
        """
        self.transform = transform

        # Load the data from the .mat file.
        mat_data = sio.loadmat(mat_file)

        # X has shape (32, 32, 3, n_samples) - need to transpose.
        self.X = mat_data['X'].transpose(3, 2, 0, 1)  # Now (n_samples, 3, 32, 32).
        self.y = mat_data['y'].flatten()  # Labels.

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Get the image and label at the given index.
        image = self.X[idx].astype('float32') / 255.0  # Normalize to [0, 1].
        label = int(self.y[idx])

        image_tensor = torch.from_numpy(image)
        image_tensor = self.transform(image_tensor)
            
        return image_tensor, label
    

class BinaryDigitDataset(Dataset):
    def __init__(self, dataset, is_digit=True, transform=None):
        self.dataset = dataset
        self.is_digit = is_digit
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, original_label = self.dataset[idx]
        # Convert to binary label: 1 for digit, 0 for non-digit
        binary_label = 1 if self.is_digit else 0
        
        return image, binary_label


def prep_data(negative_mat_file='./cnns/negative_samples.mat'):
    transform = transforms.Compose([
        #transforms.RandomRotation(10),
        #transforms.RandomAffine(0, translate=(0.1, 0.1)),
        #transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4377, 0.4438, 0.4728),
            (0.1980, 0.2010, 0.1970),
        )
    ])
    negative_data_transform = transforms.Compose([
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
    print(f'{len(train_dataset)=:_}, {len(test_dataset)=:_}')

    negative_dataset = NegativeSamplesDataset(
        negative_mat_file,
        transform=negative_data_transform
    )

    # Convert to binary datasets.
    binary_svhn_train = BinaryDigitDataset(train_dataset, is_digit=True)
    binary_svhn_test = BinaryDigitDataset(test_dataset, is_digit=True)
    binary_negative_train = BinaryDigitDataset(negative_dataset, is_digit=False)

    # Split the negative samples for training and testing.
    # Use random_split to split the negative dataset.
    neg_size = len(negative_dataset)
    train_size = int(0.3 * neg_size)
    test_size = neg_size - train_size
    neg_train, neg_test = torch.utils.data.random_split(
        binary_negative_train, [train_size, test_size]
    )

    # Combine datasets for training and testing.
    binary_train_dataset = ConcatDataset([binary_svhn_train, neg_train])
    binary_test_dataset = ConcatDataset([binary_svhn_test, neg_test])
    
    print(f'Training set: {len(binary_svhn_train):_} digits + {len(neg_train):_} non-digits')
    print(f'Test set: {len(binary_svhn_test):_} digits + {len(neg_test):_} non-digits')
    

    #batch_size = 512 * 2
    batch_size = 32
    train_loader = DataLoader(binary_train_dataset, batch_size=batch_size, shuffle=True) #, num_workers=os.cpu_count())
    test_loader = DataLoader(binary_test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, checkpoint_epoch, optimizer_state=None):
    # Move model to device.
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Training loop.
    model.train()
    num_epochs = 3
    last_epoch = checkpoint_epoch
    for epoch in range(num_epochs):
        start = time.perf_counter()
        running_loss = 0.0
        for batch_idx, (features, labels) in enumerate(train_loader):
            if batch_idx == 0:
                print(f'{features.shape=}')
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
        last_epoch += 1
        end = time.perf_counter()
        print(f"Elapsed time: {end - start:.6f} seconds")
        print(f'Epoch {last_epoch}/{num_epochs}, Loss: {running_loss/len(train_loader):.5f}')
        

    #avg_loss = running_loss / len(train_loader)
    model.eval()
    return model, optimizer, last_epoch

if '__name__' == '__name__':
    """
    uv run python cnns/digit-run.py
    """
    cm_img = 'digit_cm.png'
    model_weights = 'digit_model.pth'
    model_checkpoint = 'digit_checkpoint.pth'

    print('Creating model...')
    model = DigitDetector()
    epoch = 0
    optimizer_state = None
    if os.path.exists(model_weights):
        print(f'found: {model_weights=} and {model_checkpoint=}...')
        model.load_state_dict(torch.load(model_weights, map_location=device))

        checkpoint = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_state = checkpoint['optimizer_state_dict']
        epoch = checkpoint['epoch']
    print('Created model')

    print('Gathering data...')
    train_loader, test_loader = prep_data()
    print('Gathered data')

    print('Training model...')
    model, optimizer, epoch = train_model(model, train_loader, epoch)
    print('Trained model')

    print('Saving model...')
    torch.save(model.state_dict(), model_weights)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, model_checkpoint)
    print('saved model')

    print('Evaluating model on training set...')
    evaluate_model(model, train_loader, device)
    print('Evaluating model on test set...')
    predictions, true_labels, accuracy = evaluate_model(model, test_loader, device)
    save_confusion_matrix(true_labels, predictions, accuracy, cm_img)
