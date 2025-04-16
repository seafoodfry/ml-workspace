import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import scipy.io as sio
import numpy as np
import os
import time

from model_eval import evaluate_model, save_confusion_matrix, plot_training_curves
from finetune_model import CNN


if torch.cuda.is_available():
    _device = 'cuda'
elif torch.backends.mps.is_available():
    _device = 'mps'
else:
    _device = 'cpu'
device = torch.device(_device)
print(f'{device=}')


class SVHNWithNegativesDataset(Dataset):
    def __init__(self, svhn_dataset, negative_dataset, transform=None):
        self.svhn_dataset = svhn_dataset
        self.negative_dataset = negative_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.svhn_dataset) + len(self.negative_dataset)
    
    def __getitem__(self, idx):
        if idx < len(self.svhn_dataset):
            # This is an SVHN sample
            image, label = self.svhn_dataset[idx]
            return image, label
        else:
            # This is a negative sample - use index 10 for "not a digit"
            neg_idx = idx - len(self.svhn_dataset)
            image, _ = self.negative_dataset[neg_idx]
            return image, 10  # 10 is our "not a digit" class


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


def prep_data(negative_mat_file='./cnns/negative_samples_finetune.mat'):
    transform = transforms.Compose([
        #transforms.RandomRotation(10),
        #transforms.RandomAffine(0, translate=(0.1, 0.1)),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    
    negative_dataset = NegativeSamplesDataset(
        negative_mat_file,
        transform=negative_data_transform
    )

    # Create validation set from training data (10% of training data).
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Split negative samples for train/val/test.
    neg_size = len(negative_dataset)
    neg_train_size = int(0.7 * neg_size)
    neg_val_size = int(0.1 * neg_size)
    neg_test_size = neg_size - (neg_train_size + neg_val_size)
    neg_train, neg_val, neg_test = torch.utils.data.random_split(
        negative_dataset, [neg_train_size, neg_val_size, neg_test_size]
    )

    # Create combined datasets.
    combined_train = SVHNWithNegativesDataset(train_dataset, neg_train)
    combined_val = SVHNWithNegativesDataset(val_dataset, neg_val)
    combined_test = SVHNWithNegativesDataset(test_dataset, neg_test)
    print(f'Training set: {len(train_dataset):_} digits + {len(neg_train):_} non-digits')
    print(f'Validation set: {len(val_dataset):_} digits + {len(neg_val):_} non-digits')
    print(f'Test set: {len(test_dataset):_} digits + {len(neg_test):_} non-digits')
    

    batch_size = 512 * 2
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True) #, num_workers=os.cpu_count())
    val_loader = DataLoader(combined_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(combined_test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, checkpoint_epoch, optimizer_state=None, patience=10, num_epochs=60):
    """
    - Added a learning rate scheduler (ReduceLROnPlateau) to adjust the learning rate when
      validation loss plateaus
    - Implemented early stopping to prevent overfitting
    - Added gradient clipping to prevent exploding gradients

    Training will completely stop if the validation loss doesn't improve for 10 consecutive epochs.
    This helps prevent overfitting by stopping training when the model stops generalizing better to
    the validation data.
    """
    # Move model to device.
    model = model.to(device)

    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Learning rate scheduler.
    # When the validation loss doesn't improve for 5 consecutive epochs, the scheduler reduces
    # the learning rate (by multiplying it by the factor of 0.5). This helps the model navigate 
    # flat regions or small local minima in the loss landscape by making smaller steps, potentially
    # finding better solutions.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
   
    # Early stopping variables.
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0

    # Store metrics.
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Training loop.
    last_epoch = checkpoint_epoch
    for epoch in range(num_epochs):
        start = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            if batch_idx == 0:
                print(f'{features.shape=}')
            features, labels = features.to(device), labels.to(device)
            
            # Zero the gradients.
            optimizer.zero_grad()
            
            # Forward pass.
            logits = model(features)
            loss = criterion(logits, labels)

            # Backward pass and optimize.
            # Also add radient clipping to prevent exploding gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy.
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase.
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # No need to track gradients.
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                logits = model(features)
                loss = criterion(logits, labels)
                
                val_running_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Update learning rate based on validation loss.
        scheduler.step(val_loss)

        # Early stopping check.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Print statistics.
        last_epoch += 1
        end = time.perf_counter()
        print(f"Elapsed time: {end - start:.6f} seconds")
        print(f'Epoch {last_epoch}/{num_epochs}, Loss: {running_loss/len(train_loader):.5f}')
        print(f'{scheduler.get_last_lr()=}')

        # Save checkpoint
        checkpoint = {
            'epoch': last_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(checkpoint, 'model_checkpoint.pth')

        # Early stopping.
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {last_epoch} epochs")
            # Load best model
            model.load_state_dict(best_model_state)
            break

    # If training completed without early stopping, use the best model.
    if early_stop_counter < patience and best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save the model before exiting.
    torch.save(checkpoint, 'model_checkpoint.pth')

    # Save metrics for plotting.
    metrics = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

    model.eval()
    return model, optimizer, last_epoch, metrics


if '__name__' == '__name__':
    """
    uv run python cnns/finetune-svhn-run.py

    scp ubuntu@${EC2}:/home/ubuntu/src/deepcnn_checkpoint.pth ./vision/cnns/trained

    scp ubuntu@${EC2}:/home/ubuntu/src/deepcnn_model.pth ./vision/cnns/trained

    scp ubuntu@${EC2}:/home/ubuntu/src/deepcnn-training_curves.png .

    scp ubuntu@${EC2}:/home/ubuntu/src/deepcnn_cm.png .
    """
    # For reproducibility.
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)

    cm_img = 'deepcnn_cm.png'
    model_weights = 'deepcnn_model.pth'
    model_checkpoint = 'deepcnn_checkpoint.pth'
    curves_img = 'deepcnn-training_curves.png'

    print('Creating model...')
    model = CNN()
    epoch = 0
    optimizer_state = None
    if os.path.exists(model_checkpoint):
        print(f'found: {model_weights=} and {model_checkpoint=}...')
        #model.load_state_dict(torch.load(model_weights, map_location=device))
        checkpoint = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_state = checkpoint['optimizer_state_dict']
        epoch = checkpoint['epoch']
    print('Created model')

    print('Gathering data...')
    train_loader, val_loader, test_loader = prep_data(negative_mat_file='./cnns/negative_samples_finetune.mat')
    print('Gathered data')

    print('Training model...')
    model, optimizer, epoch, metrics = train_model(
        model, train_loader, val_loader, epoch, optimizer_state,
        num_epochs=50,
    )
    print('Trained model')

    print('Saving model...')
    torch.save(model.state_dict(), model_weights)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, model_checkpoint)
    print('saved model')

    print('Plotting training curves...')
    plot_training_curves(metrics, curves_img)
    print('Plotted training curves')

    print('Evaluating model on training set...')
    evaluate_model(model, train_loader, device)
    print('Evaluating model on test set...')
    predictions, true_labels, accuracy = evaluate_model(model, test_loader, device)
    save_confusion_matrix(true_labels, predictions, accuracy, cm_img)
