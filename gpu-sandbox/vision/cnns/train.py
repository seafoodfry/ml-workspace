import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import scipy.io as sio
import numpy as np
import os
import time
from tqdm import tqdm


def train_model(
        device, model, train_loader, val_loader,
        checkpoint_epoch,
        optimizer,
        patience=10, patience_multiplier=0.5, num_epochs=60,
        checkpoint_file='model_checkpoint.pth',
        best_model_file='best_model.pth',
        ):
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
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    #if optimizer_state_dict is not None:
    #    optimizer.load_state_dict(optimizer_state_dict)

    # Learning rate scheduler.
    # When the validation loss doesn't improve for 5 consecutive epochs, the scheduler reduces
    # the learning rate (by multiplying it by the factor of 0.5). This helps the model navigate 
    # flat regions or small local minima in the loss landscape by making smaller steps, potentially
    # finding better solutions.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience//2, factor=patience_multiplier)
   
    # Early stopping variables.
    best_val_loss = float('inf')
    best_epoch = -1 
    best_model_state = None
    early_stop_counter = 0

    # Store metrics.
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Training loop.
    last_epoch = checkpoint_epoch
    for epoch in tqdm(range(num_epochs)):
        #start = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
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

        # Early stopping check: we want to track how many iterations the validation loss has not
        # improved to see if it is greater than our patience.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
            best_epoch = last_epoch

            # Save best model separately when it's found
            best_checkpoint = {
                'epoch': last_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(best_checkpoint, best_model_file)
            tqdm.write(f'New best model saved at epoch {best_epoch}')
        else:
            early_stop_counter += 1

        # Print statistics.
        last_epoch += 1
        #end = time.perf_counter()
        #print(f"Elapsed time: {end - start:.6f} seconds")
        #print(f'Epoch {last_epoch}/{num_epochs}, Loss: {running_loss/len(train_loader):.5f}')
        #print(f'{scheduler.get_last_lr()=}')
        tqdm.write(f'{scheduler.get_last_lr()=}, Loss: {running_loss/len(train_loader):.5f}')

        # Save checkpoint.
        checkpoint = {
            'epoch': last_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # Add this line
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,  # Add this line
            'best_epoch': best_epoch  # Add this line
        }
        torch.save(checkpoint, checkpoint_file)

        # Early stopping.
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {last_epoch} epochs")
            # Load best model
            model.load_state_dict(best_model_state)
            break

    # Add this at the end of the function before return
    print(f'Training complete. Best model was from epoch {best_epoch} with val_loss: {best_val_loss:.5f}')
    # Load the best model for evaluation
    if os.path.exists(best_model_file):
        best_checkpoint = torch.load(best_model_file)
        model.load_state_dict(best_checkpoint['model_state_dict'])
    else:
        print('Warning: Best model file not found. Using latest model.')
        
    # Save metrics for plotting.
    metrics = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

    model.eval()
    return model, optimizer, last_epoch, metrics