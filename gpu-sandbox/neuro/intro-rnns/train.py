import random
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm


def train_model(
    device, model,
    train_loader, val_loader,
    optimizer, scheduler,
    num_epochs=60,
    report_every = 50,
    criterion = nn.NLLLoss(),
):
    # Move model to device.
    model = model.to(device)

    # Keep track of losses for plotting
    all_losses = []
    # Store metrics.
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        #model.zero_grad() # clear the gradients
        current_loss = 0
        correct = 0
        total = 0

        for labels, sequences, seq_lengths, _, _ in train_loader:
            labels = labels.to(device)
            sequences = sequences.to(device)
            #seq_lengths = seq_lengths.to(device)

            # Forward pass.
            # Note: the example we used in the tutorial did model.forward(...)
            # but calling forward directly does not take into account the model's
            # current mode (train vs. eval), which is important if the modelf uses
            # things like dropout or batchnorm (and the idiomatic thing figures out
            # whetehr to call model.eval() or model.train()).
            optimizer.zero_grad()
            outputs = model(sequences, seq_lengths)

            # Compute loss.
            loss = criterion(outputs, labels)

            # Backward pass and optimize.
            # Also add radient clipping to prevent exploding gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),  max_norm=1.0)
            optimizer.step()  # Update parameters

            current_loss += loss.item()

            # Calculate accuracy.
            # Pass the logics (outputs).
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # If using OneCycleLR, call scheduler.step() after each batch
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

        # Average loss for this epoch
        epoch_loss = current_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)


        # Validation phase.
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # No need to track gradients.
            for labels, sequences, seq_lengths, _, _ in val_loader:
                labels = labels.to(device)
                sequences = sequences.to(device)

                logits = model(sequences, seq_lengths)
                loss = criterion(logits, labels)

                val_running_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # If using ReduceLROnPlateau, call scheduler.step() after validation
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        if epoch % report_every == 0:
             tqdm.write(f"{epoch} ({epoch / num_epochs:.0%}): {scheduler.get_last_lr()=} train loss = {train_losses[-1]:.4f}, train acc = {train_accuracy:.2f}%, val loss = {val_loss:.4f}, val acc = {val_accuracy:.2f}%")
        current_loss = 0

    # Save metrics for plotting.
    metrics = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
    model.eval()
    return metrics
