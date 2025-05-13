import random
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm


def train_model(
    device,
    model,
    train_loader,
    optimizer,
    num_epochs=60,
    report_every = 50,
    criterion = nn.NLLLoss(),
):
    # Move model to device.
    model = model.to(device)

    # Keep track of losses for plotting
    all_losses = []
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        #model.zero_grad() # clear the gradients
        current_loss = 0

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

        # Average loss for this epoch
        epoch_loss = current_loss / len(train_loader)
        all_losses.append(epoch_loss)
        if epoch % report_every == 0:
             tqdm.write(f"{epoch} ({epoch / num_epochs:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0

    return all_losses
