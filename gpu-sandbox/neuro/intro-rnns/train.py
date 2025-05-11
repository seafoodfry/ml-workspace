import random
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm


def train_model(
    device,
    model,
    training_data,
    optimizer,
    num_epochs=60,
    n_batch_size = 64,
    report_every = 50,
    criterion = nn.NLLLoss(),
):
    # Move model to device.
    model = model.to(device)

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    model.train()
    
    print(f"training on data set with n = {len(training_data)}")

    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        model.zero_grad() # clear the gradients

        # create some minibatches
        # we cannot use dataloaders because each of our names is a different length
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) //n_batch_size )

        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch: # for each example in this batch
                (label_tensor, text_tensor, label, text) = training_data[i]
                label_tensor, text_tensor = label_tensor.to(device), text_tensor.to(device)
                
                # Forward pass.
                output = model.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            # Backward pass and optimize.
            # Also add radient clipping to prevent exploding gradients
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),  max_norm=1.0)

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)

        all_losses.append(current_loss / len(batches) )
        if epoch % report_every == 0:
             tqdm.write(f"{epoch} ({epoch / num_epochs:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0

    return all_losses
