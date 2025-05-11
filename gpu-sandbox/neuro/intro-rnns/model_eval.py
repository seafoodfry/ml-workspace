import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.metrics import confusion_matrix

import torch

from tqdm import tqdm

# Not really training curves but I'm being rushed!
def plot_training_curves(all_losses, save_path):
    plt.figure()
    plt.plot(all_losses, label='Training Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def label_from_output(output, output_labels):
    """
    top_n: The actual highest value (the confidence score/probability)
    top_i: The index position of that highest value
    """
    top_n, top_i = output.topk(1)
    #print(f'{top_n=} - {top_i=}')
    label_i = top_i[0].item()
    return output_labels[label_i], label_i


def save_confusion_matrix(device, model, testing_data, classes, img_name):
    model.eval()

    # Initialize lists to store true labels and predictions.
    true_labels = []
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(len(testing_data))):
            (__label_tensor, text_tensor, label, text) = testing_data[i]
            text_tensor = text_tensor.to(device)

            # Forward pass.
            output = model(text_tensor)
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)

            true_labels.append(label_i)
            predictions.append(guess_i)

    # Generate confusion matrix using sklearn
    cm = confusion_matrix(true_labels, predictions)

    # Normalize by rows
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Set up plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm_normalized)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Add axis labels
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(img_name)