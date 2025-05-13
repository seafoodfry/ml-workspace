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


def save_confusion_matrix(device, model, test_loader, classes, img_name):
    model.eval()

    # Initialize lists to store true labels and predictions.
    true_labels = []
    predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        for labels, sequences, seq_lengths, orig_labels, _ in tqdm(test_loader):
            labels = labels.to(device)
            sequences = sequences.to(device)

            # Forward pass with batch processing.
            outputs = model(sequences, seq_lengths)  # Shape: [batch_size, num_classes]

            # Get predicted class indices.
            _, predicted = torch.max(outputs, 1)   # Shape: [batch_size]

            # Add to lists for confusion matrix.
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

            #correct += (predicted == labels).sum().item()
            correct += torch.sum(predicted == labels).item()
            total += labels.size(0)

            # If you need the actual class names instead of indices
            # (labels_tensors are indices).
            # true_class_names.extend([classes[idx] for idx in labels.cpu().numpy()])
            # predicted_class_names.extend([classes[idx] for idx in predicted.cpu().numpy()])

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

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

    # Add axis labels.
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2f}%')

    plt.tight_layout()
    plt.savefig(img_name)