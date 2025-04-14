import torch

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


def evaluate_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient calculation.
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # The outputs, logits, are of shape [batch_size, num_classes].
            predictions = torch.argmax(outputs, dim=1)
            correct += torch.sum( predictions == labels ).item()                    
            total += labels.size(0)
            
            # Store predictions and labels for confusion matrix.
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    
    return all_preds, all_labels, accuracy


def save_confusion_matrix(true_labels, predictions, accuracy, img_name):
    # Confusion Matrix.
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2f}%')
    plt.savefig(img_name)
    #plt.show()


def plot_training_curves(metrics, save_path='training_curves.png'):
    """
    Plot training and validation curves.
    
    Args:
        metrics (dict): Dictionary containing metrics lists
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss.
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_losses'], label='Training Loss')
    plt.plot(metrics['val_losses'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_accuracies'], label='Training Accuracy')
    plt.plot(metrics['val_accuracies'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")