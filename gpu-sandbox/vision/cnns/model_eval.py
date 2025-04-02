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
    print(f'Test Accuracy: {accuracy:.2f}%')
    
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