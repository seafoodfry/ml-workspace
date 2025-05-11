import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

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
    top_n, top_i = output.topk(1)
    #print(f'{top_n=} - {top_i=}')
    label_i = top_i[0].item()
    return output_labels[label_i], label_i


def save_confusion_matrix(device, model, testing_data, classes, img_name):
    model.eval()

    confusion = torch.zeros(len(classes), len(classes))

    with torch.no_grad(): # do not record the gradients during eval phase
        for i in tqdm(range(len(testing_data))):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            text_tensor = text_tensor.to(device)

            output = model(text_tensor)
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(len(classes)):
        denom = confusion[i].sum()
        if denom > 0:
            confusion[i] = confusion[i] / denom

    # Set up plot.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(img_name)