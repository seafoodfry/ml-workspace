import string
import unicodedata
from io import open
import glob
import os
import time

import torch
from torch.utils.data import Dataset


from tqdm import tqdm

from train import train_model
from model_simple import CharRNN
from model_eval import save_confusion_matrix, plot_training_curves



if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


# We can use "_" to represent an out-of-vocabulary character, that is, any character we are not handling in our model
ALLOWED_CHARACTERS = string.ascii_letters + " .,;'" + "_"
N_LETTERS = len(ALLOWED_CHARACTERS)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALLOWED_CHARACTERS
    )

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    # return our out-of-vocabulary character if we encounter a letter unknown to our model
    if letter not in ALLOWED_CHARACTERS:
        return ALLOWED_CHARACTERS.find("_")
    else:
        return ALLOWED_CHARACTERS.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


class NamesDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir #for provenance of the dataset
        self.load_time = time.localtime #for provenance of the dataset
        labels_set = set() #set of all classes

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        #read all the ``.txt`` files in the specified directory
        text_files = glob.glob(os.path.join(data_dir, '*.txt'))
        for filename in tqdm(text_files):
            # Separates into <language> <.txt>.
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            for name in lines:
                self.data.append(name)
                self.data_tensors.append(lineToTensor(name))
                self.labels.append(label)

        #Cache the tensor representation of the labels
        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # The name as is.
        data_item = self.data[idx]
        # The name of the file (w/o extension).
        data_label = self.labels[idx]
        # Name in tensor form.
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]

        return label_tensor, data_tensor, data_label, data_item
    
def prep_data(data_dir: str):
    alldata = NamesDataset(data_dir)

    train_size = int(0.85 * len(alldata))
    test_size = len(alldata) - train_size
    train_set, test_set = torch.utils.data.random_split(
        alldata,
        [train_size, test_size],
        #generator=torch.Generator(device=device).manual_seed(2024),
    )

    return train_set, test_set, alldata

if __name__ == '__main__':
    """
    uv run python run.py
    """
    # For reproducibility.
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_weights = 'charrnn_simple_model.pth'
    model_checkpoint = 'charrnn_simple_checkpoint.pth'
    curves_img = 'charrnn_simple-training_curves.png'
    cm_img = 'charrnn_simple_cm.png'

    print('Gathering data...')
    train_set, test_set, alldata = prep_data(data_dir='data/names')
    print('Gathered data')

    print('Creating model...')
    # 8 input nodes, 128 hidden nodes, and 18 outputs.
    n_hidden = 128
    model = CharRNN(N_LETTERS, n_hidden, len(alldata.labels_uniq))
    print('Created model')

    print('Training model...')
    num_epochs=1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    all_losses = train_model(
        device, model, train_set,
        optimizer,
        num_epochs=num_epochs,
    )
    print('Trained model')

    print('Saving model...')
    torch.save(model.state_dict(), model_weights)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
    }, model_checkpoint)
    print('saved model')

    print('Doing analysis...')
    plot_training_curves(all_losses, curves_img)
    save_confusion_matrix(device, model, test_set, classes=alldata.labels_uniq, img_name=cm_img)
