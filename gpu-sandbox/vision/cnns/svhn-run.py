import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time

from model_eval import evaluate_model, save_confusion_matrix
#from svhn_script_deepercn import DeeperCNN
#from svhn_script_deepcn_batchnorm import DeeperCNN
from svhn_script_even_deepercnn import DeeperCNN


if torch.cuda.is_available():
    _device = 'cuda'
elif torch.backends.mps.is_available():
    _device = 'mps'
else:
    _device = 'cpu'
device = torch.device(_device)
print(f'{device=}')

def prep_data():
    transform = transforms.Compose([
        #transforms.RandomRotation(10),
        #transforms.RandomAffine(0, translate=(0.1, 0.1)),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4377, 0.4438, 0.4728),
            (0.1980, 0.2010, 0.1970),
        )
    ])

    train_dataset = torchvision.datasets.SVHN(
        root='./data', 
        split='train', 
        download=True, 
        transform=transform
    )

    test_dataset = torchvision.datasets.SVHN(
        root='./data', 
        split='test', 
        download=True, 
        transform=transform,
    )
    print(f'{len(train_dataset)=:_}, {len(test_dataset)=:_}')

    batch_size = 512 * 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #, num_workers=os.cpu_count())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, checkpoint_epoch, optimizer_state=None):
    # Move model to device.
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Training loop.
    model.train()
    num_epochs = 60
    last_epoch = checkpoint_epoch
    for epoch in range(num_epochs):
        start = time.perf_counter()
        running_loss = 0.0
        for batch_idx, (features, labels) in enumerate(train_loader):
            if batch_idx == 0:
                print(f'{features.shape=}')
            features, labels = features.to(device), labels.to(device)
            
            # Zero the gradients.
            optimizer.zero_grad()
            
            logits = model(features)
            loss = criterion(logits, labels)

            # Backward pass and optimize.
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print statistics.
        last_epoch += 1
        end = time.perf_counter()
        print(f"Elapsed time: {end - start:.6f} seconds")
        print(f'Epoch {last_epoch}/{num_epochs}, Loss: {running_loss/len(train_loader):.5f}')
        

    #avg_loss = running_loss / len(train_loader)
    model.eval()
    return model, optimizer, last_epoch

if '__name__' == '__name__':
    """
    uv run python cnns/svhn-run.py
    """
    cm_img = 'deepercn_cm.png'
    model_weights = 'deepcnn_model.pth'
    model_checkpoint = 'deepcnn_checkpoint.pth'

    print('Creating model...')
    model = DeeperCNN()
    epoch = 0
    optimizer_state = None
    if os.path.exists(model_weights):
        print(f'found: {model_weights=} and {model_checkpoint=}...')
        model.load_state_dict(torch.load(model_weights, map_location=device))

        checkpoint = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_state = checkpoint['optimizer_state_dict']
        epoch = checkpoint['epoch']
    print('Created model')

    print('Gathering data...')
    train_loader, test_loader = prep_data()
    print('Gathered data')

    print('Training model...')
    model, optimizer, epoch = train_model(model, train_loader, epoch)
    print('Trained model')

    print('Saving model...')
    torch.save(model.state_dict(), model_weights)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, model_checkpoint)
    print('saved model')

    print('Evaluating model on training set...')
    evaluate_model(model, train_loader, device)
    print('Evaluating model on test set...')
    predictions, true_labels, accuracy = evaluate_model(model, test_loader, device)
    save_confusion_matrix(true_labels, predictions, accuracy, cm_img)
