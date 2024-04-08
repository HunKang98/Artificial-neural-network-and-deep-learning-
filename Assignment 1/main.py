
import dataset
from model import LeNet5, CustomMLP
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# import some packages you need here


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """
    model.train()
    total_loss, total_acc = 0.0, 0.0
    n = len(trn_loader.dataset)

    for X, y in trn_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = pred.max(1)
        total_acc += predicted.eq(y).sum().item()

    trn_loss = total_loss / n
    acc = total_acc / n

    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    n = len(tst_loader.dataset)

    with torch.no_grad():
        for X, y in tst_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)

            total_loss += loss.item()
            _, predicted = pred.max(1)
            total_acc += predicted.eq(y).sum().item()

    tst_loss = total_loss / n
    acc = total_acc / n

    return tst_loss, acc


def main():
    """ Main function
    
        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    trn_dataset = dataset.MNIST(data_dir='./train.tar', augmentation=False)
    trn_dataset2 = dataset.MNIST(data_dir='./train.tar', augmentation=True)
    tst_dataset = dataset.MNIST(data_dir='./test.tar', augmentation=False)

    trn_loader = DataLoader(dataset=trn_dataset, batch_size=300, shuffle=True)
    trn_loader2 = DataLoader(dataset=trn_dataset2, batch_size=300, shuffle=True)
    tst_loader = DataLoader(dataset=tst_dataset, batch_size=300, shuffle=True)
    
    
    #train LeNet5 - No regularization
    lenet_model = LeNet5(dropout=False).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lenet_model.parameters(), lr=1e-3)

    epochs = 30
    trn_loss_li = []
    trn_acc_li = []
    tst_loss_li = []
    tst_acc_li = []

    for epoch in range(epochs):
        trn_loss, trn_acc = train(model=lenet_model, trn_loader=trn_loader, device=device, criterion=criterion, optimizer=optimizer)
        tst_loss, tst_acc = test(model=lenet_model, tst_loader=tst_loader, device=device, criterion=criterion)

        trn_loss_li.append(trn_loss)
        trn_acc_li.append(trn_acc)
        tst_loss_li.append(tst_loss)
        tst_acc_li.append(tst_acc)

        print(f'Epoch {epoch}, Train Loss: {trn_loss}, Train Accuracy: {trn_acc}, Test Loss: {tst_loss}, Test Accuracy: {tst_acc}')

    #Plot loss and accuracy
    plt.figure(figsize=(12,8))
    plt.plot(trn_loss_li, label = 'Training loss')
    plt.plot(tst_loss_li, label = 'Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./lenet5_noregul_loss.png')

    plt.figure(figsize=(12,8))
    plt.plot(trn_acc_li, label = 'Training acc')
    plt.plot(tst_acc_li, label = 'Test acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./lenet5_noregul_acc.png')
    
    """
    #train LeNet5 - With regularization
    lenet_model2 = LeNet5(dropout=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lenet_model2.parameters(), lr=1e-3)

    epochs = 30
    trn_loss_li = []
    trn_acc_li = []
    tst_loss_li = []
    tst_acc_li = []

    for epoch in range(epochs):
        trn_loss, trn_acc = train(model=lenet_model2, trn_loader=trn_loader2, device=device, criterion=criterion, optimizer=optimizer)
        tst_loss, tst_acc = test(model=lenet_model2, tst_loader=tst_loader, device=device, criterion=criterion)

        trn_loss_li.append(trn_loss)
        trn_acc_li.append(trn_acc)
        tst_loss_li.append(tst_loss)
        tst_acc_li.append(tst_acc)

        print(f'Epoch {epoch}, Train Loss: {trn_loss}, Train Accuracy: {trn_acc}, Test Loss: {tst_loss}, Test Accuracy: {tst_acc}')

    #Plot loss and accuracy
    plt.figure(figsize=(12,8))
    plt.plot(trn_loss_li, label = 'Training loss')
    plt.plot(tst_loss_li, label = 'Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./lenet5_regul_loss.png')

    plt.figure(figsize=(12,8))
    plt.plot(trn_acc_li, label = 'Training acc')
    plt.plot(tst_acc_li, label = 'Test acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./lenet5_regul_acc.png')
    
    
    #train custom MLP
    mlp = CustomMLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

    epochs = 30
    trn_loss_li = []
    trn_acc_li = []
    tst_loss_li = []
    tst_acc_li = []

    for epoch in range(epochs):
        trn_loss, trn_acc = train(model=mlp, trn_loader=trn_loader, device=device, criterion=criterion, optimizer=optimizer)
        tst_loss, tst_acc = test(model=mlp, tst_loader=tst_loader, device=device, criterion=criterion)

        trn_loss_li.append(trn_loss)
        trn_acc_li.append(trn_acc)
        tst_loss_li.append(tst_loss)
        tst_acc_li.append(tst_acc)

        print(f'Epoch {epoch}, Train Loss: {trn_loss}, Train Accuracy: {trn_acc}, Test Loss: {tst_loss}, Test Accuracy: {tst_acc}')

    #Plot loss and accuracy
    plt.figure(figsize=(12,8))
    plt.plot(trn_loss_li, label = 'Training loss')
    plt.plot(tst_loss_li, label = 'Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./custommlp_loss.png')

    plt.figure(figsize=(12,8))
    plt.plot(trn_acc_li, label = 'Training acc')
    plt.plot(tst_acc_li, label = 'Test acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./custommlp_acc.png')
    """

if __name__ == '__main__':
    main()
