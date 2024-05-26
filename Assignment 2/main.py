# import some packages you need here
import dataset
from model import CharRNN, CharLSTM
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt


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
    """

    # write your codes here
    model.train()
    total_loss = 0.0
    n = len(trn_loader.dataset)

    for X, y in trn_loader:
        X, y = X.to(device), y.to(device)
        hidden = model.init_hidden(len(X))

        if isinstance(hidden, tuple):
            hidden = tuple(h.to(device) for h in hidden)
        else:
            hidden = hidden.to(device)

        pred, _ = model(X, hidden)
        pred = pred.permute(0,2,1)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+= loss.item()

    trn_loss = total_loss / n

    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    # write your codes here
    model.eval()
    total_loss = 0.0
    n = len(val_loader.dataset)

    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        hidden = model.init_hidden(len(X))

        if isinstance(hidden, tuple):
            hidden = tuple(h.to(device) for h in hidden)
        else:
            hidden = hidden.to(device)

        pred, _ = model(X, hidden)
        pred = pred.permute(0,2,1)
        loss = criterion(pred, y)

        total_loss += loss.item()

    val_loss = total_loss / n

    return val_loss


def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    with torch.no_grad():
        torch.cuda.empty_cache()
    # write your codes here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    total_dataset = dataset.Shakespeare(input_file='./shakespeare_train.txt')
    dataset_size = len(total_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_idices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idices)
    valid_sampler = SubsetRandomSampler(val_indices)

    trn_loader = DataLoader(total_dataset, batch_size=100, sampler= train_sampler)
    val_loader = DataLoader(total_dataset, batch_size=100, sampler= valid_sampler)

    rnn_model = CharRNN().to(device)
    lstm_model = CharLSTM().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=5e-3)    
    optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=5e-3)    

    epochs = 30
    rnn_best = 1000
    lstm_best = 1000
    rnn_trn_loss_li = []
    rnn_val_loss_li = []
    lstm_trn_loss_li = []
    lstm_val_loss_li = []

    for epoch in range(epochs):
        rnn_trn_loss = train(model=rnn_model, trn_loader=trn_loader, device=device, criterion=criterion, optimizer=optimizer_rnn)
        rnn_val_loss = validate(model=rnn_model, val_loader=val_loader, device=device, criterion=criterion)
        lstm_trn_loss = train(model=lstm_model, trn_loader=trn_loader, device=device, criterion=criterion, optimizer=optimizer_lstm)
        lstm_val_loss = validate(model=lstm_model, val_loader=val_loader, device=device, criterion=criterion)

        rnn_trn_loss_li.append(rnn_trn_loss)
        rnn_val_loss_li.append(rnn_val_loss)
        lstm_trn_loss_li.append(lstm_trn_loss)
        lstm_val_loss_li.append(lstm_val_loss)

        print(f'RNN - Epoch {epoch}, Train Loss: {rnn_trn_loss}, Validation Loss: {rnn_val_loss}')
        print(f'LSTM - Epoch {epoch}, Train Loss: {lstm_trn_loss}, Validation Loss: {lstm_val_loss}')

        if rnn_val_loss < rnn_best:
            rnn_best = rnn_val_loss
            torch.save(rnn_model.state_dict(), './rnn_best_model.pkl')

        if lstm_val_loss < lstm_best:
            lstm_best = lstm_val_loss
            torch.save(lstm_model.state_dict(), './lstm_best_model.pkl')
    
    #Plot loss and accuracy
    plt.figure(figsize=(12,8))
    plt.plot(rnn_trn_loss_li, label = 'RNN training loss')
    plt.plot(rnn_val_loss_li, label = 'RNN validation loss')
    plt.plot(lstm_trn_loss_li, label = 'LSTM training loss')
    plt.plot(lstm_val_loss_li, label = 'LSTM validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./loss_plot.png')
    

if __name__ == '__main__':
    main()