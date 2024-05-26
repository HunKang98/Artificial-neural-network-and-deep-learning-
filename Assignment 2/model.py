import torch.nn as nn
import torch

class CharRNN(nn.Module):
    def __init__(self):
        
        # write your codes here
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(62,64)
        self.rnn = nn.RNN(input_size=64, hidden_size=64, num_layers=2, bias=True, batch_first=True)
        self.fc = nn.Linear(64, 62)

    def forward(self, input, hidden):

        # write your codes here
        input_embed = self.embedding(input)
        rnn_output, hidden = self.rnn(input_embed, hidden)
        output = self.fc(rnn_output)

        return output, hidden

    def init_hidden(self, batch_size):

        # write your codes here
        initial_hidden = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size)
        return initial_hidden


class CharLSTM(nn.Module):
    def __init__(self):

        # write your codes here
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(62,64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, bias=True, batch_first=True)
        self.fc = nn.Linear(64, 62)

    def forward(self, input, hidden):

        # write your codes here
        input_embed = self.embedding(input)
        lstm_output, hidden = self.lstm(input_embed, hidden)
        output = self.fc(lstm_output)

        return output, hidden

    def init_hidden(self, batch_size):

        # write your codes here
        initial_hidden = (
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size),
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size)
        )

        return initial_hidden