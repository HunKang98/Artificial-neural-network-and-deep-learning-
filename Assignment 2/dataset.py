#import some packages you need here
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
		You need this dictionary to generate characters.
		2) Make list of character indices using the dictionary
		3) Split the data into chunks of sequence length 30. 
        You should create targets appropriately.
    """

    def __init__(self, input_file):

        # write your codes here
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        self.sequence_length = 30
        self.unique_char = sorted(list(set(text)))
        self.char_idx = {c:idx for idx,c in enumerate(self.unique_char)}
        self.idx_char = {idx:c for idx,c in enumerate(self.unique_char)}
        self.data_idx = [self.char_idx[t] for t in text]
        self.num_seq = len(self.data_idx) // self.sequence_length

    def __len__(self):

        # write your codes here
        return self.num_seq - 1

    def __getitem__(self, idx):

        # write your codes here
        start_idx = idx*self.sequence_length
        end_idx = start_idx + self.sequence_length

        input_idx = self.data_idx[start_idx:end_idx]
        target_idx = self.data_idx[start_idx+1:end_idx+1]

        input = torch.tensor(input_idx)
        target = torch.tensor(target_idx)

        return input, target

if __name__ == '__main__':

    # write test codes to verify your implementations
    dataset = Shakespeare('shakespeare_train.txt')
    data_loader = DataLoader(dataset, batch_size=100, shuffle=True)
    batch_input, batch_target = next(iter(data_loader))
    print(batch_input.shape)
    print(batch_target.shape)
