# import some packages you need here
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import tarfile
import io
from PIL import Image
import torchvision.transforms as tr

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir, augmentation):
        self.data_dir = data_dir
        self.tf = tarfile.open(self.data_dir)
        self.img_li = self.tf.getnames()[1:]
        self.labels = []
        for i in self.img_li:
            self.labels.append(int(i[-5]))

        self.len = len(self.labels)
        if augmentation:
            self.transf = tr.Compose([tr.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),tr.Resize(32),tr.ToTensor(),tr.Normalize(mean = 0.1307, std = 0.3081)])
        else:
            self.transf = tr.Compose([tr.Resize(32),tr.ToTensor(),tr.Normalize(mean = 0.1307, std = 0.3081)])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        file_name = self.img_li[idx]
        img = self.tf.extractfile(file_name)
        img = img.read()
        img = Image.open(io.BytesIO(img))
        img = self.transf(img)
        label = self.labels[idx]

        return img, label

if __name__ == '__main__':

    train_dataset = MNIST(data_dir = './train.tar', augmentation= True)
    train_loader = DataLoader(dataset = train_dataset, batch_size = 300, shuffle = True)
    batch_img, lbl = next(iter(train_loader))
    print(batch_img.shape)
    print(lbl.shape)
