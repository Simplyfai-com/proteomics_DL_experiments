import torch
import numpy as np
from torch.utils.data.dataset import Dataset


class PRDataset(Dataset):
    def __init__(self, phase, data_path, label_path):
        super(PRDataset, self).__init__()
        
        self.phase = phase
        self.data_path = data_path
        self.label_path = label_path
        
        data = np.load(self.data_path)
        label = np.load(self.label_path)

        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label).unsqueeze(1)

    def __getitem__(self, index):

        x = self.data[index]
        y = self.label[index]

        sample = {'data':x, 'label':y}

        return sample

    def __len__(self):
        return self.data.shape[0]

