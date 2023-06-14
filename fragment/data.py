import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import lmdb
import pickle as pkl 
from typing import Union
from pathlib import Path



#Data
class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """
    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False)->None:

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples
        
    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item
        
class PRDataset(Dataset):
    def __init__(self, phase, data_path):
        super(PRDataset, self).__init__()
        
        self.phase = phase
        self.data_path = data_path

        self.data = LMDBDataset(data_path)

    def __getitem__(self, index):

        data_dict = self.data.__getitem__(index)

        collision_energy = data_dict['collision_energy_aligned_normed']
        precursor_charge_onehot = data_dict['precursor_charge_onehot']
        peptide_sequence = self.string_to_array(data_dict['peptide_sequence'])
        intensities_raw = data_dict['intensities_raw']

        precursor_charge = torch.argmax(torch.from_numpy(precursor_charge_onehot), dim=-1).unsqueeze(-1)
        collision_energy = torch.from_numpy(collision_energy).unsqueeze(-1)
        peptide_sequence = torch.from_numpy(peptide_sequence)
        target = torch.from_numpy(intensities_raw)

        sample = {
                  'precursor_charge':precursor_charge, 'collision_energy':collision_energy,
                  'peptide_sequence':peptide_sequence, 'label':target
                  }

        return sample

    def __len__(self):
        return self.data.__len__()

    def string_to_array(self, sequence):
        ALPHABET = {
              "A": 1,
              "C": 2,
              "D": 3,
              "E": 4,
              "F": 5,
              "G": 6,
              "H": 7,
              "I": 8,
              "K": 9,
              "L": 10,
              "M": 11,
              "N": 12,
              "P": 13,
              "Q": 14,
              "R": 15,
              "S": 16,
              "T": 17,
              "V": 18,
              "W": 19,
              "Y": 20,
              "X": 21,
          }
        # Initialize the array with zeros
        result = np.zeros(30, dtype=int)
        
        # Convert each character in the string to its corresponding integer value
        for i, char in enumerate(sequence):
            if i >= 30:
                break
            result[i] = ALPHABET.get(char, 0)
        
        return result

