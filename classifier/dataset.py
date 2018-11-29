""" This file defines a pytorch Dataset class for loading our SST-5 data, and applying
    our specific preprocessing (mapping word tokens to integer ids)
"""

from os import path
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    """
    This is the Dataset class for this net. When initializing the dataset we
    specify mode "train", "dev", or "test". We will be using the 5-class Stanford Sentiment Treebank (SST-5).
    """
    def __init__(self, data_dir, mode):
        """
        This is the constructor of the dataset
        Args:
            :param data_dir (str): Path to the directory containing your data files.
            :param mode (str): This determines which dataset to load (train, dev or test).
        """
        data = pd.read_csv(path.join(data_dir, mode+"_data.txt"), delimiter='|')
        word2idx = np.load(path.join(data_dir, "word2idx.dict"))
        sentences = [sentence.split(' ') for sentence in data['sentence']]
        self.data_len = data['sequence_length'].values
        self.labels = data['sentiment_label'].values
        self.sequence_data = np.array([[word2idx[w] for w in s] for s in sentences])

    def __len__(self):
        """ Returns the length of the data (number of sentences) """
        return self.sequence_data.shape[0]

    def __getitem__(self, idx):
        """ Given an index, returns the vectorized sentence, the original length
            of that sentence, and the sentiment label.
        """
        return (self.sequence_data[idx], self.data_len[idx]), self.labels[idx]
