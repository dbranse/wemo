""" This file defines modules for looking up embeddings given word ids. """

from os import path
import pickle

import numpy as np
import torch
from torch import nn

import allennlp.modules.elmo as allen_elmo

# These are the different embedding sizes. Feel free to experiment
# with different sizes for random.
sizes = {"elmo": 1024, "glove": 200, "random": 10}
sizes["both"] = sizes["elmo"] + sizes["glove"]

class Elmo(nn.Module):
    """ TODO: Finish implementing __init__, forward, and _get_charids for Elmo embeddings.
        Take a look at the Allen AI documentation on using Elmo:
            https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
        In particular, reference the section "Using ELMo as a PyTorch Module".
        In addition, the Elmo model documentation may be useful:
            https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L34
    """

    def __init__(self, idx2word, device=torch.device('cpu')):
        """ Load the ELMo model. The first time you run this, it will download a pretrained model. """
        super(Elmo, self).__init__()
        options = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weights = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        self.elmo = allen_elmo.Elmo(options, weights, 1) # initialise an allen_elmo.Elmo model

        self.idx2word = idx2word

        self.embed_size = sizes["elmo"]
        self._dev = device

    def forward(self, batch):
        char_ids = self._get_charids(batch).to(self._dev)
        embeddings = self.elmo(char_ids)
        return embeddings['elmo_representations'][0]

    def _get_charids(self, batch):
        """ Given a batch of sentences, return a torch tensor of character ids.
                :param batch: List of sentences - each sentence is a list of int ids
            Return:
                torch tensor on self._dev
        """
         # 1. Map each sentence in batch to a list of string tokens (hint: use idx2word)
        batch = [[self.idx2word[id] for id in sentence] for sentence in batch]
        # 2. Use allen_elmo.batch_to_ids to convert sentences to character ids.
        return allen_elmo.batch_to_ids(batch)

class Glove(nn.Module):
    def __init__(self, data_dir, idx2word, device=torch.device('cpu')):
        """ TODO load pre-trained GloVe embeddings from disk """
        super(Glove, self).__init__()
        # 1. Load glove.6B.200d.npy from inside data_dir into a numpy array
        #    (hint: np.load)
        self.glove_vecs = np.load(path.join(data_dir, "glove.6B.200d.npy"))
        # 2. Load glove_tok2id.dict from inside data_dir. This is used to map
        #    a word token (as str) to glove vocab id (hint: pickle.load)
        with open(path.join(data_dir, "glove_tok2id.dict"), "rb") as f:
            self.glove_vocab = pickle.load(f)

        self.idx2word = idx2word
        self.embed_size = sizes["glove"]

        # 3. Create a torch tensor of the glove vectors and construct a
        #    a nn.Embedding out of it (hint: see how RandEmbed does it)
        self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(self.glove_vecs))

        self._dev = device

    def _lookup_glove(self, word_id):
        # given a word_id, convert to string and get glove id from the string:
        # unk if necessary.
        return self.glove_vocab.get(self.idx2word[word_id].lower(), self.glove_vocab["unk"])

    def _get_gloveids(self, batch):
        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.apply_
        return batch.apply_(self._lookup_glove).to(self._dev)

    def forward(self, batch):
        return self.embeddings(self._get_gloveids(batch))

class ElmoGlove(nn.Module):
    def __init__(self, data_dir, idx2word, device=torch.device('cpu')):
         """ Construct Elmo and Glove lookup instances """
         super(ElmoGlove, self).__init__()

         self.elmo = Elmo(idx2word, device=device)
         self.glove = Glove(data_dir, idx2word, device=device)
        
         self.embed_size = sizes["both"]
         self._dev = device

    def forward(self, batch):
        """ Concatenate ELMo and GloVe embeddings together """
        return torch.cat((self.elmo.forward(batch), self.glove.forward(batch)), 2)

class RandEmbed(nn.Module):
    def __init__(self, vocab_size, device=torch.device('cpu')):
        super(RandEmbed, self).__init__()
        self.embed_size = sizes["random"]
        self._dev = device

        self.embeddings = nn.Embedding.from_pretrained(torch.rand(vocab_size, self.embed_size))

    def forward(self, batch):
        batch = batch.to(self._dev)
        return self.embeddings(batch)

