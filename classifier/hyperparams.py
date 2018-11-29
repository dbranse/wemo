""" This file defines hyperparameters for training. Any instance of a subclass
    of HParams will have the fields:
        batch_size
        learn_rate
        num_epochs
        rnn_hidden_size

    TODO: Some hyperparameters may need to vary depending on the embeddings used.
"""

class HParams:
    def __init__(self, bs, lr, ne, hs):
        self.batch_size = bs
        self.learn_rate = lr
        self.num_epochs = ne
        self.rnn_hidden_size = hs

class ElmoHParams(HParams):
    def __init__(self):
        super(ElmoHParams, self).__init__(bs=64, lr=1e-4, ne=5, hs=1024)

class GloveHParams(HParams):
    def __init__(self):
        super(GloveHParams, self).__init__(bs=64, lr=1e-4, ne=15, hs=1024)

class ElmoGloveHParams(HParams):
    def __init__(self):
        super(ElmoGloveHParams, self).__init__(bs=64, lr=1e-4, ne=5, hs=1024)

class RandEmbedHParams(HParams):
    def __init__(self, embed_size):
        super(RandEmbedHParams, self).__init__(bs=64, lr=1e-4, ne=10, hs=embed_size)

