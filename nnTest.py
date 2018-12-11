import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


with open('vectors.txt', 'r') as vecs:
	modified_vecs = [np.asarray(line.split(' '), dtype=np.float64) for line in vecs]

with open('originalVectors.txt', 'r') as vecs:
	original_vecs = [np.asarray(line.split(' '), dtype=np.float64) for line in vecs]

with open('vocab.txt', 'r') as vocab:
	id2word = {id:word.replace('\n', '') for (id, word) in enumerate(vocab)}
	word2id = {word:id for (id, word) in id2word.items()}

vecs = [modified_vecs, original_vecs]
words = ['good', 'food', 'cool']
for vec in vecs:
	nbrs = NearestNeighbors(8, metric='cosine').fit(vec)
	distances, indices = nbrs.kneighbors([vec[word2id[word]] for word in words])
	for id,word in enumerate(words):
		print("Word: " + word)
		for index in indices[id]:
			print(id2word[index])
		print('\n')
	print('\n')

