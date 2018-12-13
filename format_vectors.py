with open('vectors.txt', 'r') as vecs, open('vocab.txt', 'r') as vocab, open('formattedSentiment.txt', 'w') as outp:
  for voc, vec in zip(vocab, vecs):
    outp.write('{} {}\n'.format(voc.rstrip(), vec.rstrip()))

