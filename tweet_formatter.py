from tweet_collector import hashtags
from random import random
import pickle
import re
import string

max_split = 0
padding = '<pad>'
formatting = '{num}|{seq}|{l}|{lab}\n'
lines = []
test_num = 0
dev_num = 0
percent_dev = 0.80
label_map = {"other":0, "happy":1, "sad":2, "angry":3}
smile_emoji = set([padding, ''])
for hashtag, emotions in hashtags.items():
    for e in emotions:
        with open(e + '.txt', 'r+') as f:
            for line in f:
                s = line.split(' ')
                length = len(s)
                lines.append((s, length, label_map[hashtag]))
                max_split = max(max_split, length)

f_dev = open('train_data.txt', 'a')
f_test = open('dev_data.txt', 'a')
f_dict1 = open('word2idx.dict', 'wb')
f_dict2 = open('idx2word.dict', 'wb')
f_dev.seek(0)
f_test.seek(0)
f_dev.truncate()
f_test.truncate()
f_dev.write('|sentence|sequence_length|sentiment_label\n')
f_test.write('|sentence|sequence_length|sentiment_label\n')
for line, length, label in lines:
    if random() > percent_dev:
        file = f_test
        num = test_num
        test_num += 1
    else:
        file = f_dev
        num = dev_num
        dev_num += 1
    line = ' '.join(line + [padding]*(max_split - length)).replace('|', '')
    repeatedChars = ['.', '?', '!', ',', "'"]
    for c in repeatedChars:
        lineSplit = line.split(c)
        while True:
            try:
                lineSplit.remove('')
            except:
                break
        cSpace = ' ' + c + ' '    
        line = cSpace.join(lineSplit)
    line = line.strip().split('\t')
    conv = ' '.join(line)
    # Remove any duplicate spaces
    duplicateSpacePattern = re.compile(r'\ +')
    conv = re.sub(duplicateSpacePattern, ' ', conv)
    for word in conv.split(' '):
        smile_emoji.add(word)
    file.write(formatting.format(num=num, 
                                 seq=conv,
                                 l=length,
                                 lab=label))
pickle.dump({word:i for i, word in enumerate(smile_emoji)}, f_dict1, pickle.HIGHEST_PROTOCOL)
pickle.dump({i:word for i, word in enumerate(smile_emoji)}, f_dict2, pickle.HIGHEST_PROTOCOL)
f_dict1.close()
f_dict2.close()
f_dev.close()
f_test.close()
print(test_num, dev_num)
