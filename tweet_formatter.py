from tweet_collector import hashtags
from random import random
import pickle

max_split = 0
padding = '<pad>'
formatting = '{num}|{seq}|{l}|{lab}\n'
lines = []
test_num = 0
dev_num = 0
percent_dev = 0.80
label_map = {"other":0, "happy":1, "sad":2, "angry":3}
smile_emoji = set()
for hashtag, emotions in hashtags.items():
	for e in emotions:
		with open(e + '.txt', 'r+') as f:
			for line in f:
				s = line.split(' ')
				length = len(s)
				lines.append((s, length, label_map[hashtag]))
				max_split = max(max_split, length)
				for word in s:
					smile_emoji.add(word)

f_dev = open('twitter_train.txt', 'a')
f_test = open('twitter_dev.txt', 'a')
f_dict1 = open('words2ids.dict', 'wb')
f_dict2 = open('ids2words.dict', 'wb')

for line, length, label in lines:
	if random() > percent_dev:
		file = f_test
		num = test_num
		test_num += 1
	else:
		file = f_dev
		num = dev_num
		dev_num += 1
	added_padding = ' '.join(line[:-1] + [padding]*(max_split - length)).replace('|', '')
	file.write(formatting.format(num=num, 
								 seq=added_padding,
								 l=length,
								 lab=label))
pickle.dump({word:i for i, word in enumerate(smile_emoji)}, f_dict1, pickle.HIGHEST_PROTOCOL)
pickle.dump({i:word for i, word in enumerate(smile_emoji)}, f_dict2, pickle.HIGHEST_PROTOCOL)
f_dict1.close()
f_dict2.close()
f_dev.close()
f_test.close()
print(test_num, dev_num)
