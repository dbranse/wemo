from tweet_collector import hashtags
from random import random

max_split = 0
padding = '<pad>'
formatting = '{num}|{seq}|{l}|{lab}\n'
lines = []
test_num = 0
dev_num = 0
percent_dev = 0.80
label_map = {"other":0, "happy":1, "sad":2, "angry":3}

for hashtag, emotions in hashtags.items():
	for e in emotions:
		with open(e + '.txt', 'r+') as f:
			for line in f:
				s = line.split(' ')
				length = len(s)
				lines.append((s, length, label_map[hashtag]))
				max_split = max(max_split, length)

f_dev = open('twitter_train.txt', 'a')
f_test = open('twitter_dev.txt', 'a')
for line, length, label in lines:
	if random() > percent_dev:
		file = f_test
		num = test_num
		test_num += 1
	else:
		file = f_dev
		num = dev_num
		dev_num += 1
	added_padding = ' '.join(line + [padding]*(max_split - length))
	file.write(formatting.format(num=num, 
								 seq=added_padding,
								 l=length,
								 lab=label))

f_dev.close()
f_test.close()