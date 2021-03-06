from tweet_collector import hashtags
from random import random
import pickle
import re
import string
import json

files = ['FoxNews', 'cnn', 'ESPN', 'nytimes', 'time', 'HuffPostWeirdNews', 'theguardian', 'CartoonNetwork', \
'CookingLight', 'homecookingadventure', 'JustinBieber', 'nickelodeon', 'spongebob', 'Disney']

happy_threshold = 0.3  # haha, love
sad_threshold = 0.3 # sad
angry_threshold = 0.3 # angry
negative_threshold = 0.3 # non-like emojis outside of class should be less than this percent
reaction_dict = {'total': 0, 'like': 1, 'love': 2, 'haha': 3, 'wow': 4, 'sad': 5, 'angry': 6, 'thankful': 7}

max_split = 0
padding = '<pad>'
formatting = '{seq}|{l}|{lab}\n'
eval_formatting = '{seq1}|{l1}|{seq3}|{l3}|{lab}\n'
lines = []
percent_dev = 0.80
label_map = {"other":0, "others":0, "happy":1, "sad":2, "angry":3}
word_set = set([padding, ''])
for hashtag, emotions in hashtags.items():
    for e in emotions:
        with open(e + '.txt', 'r+') as f:
            for line in f:
                s = line.split(' ')
                lines.append((s, label_map[hashtag]))

f_dev = open('train_data.txt', 'a')
f_test = open('dev_data.txt', 'a')
f_dict1 = open('word2idx.dict', 'wb')
f_dict2 = open('idx2word.dict', 'wb')
comp_data = open('competitionTrain.txt', 'r')
test_data = open('test_data.txt', 'a')
fb_train = open('fb_train_data.txt', 'a')
fb_dev = open('fb_dev_data.txt', 'a')


f_dev.seek(0)
f_test.seek(0)
f_dev.truncate()
f_test.truncate()
test_data.truncate(0)
fb_train.truncate(0)
fb_dev.truncate(0)
f_dev.write('sentence|sequence_length|sentiment_label\n')
f_test.write('sentence|sequence_length|sentiment_label\n')
test_data.write('turn1|seq1_length|turn3|seq3_length|sentiment_label\n')
fb_train.write('sentence|sequence_length|sentiment_label\n')
fb_dev.write('sentence|sequence_length|sentiment_label\n')

convs = []
for line, label in lines:
    line = ' '.join(line[:-1]).replace('|', '').replace('\"','')
    repeatedChars = ['.', '?', '!', ',']#, "'", "\""]
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
    length = len(conv.split(' '))
    max_split = max(max_split, length)
    convs.append((conv, length, label))

message_data = []
for file in files:
    filename = 'fb-reaction-json/' + file + '.json'
    with open(filename, 'r+') as f:
        data = json.loads(f.read())
        for post in data:
            post_dict = post[0]
            message = post_dict['message']
            reactions = post_dict['reactions']
            label = 'others'
            # Order of reaction values: total, like, love, haha, wow, sad, angry, thankful
            total_count = reactions[reaction_dict['total']]
            if total_count > 0:
                happy_count = reactions[reaction_dict['haha']]+reactions[reaction_dict['love']]
                sad_count = reactions[reaction_dict['sad']]
                angry_count = reactions[reaction_dict['angry']]
                total_emotion_count = happy_count+sad_count+angry_count
                if (happy_count/total_count) > happy_threshold and (sad_count + angry_count) / total_emotion_count < negative_threshold:
                    label = 'happy'
                elif (sad_count/total_count) > sad_threshold and (happy_count + angry_count) / total_emotion_count < negative_threshold:
                    label = 'sad'
                elif (angry_count/total_count) > angry_threshold and (happy_count + sad_count) / total_emotion_count < negative_threshold:
                    label = 'angry'
            label = label_map[label]

            message = message.replace('|', '').replace('\"','').replace('\n', ' ').replace('\r', ' ')
            s = message.split(' ')
            length = len(s)
            if length < 150: # Cut off messages >150
                max_split = max(max_split, length)
                for w in s:
                    word_set.add(w)

                message_data.append((message, length, label))

next(comp_data) # Skip first line
for line in comp_data: #find max length and add words from test data
    for turn in [1,3]:  # Loop over turn 1 and turn 3
        s = line.split('\t')[turn].replace('|', '').replace('\"','')
        s = s.split(' ')
        length = len(s)
        max_split = max(max_split, length)
        for w in s:
            word_set.add(w)

for conv, length, label in convs:
    if random() > percent_dev:
        file = f_test
    else:
        file = f_dev
    for word in conv.split(' '):
        word_set.add(word)
    if length != max_split:
        conv += ' ' + ' '.join([padding]*(max_split - length))
    file.write(formatting.format(seq=conv,
                                 l=length,
                                 lab=label))

for (message, length, label) in message_data:
    if random() > percent_dev:
        file = fb_dev
    else:
        file = fb_train
    padded_message = message
    if length != max_split:
        padded_message += ' ' + ' '.join([padding]*(max_split - length))
    file.write(formatting.format(seq=padded_message,
                                 l=length,
                                 lab=label))
    
comp_data.seek(0)
next(comp_data)
for line in comp_data:
    s = line.split('\t')

    #1
    turn1 = s[1].replace('|', '').replace('\"','')
    length1 = len(turn1.split(' '))
    paddedTurn1 = turn1
    if length1 < max_split:
        paddedTurn1 += ' ' + ' '.join([padding]*(max_split - length1))
    #3
    turn3 = s[3].replace('|', '').replace('\"','')
    length3 = len(turn3.split(' '))
    paddedTurn3 = turn3
    if length3 < max_split:
        paddedTurn3 += ' ' + ' '.join([padding]*(max_split - length3))


    label = label_map[s[4].replace('\n', '')]
    test_data.write(eval_formatting.format(seq1=paddedTurn1,l1=length1,seq3=paddedTurn3,l3=length3,lab=label))

pickle.dump({word:i for i, word in enumerate(word_set)}, f_dict1, pickle.HIGHEST_PROTOCOL)
pickle.dump({i:word for i, word in enumerate(word_set)}, f_dict2, pickle.HIGHEST_PROTOCOL)
f_dict1.close()
f_dict2.close()
f_dev.close()
f_test.close()
comp_data.close()
test_data.close()
fb_train.close()
fb_dev.close()
