import json

files = ['FoxNews', 'CNN', 'ESPN', 'nytimes', 'time', 'HuffPostWeirdNews', 'theguardian', 'CartoonNetwork', \
'CookingLight', 'homecookingadventure', 'JustinBieber', 'nickelodeon', 'spongebob', 'Disney']

happy_threshold = 0.5 # haha, love
sad_threshold = 0.5 # sad
angry_threshold = 0.5 # angry
reaction_dict = {'total': 0, 'like': 1, 'love': 2, 'haha': 3, 'wow': 4, 'sad': 5, 'angry': 6, 'thankful': 7}

formatting = '{seq}|{l}|{lab}\n'
label_map = {"others":0, "happy":1, "sad":2, "angry":3}

fb_train = open('fb_train.txt', 'a')
fb_train.truncate(0)
fb_train.write('sentence|sequence_length|sentiment_label\n')

max_length = 0
padding = '<pad>'
word_set = set([padding, ''])

message_data = []

for file in files:
	filename = 'fb-reaction-json/' + file + '.json'
	with open(filename, 'r+') as f:
		data = json.loads(f.read())
		for post in data:
			post_dict = post[0]
			message = post_dict['message']
			reactions = post_dict['reactions']
			label = label_map['others']
			# Order of reaction values: total, like, love, haha, wow, sad, angry, thankful
			if reactions[reaction_dict['total']] > 0:
				if ((reactions[reaction_dict['haha']]+reactions[reaction_dict['love']])/reactions[reaction_dict['total']]) > happy_threshold:
					label = label_map['happy']
				elif (reactions[reaction_dict['sad']]/reactions[reaction_dict['total']]) > sad_threshold:
					label = label_map['sad']
				elif (reactions[reaction_dict['angry']]/reactions[reaction_dict['total']]) > angry_threshold:
					label = label_map['angry']

			
			message = message.replace('|', '').replace('\"','')
			s = message.split(' ')
			length = len(s)
			max_length = max(max_length, length)
			for w in s:
				word_set.add(w)

			message_data.append((message, length, label))


for (message, length, label) in message_data:
	padded_message = message
	if length != max_length:
		padded_message += ' ' + ' '.join([padding]*(max_length - length))
	fb_train.write(formatting.format(seq=padded_message,
                                 l=length,
                                 lab=label))

fb_train.close()
