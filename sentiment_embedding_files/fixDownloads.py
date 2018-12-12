import string

formatting = '{SID}\t{UID}\t{emote}\t{message}\n'
with open('downloaded_tweets.tsv', 'r') as tweets:
	with open('tweets.tsv', 'a') as file:
		for line in tweets:
			s = line.split('\t')
			file.write(formatting.format(SID=s[0], UID=s[1], emote=s[2].replace('\"',''), message=s[3]))
