

comp_label_file = 'competitionTrain.txt'
weak_label_file = 'labeledTrain.txt'
formatting = '{id}\t{turn1}\t{turn2}\t{turn3}\t{label}\t{source}\n'

source_map = {0: "competition", 1: "weak"}

with open('mergedTrain.txt', 'a') as f, open(comp_label_file, 'r') as comp, open(weak_label_file, 'r') as weak:
	f.truncate(0)
	f.write(formatting.format(id = 'id', turn1 = 'turn1', turn2 = 'turn2', turn3 = 'turn3', label = 'label', source = 'source'))

	sources = [comp, weak]
	counter = 0
	for index, source in enumerate(sources):
		next(source) # skip header
		for line in source:
			line = line.replace('\n','').split('\t')
			f.write(formatting.format(id=counter,turn1=line[1],turn2=line[2],turn3=line[3],label=line[4],source=source_map[index]))
			counter += 1

