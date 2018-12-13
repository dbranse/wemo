import csv
with open('newLabeledTrain.txt', 'r') as tsv, open('newLabeledTrainOtherOnly.txt', 'w') as tsv2:
  c = csv.DictReader(tsv, delimiter='\t')
  m = 0
  ls = []
  a = []
  ctr = 0
  for i in c:
    if i['turn1'] is None or i['turn2'] is None or i['turn3'] is None:
      continue
    k = len(i['turn1'].split()) + len(i['turn2'].split()) + len(i['turn3'].split()) + 2
    m = max(m, k)
    ls.append(k)
    if i['label'] == 'others':
      i['id'] = ctr
      ctr += 1
      a.append(i)
  print(m)
  print(sum(ls)/len(ls))
  dict_writer = csv.DictWriter(tsv2, a[0].keys(), delimiter='\t')
  dict_writer.writeheader()
  dict_writer.writerows(a)
