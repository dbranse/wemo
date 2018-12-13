import csv
with open('train_data.txt', 'r') as tsv, open('newLabeledTrainWithTriples.txt', 'w') as tsv2:
  c = csv.DictReader(tsv, delimiter='|')
  m = 0
  ls = []
  a = []
  for_u = {0:'others', 1:'happy', 2:'sad', 3:'angry'}
  for i in c:
    no_pad = ' '.join(filter(lambda x: x!='<pad>', i['sentence'].split()))
    if len(no_pad.split())*3 + 2 <= 100:
      newd = {'id':m, 'turn1':no_pad, 'turn2':no_pad, 'turn3':no_pad, 'label':for_u[int(i['sentiment_label'])]}
      a.append(newd)
      m += 1
  print(m)
  dict_writer = csv.DictWriter(tsv2, a[0].keys(), delimiter='\t')
  dict_writer.writeheader()
  dict_writer.writerows(a)
