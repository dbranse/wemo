from tccp import search_conversation

with open('collected_convos.txt', 'a') as f:
    for conversation in search_conversation({"l": "en"}, continue_path="search_history.tmp"):
        if len(conversation) != 3:
            continue
        t1 = conversation[0]['contents'].replace('\t', ' ').replace('\n', ' ')
        a1 = conversation[0]['author']
        t2 = conversation[1]['contents'].replace('\t', ' ').replace('\n', ' ')
        a2 = conversation[1]['author']
        t3 = conversation[2]['contents'].replace('\t', ' ').replace('\n', ' ')
        a3 = conversation[2]['author']
        if a1 != a3 or a2 == a1 or len(t1) == 0 or len(t2) == 0 or len(t3) == 0:
            continue
        f.write('{}\t{}\t{}\n'.format(t1, t2, t3))